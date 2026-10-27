//! Focused Vulkan GPU checks for mesh dispatch and its raster bindings.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::{Vulkan, VulkanInstance};
use std::{
    future::Future,
    sync::{Arc, OnceLock},
};

fn block_on<F: Future>(f: F) -> F::Output {
    let w = std::task::Waker::noop();
    let mut c = std::task::Context::from_waker(w);
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut c) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

#[test]
fn mesh_direct_depth_readback() {
    exercise(Dispatch::Direct, false);
}
#[test]
fn mesh_indirect_depth_readback() {
    exercise(Dispatch::Indirect, false);
}
#[test]
fn mesh_multi_indirect_depth_readback() {
    exercise(Dispatch::Multi, false);
}
#[test]
fn mesh_indirect_count_depth_readback() {
    exercise(Dispatch::Count, false);
}
#[test]
fn mesh_indirect_count_below_max_depth_readback() {
    exercise(Dispatch::CountOne, false);
}
#[test]
fn mesh_indirect_count_zero_depth_readback() {
    exercise(Dispatch::CountZero, false);
}
#[test]
fn mesh_indirect_count_clamp_depth_readback() {
    exercise(Dispatch::CountClamped, false);
}
#[test]
fn task_mesh_depth_readback() {
    exercise(Dispatch::Direct, true);
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dispatch {
    Direct,
    Indirect,
    Multi,
    Count,
    CountOne,
    CountZero,
    CountClamped,
}
impl Dispatch {
    fn generated(self) -> bool {
        matches!(
            self,
            Self::Multi | Self::Count | Self::CountOne | Self::CountZero | Self::CountClamped
        )
    }
    fn count_value(self) -> u32 {
        match self {
            Self::Count => 2,
            Self::CountOne => 1,
            Self::CountZero => 0,
            Self::CountClamped => 3,
            _ => unreachable!("only count variants have a GPU count"),
        }
    }
}
fn exercise(dispatch: Dispatch, task: bool) {
    let d = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    if !d.capabilities().mesh {
        eprintln!("SKIP: mesh shader tier unavailable");
        return;
    }
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let shader = |bytes: &'static [u8], name: &str, stage| {
        block_on(d.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::SpirV,
            bytes: Arc::from(bytes),
            entries: vec![ShaderEntryPoint {
                name: name.into(),
                stage,
                workgroup_size: None,
            }],
            required_features: vec![],
        }))
        .unwrap()
    };
    let vs = shader(
        include_bytes!("assets/mesh/mesh_bound.spv"),
        "main",
        ShaderStage::Mesh,
    );
    let ps = if dispatch.generated() {
        fractional_pixel_shader(&d)
    } else {
        shader(
            include_bytes!("assets/mesh/pixel.spv"),
            "main",
            ShaderStage::Fragment,
        )
    };
    let task_shader = task.then(|| {
        shader(
            include_bytes!("assets/mesh/task.spv"),
            "main",
            ShaderStage::Task,
        )
    });
    let positions = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 48,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let group_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Mesh],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 48,
                    access: StorageAccess::Read,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::StorageBufferView {
                    buffer: &positions,
                    range: 0..48,
                    view: StorageBufferView::Structured { stride: 16 },
                },
            }],
        })
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[PushConstantRange {
                stages: vec![ShaderStage::Mesh],
                range: 0..4,
            }],
        })
        .unwrap();
    let face = StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    let mesh = d.mesh().unwrap();
    let pipeline = block_on(mesh.create_pipeline(MeshPipelineDesc {
        layout,
        task: task_shader.map(|module| PipelineStage {
            module,
            entry: "main".into(),
            constants: vec![],
        }),
        mesh: PipelineStage {
            module: vs,
            entry: "main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: ps,
            entry: "main".into(),
            constants: vec![],
        }),
        targets: vec![Some(ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
            blend: dispatch.generated().then_some(BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::Zero,
                    operation: BlendOperation::Add,
                },
            }),
            write_mask: 15,
        })],
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            // The multi/count fixture intentionally draws the same triangle
            // twice, so both records need to survive the depth test.
            depth_compare: if dispatch.generated() {
                CompareFunction::Always
            } else {
                CompareFunction::Less
            },
            stencil_front: face,
            stencil_back: face,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            depth_bias_constant: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        multisample: MultisampleState {
            count: 1,
            mask: u64::from(u32::MAX),
            alpha_to_coverage_enabled: false,
        },
        cache: None,
    }))
    .unwrap();
    let color = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap();
    let depth = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 1,
            format: TextureFormat::Depth32Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let views = |texture: &Texture<_>, format, aspect| {
        (0..1)
            .map(|layer| {
                d.create_view(
                    texture,
                    &TextureViewDesc {
                        format,
                        dimension: TextureViewDimension::D2,
                        aspects: aspect,
                        mips: 0..1,
                        layers: layer..layer + 1,
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let colors = views(&color, TextureFormat::Rgba8Unorm, TextureAspects::COLOR);
    let depths = views(&depth, TextureFormat::Depth32Float, TextureAspects::DEPTH);
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            // The first page holds the color readback. The second holds the
            // indirect source after the mesh pass, so the fixture can prove
            // that the unused trailing record is not clobbered.
            size: 8192,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let args = d
        .create_buffer(&BufferDesc {
            label: None,
            // The multi/count variants deliberately begin at a non-zero
            // record offset, leave a sentinel prefix untouched, and retain a
            // sentinel word after the two records.
            size: 44,
            usage: BufferUsage::INDIRECT
                .union(BufferUsage::COPY_DST)
                .union(BufferUsage::COPY_SRC)
                .union(BufferUsage::STORAGE),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let producer = dispatch.generated().then(|| {
        let group_layout = d
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: None,
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 44,
                        access: StorageAccess::ReadWrite,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                }],
            })
            .unwrap();
        let group = d
            .create_bind_group(&BindGroupDesc {
                label: None,
                layout: &group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::StorageBufferView {
                        buffer: &args,
                        range: 0..44,
                        view: StorageBufferView::Raw,
                    },
                }],
            })
            .unwrap();
        let layout = d
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: None,
                groups: &[&group_layout],
                push_constants: &[],
            })
            .unwrap();
        let count = if dispatch == Dispatch::Multi {
            0
        } else {
            dispatch.count_value()
        };
        let source = format!(
            "@group(0) @binding(0) var<storage, read_write> a: array<u32>; \
             @compute @workgroup_size(1) fn main() {{ \
             a[1] = {count}u; a[4] = 1u; a[5] = 1u; a[6] = 1u; \
             a[7] = 1u; a[8] = 1u; a[9] = 1u; }}"
        );
        let shader = block_on(d.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.into_bytes()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: vec![],
        }))
        .unwrap();
        let pipeline = block_on(d.compute().unwrap().create_pipeline(ComputePipelineDesc {
            label: None,
            layout,
            compute: PipelineStage {
                module: shader,
                entry: "main".into(),
                constants: vec![],
            },
            cache: None,
            required_subgroup_size: None,
        }))
        .unwrap();
        (group, pipeline)
    });
    {
        let mut invalid = d.encoder(lane).unwrap();
        let attachments = [Some(ColorAttachment {
            view: AttachmentView::Texture(&colors[0]),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
            store: StoreOp::Store,
        })];
        let mut pass = invalid
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &attachments,
                depth_stencil: None,
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 16,
                    height: 16,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        assert!(matches!(
            mesh.bind_pipeline(&mut pass, &pipeline),
            Err(Error::InvalidState)
        ));
    }
    if matches!(dispatch, Dispatch::Multi | Dispatch::Count) {
        let invalid_usage = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 12,
                usage: BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let mut negative = d.encoder(lane).unwrap();
        let attachments = [Some(ColorAttachment {
            view: AttachmentView::Texture(&colors[0]),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
            store: StoreOp::Store,
        })];
        let depth_attachment = || DepthStencilAttachment {
            view: AttachmentView::Texture(&depths[0]),
            depth: Some((LoadOp::Clear(1.0), StoreOp::Store)),
            stencil: None,
            depth_read_only: false,
            stencil_read_only: false,
        };
        let mut pass = negative
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &attachments,
                depth_stencil: Some(depth_attachment()),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 16,
                    height: 16,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        mesh.bind_pipeline(&mut pass, &pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        assert_eq!(
            mesh.dispatch_indirect(&mut pass, &invalid_usage, 0),
            Err(Error::InvalidRange),
            "the Vulkan native boundary must reject a non-INDIRECT argument buffer"
        );
        // A rejected backend command poisons this recording by design; drop
        // it instead of asserting that a poisoned pass can finish.
        drop(pass);
        let mut range_encoder = d.encoder(lane).unwrap();
        let mut range_pass = range_encoder
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &attachments,
                depth_stencil: Some(depth_attachment()),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 16,
                    height: 16,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        mesh.bind_pipeline(&mut range_pass, &pipeline).unwrap();
        range_pass.set_bind_group(0, &group, &[]).unwrap();
        assert_eq!(
            mesh.dispatch_indirect_multi(&mut range_pass, &args, 36, 1),
            Err(Error::InvalidRange),
            "the Vulkan native boundary must reject records extending past the buffer"
        );
        drop(range_pass);
        if dispatch == Dispatch::Count {
            let mut count_encoder = d.encoder(lane).unwrap();
            let mut count_pass = count_encoder
                .begin_raster(&RasterPassDesc {
                    label: None,
                    colors: &attachments,
                    depth_stencil: Some(depth_attachment()),
                    render_area: Rect {
                        x: 0,
                        y: 0,
                        width: 16,
                        height: 16,
                    },
                    multiview_mask: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    secondary_inheritance: None,
                })
                .unwrap();
            mesh.bind_pipeline(&mut count_pass, &pipeline).unwrap();
            count_pass.set_bind_group(0, &group, &[]).unwrap();
            assert_eq!(
                mesh.dispatch_indirect_count(&mut count_pass, &args, 16, &args, 44, 1),
                Err(Error::InvalidRange),
                "the Vulkan native boundary must reject count words extending past the buffer"
            );
            drop(count_pass);
        }
    }
    let mut e = d.encoder(lane).unwrap();
    let vertices: Vec<u8> = [-1.0f32, -1., 0., 1., 0., 1., 0., 1., 1., -1., 0., 1.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    e.write_buffer(&positions, 0, &vertices).unwrap();
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &positions,
            range: 0..48,
        },
        usage: StateTransition {
            from: BufferUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            },
            to: BufferUses {
                stages: StageMask::MESH,
                accesses: AccessMask::SHADER_READ,
            },
        },
        ownership: None,
    })])
    .unwrap();
    const TAIL_SENTINEL: u32 = 0xC0DE_CAFE;
    let initial_args = [1u32, 1, 1, 0, 0, 0, 0, 0, 0, 0, TAIL_SENTINEL];
    let initial_args: Vec<u8> = initial_args
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    e.write_buffer(&args, 0, &initial_args).unwrap();
    if let Some((group, pipeline)) = producer.as_ref() {
        e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &args,
                range: 0..44,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                to: BufferUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_WRITE,
                },
            },
            ownership: None,
        })])
        .unwrap();
        let mut pass = d.compute().unwrap().begin(&mut e).unwrap();
        pass.set_pipeline(pipeline).unwrap();
        pass.set_bind_group(0, group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &args,
            range: 0..44,
        },
        usage: StateTransition {
            from: BufferUses {
                stages: if dispatch.generated() {
                    StageMask::COMPUTE
                } else {
                    StageMask::COPY
                },
                accesses: if dispatch.generated() {
                    AccessMask::SHADER_WRITE
                } else {
                    AccessMask::COPY_WRITE
                },
            },
            to: BufferUses {
                stages: StageMask::INDIRECT,
                accesses: AccessMask::INDIRECT_READ,
            },
        },
        ownership: None,
    })])
    .unwrap();
    // The compute bind group is no longer an owner; later drop(args) leaves
    // submission as the only owner of the compute-to-indirect buffer.
    drop(producer);
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let render = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let copy = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let barrier = |from, to, layouts| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &color,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition { from, to },
            layout: Some(layouts),
            ownership: None,
        })
    };
    e.transition_resources(&[barrier(
        empty,
        render,
        (ImageLayout::Undefined, ImageLayout::ColorAttachment),
    )])
    .unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &depth,
            aspects: TextureAspects::DEPTH,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition {
            from: empty,
            to: TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::DEPTH_WRITE,
            },
        },
        layout: Some((ImageLayout::Undefined, ImageLayout::DepthStencilAttachment)),
        ownership: None,
    })])
    .unwrap();
    for layer in 0..1 {
        let attachments = [Some(ColorAttachment {
            view: AttachmentView::Texture(&colors[layer]),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        })];
        let depth_attachment = DepthStencilAttachment {
            view: AttachmentView::Texture(&depths[layer]),
            depth: Some((LoadOp::Clear(1.0), StoreOp::Store)),
            stencil: None,
            depth_read_only: false,
            stencil_read_only: false,
        };
        let mut pass = e
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &attachments,
                depth_stencil: Some(depth_attachment),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 16,
                    height: 16,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: 16.0,
            height: 16.0,
            min_depth: 0.0,
            max_depth: 1.0,
        })
        .unwrap();
        pass.scissor(Rect {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
        })
        .unwrap();
        mesh.bind_pipeline(&mut pass, &pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        d.raster_extras()
            .unwrap()
            .push_constants(&mut pass, StageMask::MESH, 0, &1.0f32.to_le_bytes())
            .unwrap();
        match dispatch {
            Dispatch::Direct => mesh.dispatch(&mut pass, [1, 1, 1]).unwrap(),
            Dispatch::Indirect => mesh.dispatch_indirect(&mut pass, &args, 0).unwrap(),
            Dispatch::Multi => mesh
                .dispatch_indirect_multi(&mut pass, &args, 16, 2)
                .unwrap(),
            Dispatch::Count | Dispatch::CountOne | Dispatch::CountZero | Dispatch::CountClamped => {
                mesh.dispatch_indirect_count(&mut pass, &args, 16, &args, 4, 2)
                    .unwrap()
            }
        }
        pass.finish().unwrap();
    }
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &args,
            range: 0..44,
        },
        usage: StateTransition {
            from: BufferUses {
                stages: StageMask::INDIRECT,
                accesses: AccessMask::INDIRECT_READ,
            },
            to: BufferUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        },
        ownership: None,
    })])
    .unwrap();
    e.copy_buffer(&args, 0, &dst, 4096, 44).unwrap();
    // No public owner remains before submission. VulkanEncoder must retain the
    // same buffer through the compute write, mesh indirect read, and sentinel
    // copy.
    drop(args);
    e.transition_resources(&[barrier(
        render,
        copy,
        (ImageLayout::ColorAttachment, ImageLayout::CopySrc),
    )])
    .unwrap();
    for layer in 0..1 {
        e.texture_to_buffer(
            TextureRegion {
                texture: &color,
                mip: 0,
                origin: [0, 0, layer],
                extent: [16, 16, 1],
                aspect: TextureAspects::COLOR,
            },
            &dst,
            TexelCopyBufferLayout {
                offset: u64::from(layer) * 4096,
                bytes_per_row: 256,
                rows_per_image: 16,
            },
        )
        .unwrap();
    }
    let ticket = e.readback_buffer(&dst, 0..8192).unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755321,
    };
    // SAFETY: the one-step plan uses this device's graphics lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: the work is consumed once on its recording lane.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![e.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block_on(d.wait_completion(&point)).unwrap();
    // SAFETY: the awaited completion point covers the readback.
    let bytes = unsafe { block_on(d.read_after(ticket, &point)).unwrap().bytes };
    let center = 8 * 256 + 8 * 4;
    let tail = u32::from_le_bytes(bytes[4096 + 40..4096 + 44].try_into().unwrap());
    assert_eq!(
        tail, TAIL_SENTINEL,
        "the record tail sentinel must survive compute production and mesh indirect execution"
    );
    if matches!(
        dispatch,
        Dispatch::Multi | Dispatch::Count | Dispatch::CountClamped
    ) {
        assert!(
            (120..=136).contains(&bytes[center]),
            "two 0.25-red mesh records must add to approximately 0.5, got {}",
            bytes[center]
        );
    } else if dispatch == Dispatch::CountOne {
        assert!(
            (56..=72).contains(&bytes[center]),
            "one 0.25-red mesh record must produce approximately 0.25, got {}",
            bytes[center]
        );
    } else if dispatch == Dispatch::CountZero {
        assert_eq!(&bytes[center..center + 4], &[0, 0, 0, 255]);
    } else {
        assert_eq!(&bytes[center..center + 4], &[255, 0, 0, 255]);
    }
}

fn fractional_pixel_shader(d: &Device<Vulkan>) -> Shader<Vulkan> {
    static PIXEL_SPIRV: OnceLock<Vec<u8>> = OnceLock::new();
    let bytes = PIXEL_SPIRV
        .get_or_init(|| {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "fluxel-mesh-indirect-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let source = dir.join("mesh_pixel.hlsl");
            let output = dir.join("mesh_pixel.spv");
            std::fs::write(
                &source,
                "float4 main() : SV_Target { return float4(0.25,0,0,1); }",
            )
            .unwrap();
            let result = std::process::Command::new("dxc")
                .args([
                    "-spirv",
                    "-fspv-target-env=vulkan1.2",
                    "-T",
                    "ps_6_0",
                    "-E",
                    "main",
                    "-Fo",
                ])
                .arg(&output)
                .arg(&source)
                .output()
                .expect("hardware test requires dxc in PATH");
            assert!(
                result.status.success(),
                "dxc: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            let bytes = std::fs::read(&output).unwrap();
            let _ = std::fs::remove_dir_all(&dir);
            bytes
        })
        .clone();
    block_on(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from(bytes),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Fragment,
            workgroup_size: None,
        }],
        required_features: vec![],
    }))
    .unwrap()
}
