//! Focused Vulkan GPU checks for mesh dispatch and its raster bindings.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::{future::Future, sync::Arc};

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
    exercise(false, false);
}
#[test]
fn mesh_indirect_depth_readback() {
    exercise(true, false);
}
#[test]
fn task_mesh_depth_readback() {
    exercise(false, true);
}
fn exercise(indirect: bool, task: bool) {
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
    let ps = shader(
        include_bytes!("assets/mesh/pixel.spv"),
        "main",
        ShaderStage::Fragment,
    );
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
            blend: None,
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
            depth_compare: CompareFunction::Less,
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
            size: 4096,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let args = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 12,
            usage: BufferUsage::INDIRECT.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })
        .unwrap();
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
    e.write_buffer(&args, 0, &[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        .unwrap();
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &args,
            range: 0..12,
        },
        usage: StateTransition {
            from: BufferUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            },
            to: BufferUses {
                stages: StageMask::INDIRECT,
                accesses: AccessMask::INDIRECT_READ,
            },
        },
        ownership: None,
    })])
    .unwrap();
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
        if indirect {
            mesh.dispatch_indirect(&mut pass, &args, 0).unwrap();
        } else {
            mesh.dispatch(&mut pass, [1, 1, 1]).unwrap();
        }
        pass.finish().unwrap();
    }
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
    let ticket = e.readback_buffer(&dst, 0..4096).unwrap();
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
    assert_eq!(&bytes[center..center + 4], &[255, 0, 0, 255]);
}
