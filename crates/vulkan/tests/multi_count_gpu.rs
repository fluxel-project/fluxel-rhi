//! GPU-generated indirect draw count, including aliased argument/count buffers.
#![cfg(all(windows, feature = "hardware-tests"))]
#[path = "support/mod.rs"]
mod support;
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::sync::Arc;
use support::block;
#[test]
fn gpu_counts_aliased() {
    gpu_counts_are_clamped(true, false);
}
#[test]
fn gpu_counts_separate() {
    gpu_counts_are_clamped(false, false);
}
#[test]
fn gpu_indexed_counts_aliased() {
    gpu_counts_are_clamped(true, true);
}
#[test]
fn gpu_indexed_counts_separate() {
    gpu_counts_are_clamped(false, true);
}
fn gpu_counts_are_clamped(aliased: bool, indexed: bool) {
    let d = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|x| x.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let shader = |bytes: &'static [u8], stage| {
        block(d.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::SpirV,
            bytes: Arc::from(bytes),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage,
                workgroup_size: None,
            }],
            required_features: vec![],
        }))
        .unwrap()
    };
    let args = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 80,
            usage: BufferUsage::STORAGE
                | BufferUsage::INDIRECT
                | BufferUsage::COPY_DST
                | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let counts = if aliased {
        args.clone()
    } else {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 80,
            usage: BufferUsage::STORAGE | BufferUsage::INDIRECT,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let group_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 80,
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
                    buffer: &counts,
                    range: 0..80,
                    view: StorageBufferView::Raw,
                },
            }],
        })
        .unwrap();
    let cl = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();
    let cp = block(d.compute().unwrap().create_pipeline(ComputePipelineDesc {
        label: None,
        layout: cl,
        compute: PipelineStage {
            module: shader(
                include_bytes!("assets/multi_count/count.spv"),
                ShaderStage::Compute,
            ),
            entry: "main".into(),
            constants: vec![],
        },
        cache: None,
        required_subgroup_size: None,
    }))
    .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let rp = block(d.raster_pipeline(RasterPipelineDesc {
        label: None,
        layout,
        vertex: PipelineStage {
            module: shader(
                include_bytes!("assets/multi_count/vert.spv"),
                ShaderStage::Vertex,
            ),
            entry: "main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: shader(
                include_bytes!("assets/multi_count/pixel.spv"),
                ShaderStage::Fragment,
            ),
            entry: "main".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![],
        color_targets: vec![Some(ColorTargetState {
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
        depth_stencil: None,
        multisample: MultisampleState {
            count: 1,
            mask: u64::from(u32::MAX),
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    }))
    .unwrap();
    let tex = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 4,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let views = (0..4)
        .map(|layer| {
            d.create_view(
                &tex,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2Array,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: layer..layer + 1,
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16384,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    // Exercise both Core alignment and native bounds rejection on fresh
    // encoders because a backend recording error poisons its pass.
    for (args_offset, count_offset) in [(2, 64), (64, 64), (0, 80)] {
        let mut invalid = d.encoder(lane).unwrap();
        let color = [Some(ColorAttachment {
            view: AttachmentView::Texture(&views[0]),
            resolve_target: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
        })];
        let mut pass = invalid
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &color,
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
        pass.set_pipeline(&rp).unwrap();
        assert!(matches!(
            d.raster_extras().unwrap().multi_draw_count(
                &mut pass,
                &args,
                args_offset,
                &counts,
                count_offset,
                2,
                if indexed { 20 } else { 16 },
                indexed
            ),
            Err(Error::InvalidRange)
        ));
    }
    let mut e = d.encoder(lane).unwrap();
    let mut records = vec![0u8; 80];
    let words: Vec<u32> = if indexed {
        vec![3, 1, 0, 0, 0, 6, 1, 3, 0, 0]
    } else {
        vec![3, 1, 0, 0, 6, 1, 0, 0]
    };
    for (i, v) in words.into_iter().enumerate() {
        records[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
    }
    let index = if indexed {
        Some(
            d.create_buffer(&BufferDesc {
                label: None,
                size: 36,
                usage: BufferUsage::INDEX | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap(),
        )
    } else {
        None
    };
    if let Some(index) = &index {
        let indices = [0u32, 1, 2, 0, 1, 2, 3, 4, 5]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        e.write_buffer(index, 0, &indices).unwrap();
        e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: index,
                range: 0..36,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: BufferUses {
                    stages: StageMask::VERTEX_INPUT,
                    accesses: AccessMask::INDEX_READ,
                },
            },
            ownership: None,
        })])
        .unwrap();
    }
    e.write_buffer(&args, 0, &records).unwrap();
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let bb = |from, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &counts,
                range: 0..80,
            },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
    e.transition_resources(&[bb(empty, write)]).unwrap();
    {
        let mut p = d.compute().unwrap().begin(&mut e).unwrap();
        p.set_pipeline(&cp).unwrap();
        p.set_bind_group(0, &group, &[]).unwrap();
        p.dispatch([1, 1, 1]).unwrap();
        p.finish().unwrap();
    }
    let indirect = BufferUses {
        stages: StageMask::INDIRECT,
        accesses: AccessMask::INDIRECT_READ,
    };
    e.transition_resources(&[bb(write, indirect)]).unwrap();
    if !aliased {
        e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &args,
                range: 0..80,
            },
            usage: StateTransition {
                from: empty,
                to: indirect,
            },
            ownership: None,
        })])
        .unwrap();
    }
    let te = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let tr = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let tc = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let tb = |from, to, layout| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..4,
            },
            usage: StateTransition { from, to },
            layout: Some(layout),
            ownership: None,
        })
    };
    e.transition_resources(&[tb(
        te,
        tr,
        (ImageLayout::Undefined, ImageLayout::ColorAttachment),
    )])
    .unwrap();
    for i in 0..4 {
        let color = [Some(ColorAttachment {
            view: AttachmentView::Texture(&views[i]),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
            store: StoreOp::Store,
        })];
        let mut p = e
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &color,
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
        p.viewport(Viewport {
            x: 0.,
            y: 0.,
            width: 16.,
            height: 16.,
            min_depth: 0.,
            max_depth: 1.,
        })
        .unwrap();
        p.scissor(Rect {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
        })
        .unwrap();
        p.set_pipeline(&rp).unwrap();
        if let Some(index) = &index {
            p.set_index_buffer(index, 0..36, IndexFormat::Uint32)
                .unwrap();
        }
        d.raster_extras()
            .unwrap()
            .multi_draw_count(
                &mut p,
                &args,
                0,
                &counts,
                64 + i as u64 * 4,
                2,
                if indexed { 20 } else { 16 },
                indexed,
            )
            .unwrap();
        p.finish().unwrap();
    }
    e.transition_resources(&[tb(
        tr,
        tc,
        (ImageLayout::ColorAttachment, ImageLayout::CopySrc),
    )])
    .unwrap();
    for layer in 0..4 {
        e.texture_to_buffer(
            TextureRegion {
                texture: &tex,
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
    let ticket = e.readback_buffer(&dst, 0..16384).unwrap();
    let work = e.finish().unwrap();
    // Native encoder keeps indirect, index and descriptor resources alive.
    drop(args);
    drop(counts);
    drop(index);
    drop(group);
    drop(cp);
    drop(rp);

    let id = PlanId {
        device: d.id(),
        ordinal: 755335,
    };
    // SAFETY: the recorded command list and plan use this device and lane.
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
    // SAFETY: the work is submitted once on its recording lane.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    // SAFETY: completion covers the readback ticket.
    let data = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    for i in 0..4 {
        let left = i * 4096 + 8 * 256 + 4 * 4;
        let right = i * 4096 + 8 * 256 + 12 * 4;
        assert_eq!(
            &data[left..left + 4],
            if i == 0 {
                &[0, 0, 0, 255]
            } else {
                &[255, 0, 0, 255]
            }
        );
        assert_eq!(
            &data[right..right + 4],
            if i < 2 {
                &[0, 0, 0, 255]
            } else {
                &[0, 255, 0, 255]
            }
        );
    }
}
