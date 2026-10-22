//! Primary bind group after secondary bundle execution.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};
fn block<F: Future>(f: F) -> F::Output {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}
#[test]
fn primary_bind_group_survives_secondary() {
    run(true);
}
fn run(is_secondary: bool) {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|x| x.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let shader = |bytes: &'static [u8], stage| {
        block(d.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Dxil,
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
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 32,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let uniform = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 512,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let group_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Vertex],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: true,
                        min_size: 4,
                        access: StorageAccess::Read,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: vec![ShaderStage::Vertex],
                    ty: BindingType::UniformBuffer {
                        dynamic_offset: true,
                        min_size: 16,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
            ],
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::StorageBufferView {
                        buffer: &buffer,
                        range: 0..4,
                        view: StorageBufferView::Raw,
                    },
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer {
                        buffer: &uniform,
                        range: 0..256,
                    },
                },
            ],
        })
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();
    let desc = || RasterPipelineDesc {
        label: None,
        layout: layout.clone(),
        vertex: PipelineStage {
            module: shader(
                include_bytes!("assets/dynamic_storage_vertex.dxil"),
                ShaderStage::Vertex,
            ),
            entry: "main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: shader(
                include_bytes!("assets/multi_count/pixel.dxil"),
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
    };
    let rp = block(d.raster_pipeline(desc())).unwrap();
    let rp2 = block(d.raster_pipeline(desc())).unwrap();
    let tex = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let views = (0..1)
        .map(|layer| {
            d.create_view(
                &tex,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2,
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
            size: 4096,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let sig = SecondaryRasterDesc {
        color_formats: vec![Some(TextureFormat::Rgba8Unorm)],
        depth_stencil_format: None,
        samples: 1,
        multiview_mask: None,
    };
    let domain = d.secondary().unwrap();
    let mut secondary = domain.begin(&sig).unwrap();
    secondary.set_pipeline(&rp).unwrap();
    secondary
        .viewport(Viewport {
            x: 0.,
            y: 0.,
            width: 16.,
            height: 16.,
            min_depth: 0.,
            max_depth: 1.,
        })
        .unwrap();
    secondary
        .scissor(Rect {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
        })
        .unwrap();
    secondary
        .scissor(Rect {
            x: 0,
            y: 0,
            width: 8,
            height: 16,
        })
        .unwrap();
    secondary.set_bind_group(0, &group, &[0, 256]).unwrap();
    secondary.draw(0..3, 0..1).unwrap();
    let first_bundle = domain.finish(secondary).unwrap();
    let mut secondary = domain.begin(&sig).unwrap();
    secondary.set_pipeline(&rp2).unwrap();
    secondary
        .viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: 16.0,
            height: 16.0,
            min_depth: 0.0,
            max_depth: 1.0,
        })
        .unwrap();
    secondary
        .scissor(Rect {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
        })
        .unwrap();
    secondary.set_bind_group(0, &group, &[16, 256]).unwrap();
    secondary.draw(0..3, 0..1).unwrap();
    let bundle = domain.finish(secondary).unwrap();
    let mut e = d.encoder(lane).unwrap();
    let mut bytes = [0u8; 32];
    bytes[..4].copy_from_slice(&42u32.to_le_bytes());
    e.write_buffer(&buffer, 0, &bytes).unwrap();
    let mut uniform_bytes = [0u8; 512];
    uniform_bytes[256..260].copy_from_slice(&42u32.to_le_bytes());
    e.write_buffer(&uniform, 0, &uniform_bytes).unwrap();
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &uniform,
            range: 0..512,
        },
        usage: StateTransition {
            from: BufferUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            to: BufferUses {
                stages: StageMask::VERTEX,
                accesses: AccessMask::UNIFORM_READ,
            },
        },
        ownership: None,
    })])
    .unwrap();
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &buffer,
            range: 0..32,
        },
        usage: StateTransition {
            from: BufferUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            to: BufferUses {
                stages: StageMask::VERTEX,
                accesses: AccessMask::SHADER_READ,
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
    let barrier = |from, to, layout| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition { from, to },
            layout: Some(layout),
            ownership: None,
        })
    };
    e.transition_resources(&[barrier(
        empty,
        render,
        (ImageLayout::Common, ImageLayout::ColorAttachment),
    )])
    .unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&views[0]),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
        store: StoreOp::Store,
    })];
    let mut pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
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
            secondary_inheritance: if is_secondary { Some(sig) } else { None },
        })
        .unwrap();
    pass.set_pipeline(&rp).unwrap();
    pass.set_bind_group(0, &group, &[0, 256]).unwrap();
    pass.viewport(Viewport {
        x: 0.,
        y: 0.,
        width: 16.,
        height: 16.,
        min_depth: 0.,
        max_depth: 1.,
    })
    .unwrap();
    pass.scissor(Rect {
        x: 0,
        y: 0,
        width: 16,
        height: 16,
    })
    .unwrap();
    domain.execute(&mut pass, &bundle).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    drop(first_bundle);
    drop(bundle);
    drop(group);
    drop(buffer);
    drop(uniform);
    drop(rp);
    drop(rp2);
    e.transition_resources(&[barrier(
        render,
        copy,
        (ImageLayout::ColorAttachment, ImageLayout::CopySrc),
    )])
    .unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &tex,
            mip: 0,
            origin: [0, 0, 0],
            extent: [16, 16, 1],
            aspect: TextureAspects::COLOR,
        },
        &dst,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 16,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&dst, 0..4096).unwrap();
    let work = e.finish().unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755350,
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
    let left = 8 * 256 + 4 * 4;
    let right = 8 * 256 + 12 * 4;
    eprintln!(
        "secondary={is_secondary} left={:?} right={:?}",
        &data[left..left + 4],
        &data[right..right + 4]
    );
    assert_eq!(&data[left..left + 4], &[255, 0, 0, 255]);
    assert_eq!(&data[right..right + 4], &[255, 0, 0, 255]);
}
