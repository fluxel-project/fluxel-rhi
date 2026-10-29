//! WGSL raster shader specialization and binding ABI GPU checks.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block as block_on;

#[test]
fn wgsl_raster_override_is_applied() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let source = b"@id(7) override red: u32 = 0u; @vertex fn vs_main(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(position,0.5,1.0); } @fragment fn ps_main() -> @location(0) vec4<f32> { return vec4<f32>(f32(red) / 255.0,0.0,0.0,1.0); }";
    let shader = block_on(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![
            ShaderEntryPoint {
                name: "vs_main".into(),
                stage: ShaderStage::Vertex,
                workgroup_size: None,
            },
            ShaderEntryPoint {
                name: "ps_main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            },
        ],
        required_features: vec![],
    }))
    .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = block_on(d.raster_pipeline(RasterPipelineDesc {
        label: None,
        layout,
        vertex: PipelineStage {
            module: shader.clone(),
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: shader,
            entry: "ps_main".into(),
            constants: vec![("7".into(), SpecializationValue::U32(64))],
        }),
        vertex_buffers: vec![VertexBufferLayout {
            stride: 8,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![VertexAttribute {
                location: 0,
                offset: 0,
                format: VertexFormat::Float32(2),
            }],
        }],
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
    let color = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap();
    let views = |texture: &Texture<_>, format, aspect| {
        (0..2)
            .map(|layer| {
                d.create_view(
                    texture,
                    &TextureViewDesc {
                        format,
                        dimension: TextureViewDimension::D2Array,
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
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 8192,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let vertex = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 24,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    let vertices = [-1.0f32, -1.0, 3.0, -1.0, -1.0, 3.0]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect::<Vec<_>>();
    e.write_buffer(&vertex, 0, &vertices).unwrap();
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
                layers: 0..2,
            },
            usage: StateTransition { from, to },
            layout: Some(layouts),
            ownership: None,
        })
    };
    e.transition_resources(&[barrier(
        empty,
        render,
        (ImageLayout::Common, ImageLayout::ColorAttachment),
    )])
    .unwrap();
    let attachments = [Some(ColorAttachment {
        view: AttachmentView::Texture(&colors[0]),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
        store: StoreOp::Store,
    })];
    let mut pass = e
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
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_vertex_buffer(0, &vertex, 0..24).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    e.transition_resources(&[barrier(
        render,
        copy,
        (ImageLayout::ColorAttachment, ImageLayout::CopySrc),
    )])
    .unwrap();
    for layer in 0..2 {
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
    assert_eq!(&bytes[center..center + 4], &[64, 0, 0, 255]);
}

#[test]
fn wgsl_raster_rejects_structured_storage_view() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let source = b"@id(7) override red: u32 = 0u; @group(0) @binding(0) var<storage, read> data: array<u32>; @vertex fn vs_main(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(position,0.5,1.0); } @fragment fn ps_main() -> @location(0) vec4<f32> { return vec4<f32>(f32(red + data[0]) / 255.0,0.0,0.0,1.0); }";
    let shader = block_on(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![
            ShaderEntryPoint {
                name: "vs_main".into(),
                stage: ShaderStage::Vertex,
                workgroup_size: None,
            },
            ShaderEntryPoint {
                name: "ps_main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            },
        ],
        required_features: vec![],
    }))
    .unwrap();
    let group_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 16,
                    access: StorageAccess::Read,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let data = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::STORAGE,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::StorageBufferView {
                    buffer: &data,
                    range: 0..16,
                    view: StorageBufferView::Structured { stride: 4 },
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
    let pipeline = block_on(d.raster_pipeline(RasterPipelineDesc {
        label: None,
        layout,
        vertex: PipelineStage {
            module: shader.clone(),
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: shader,
            entry: "ps_main".into(),
            constants: vec![("7".into(), SpecializationValue::U32(64))],
        }),
        vertex_buffers: vec![VertexBufferLayout {
            stride: 8,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![VertexAttribute {
                location: 0,
                offset: 0,
                format: VertexFormat::Float32(2),
            }],
        }],
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
    let color = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap();
    let views = |texture: &Texture<_>, format, aspect| {
        (0..2)
            .map(|layer| {
                d.create_view(
                    texture,
                    &TextureViewDesc {
                        format,
                        dimension: TextureViewDimension::D2Array,
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
    let vertex = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 24,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    let vertices = [-1.0f32, -1.0, 3.0, -1.0, -1.0, 3.0]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect::<Vec<_>>();
    e.write_buffer(&vertex, 0, &vertices).unwrap();
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let render = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let barrier = |from, to, layouts| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &color,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..2,
            },
            usage: StateTransition { from, to },
            layout: Some(layouts),
            ownership: None,
        })
    };
    e.transition_resources(&[barrier(
        empty,
        render,
        (ImageLayout::Common, ImageLayout::ColorAttachment),
    )])
    .unwrap();
    let attachments = [Some(ColorAttachment {
        view: AttachmentView::Texture(&colors[0]),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
        store: StoreOp::Store,
    })];
    let mut pass = e
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
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_vertex_buffer(0, &vertex, 0..24).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    assert!(
        pass.draw(0..3, 0..1).is_err(),
        "WGSL raw storage ABI must reject structured buffer views"
    );
}
