//! Direct typed-RHI validation for Vulkan vertex draw-index admission.
#![cfg(all(windows, feature = "hardware-tests"))]

#[path = "support/mod.rs"]
mod support;
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::sync::Arc;
use support::block_on;

#[test]
fn draw_index_distinguishes_commands_in_native_multi_draw() {
    let device = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    assert!(device.capabilities().shader_draw_index);
    assert!(device.capabilities().multi_draw);
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let wgsl_draw_index = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(
            b"@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }".as_slice(),
        ),
        entries: vec![ShaderEntryPoint {
            name: "vs".into(),
            stage: ShaderStage::Vertex,
            workgroup_size: None,
        }],
        required_features: vec![SHADER_FEATURE_DRAW_INDEX.into()],
    };
    assert!(matches!(
        device.shader_route(&wgsl_draw_index),
        ShaderRoute::Unsupported
    ));

    let vertex = block_on(device.shader(ShaderDesc {
        label: Some("draw-index vertex".into()),
        format: ShaderFormat::SpirV,
        bytes: Arc::from(include_bytes!("assets/draw_index.vert.spv").as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Vertex,
            workgroup_size: None,
        }],
        required_features: vec![SHADER_FEATURE_DRAW_INDEX.into()],
    }))
    .unwrap();
    let fragment = block_on(device.shader(ShaderDesc {
        label: Some("draw-index fragment".into()),
        format: ShaderFormat::SpirV,
        bytes: Arc::from(include_bytes!("assets/draw_index.frag.spv").as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Fragment,
            workgroup_size: None,
        }],
        required_features: vec![],
    }))
    .unwrap();
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = block_on(device.raster_pipeline(RasterPipelineDesc {
        label: Some("draw-index pipeline".into()),
        layout,
        vertex: PipelineStage {
            module: vertex,
            entry: "main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: fragment,
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
            mask: u64::MAX,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    }))
    .unwrap();
    let color = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 4, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap();
    let view = device
        .create_view(
            &color,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let args = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 32,
            usage: BufferUsage::INDIRECT.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 1024,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    let words = [3u32, 1, 0, 0, 3, 1, 3, 0];
    let argument_bytes = words
        .into_iter()
        .flat_map(u32::to_ne_bytes)
        .collect::<Vec<_>>();
    encoder.write_buffer(&args, 0, &argument_bytes).unwrap();
    encoder
        .transition_resources(&[
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &args,
                    range: 0..32,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    },
                    to: BufferUses {
                        stages: StageMask::INDIRECT,
                        accesses: AccessMask::INDIRECT_READ,
                    },
                },
                ownership: None,
            }),
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &color,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: TextureUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    },
                    to: TextureUses {
                        stages: StageMask::COLOR_OUTPUT,
                        accesses: AccessMask::COLOR_WRITE,
                    },
                },
                layout: Some((ImageLayout::Undefined, ImageLayout::ColorAttachment)),
                ownership: None,
            }),
        ])
        .unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
        store: StoreOp::Store,
    })];
    let mut pass = encoder
        .begin_raster(&RasterPassDesc {
            label: Some("draw-index pass".into()),
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 4,
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
        width: 8.0,
        height: 4.0,
        min_depth: 0.0,
        max_depth: 1.0,
    })
    .unwrap();
    pass.scissor(Rect {
        x: 0,
        y: 0,
        width: 8,
        height: 4,
    })
    .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    device
        .raster_extras()
        .unwrap()
        .multi_draw(&mut pass, &args, 0, 2, 16, false)
        .unwrap();
    pass.finish().unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &color,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COLOR_OUTPUT,
                    accesses: AccessMask::COLOR_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::ColorAttachment, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &color,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..1024).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 91_006,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    let outcome = unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![encoder.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("draw-index GPU work was rejected"),
    };
    block_on(device.wait_completion(&point)).unwrap();
    let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };
    let y = 2 * 256;
    assert_eq!(&bytes[y + 1 * 4..y + 1 * 4 + 4], &[255, 0, 0, 255]);
    assert_eq!(&bytes[y + 6 * 4..y + 6 * 4 + 4], &[0, 255, 0, 255]);
}
