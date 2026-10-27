//! Direct typed-RHI validation for WGSL early depth-test admission and execution.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::{future::Future, sync::Arc};

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = std::task::Waker::noop();
    let mut context = std::task::Context::from_waker(waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

#[test]
fn force_and_conservative_early_depth_modes_match_readback() {
    let device = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    assert!(device.capabilities().shader_early_depth_test);
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;

    let vertex_source = br#"
@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(positions[index], 0.75, 1.0);
}
"#;
    let fragment_source = br#"
struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
@early_depth_test(force)
fn fs_force() -> FragmentOutput {
    return FragmentOutput(vec4<f32>(1.0, 0.0, 0.0, 1.0), 0.25);
}

@fragment
@early_depth_test(less_equal)
fn fs_conservative() -> FragmentOutput {
    return FragmentOutput(vec4<f32>(0.0, 1.0, 0.0, 1.0), 0.25);
}
"#;
    let vertex = block_on(device.shader(ShaderDesc {
        label: Some("early-depth-test vertex".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(vertex_source.as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "vs".into(),
            stage: ShaderStage::Vertex,
            workgroup_size: None,
        }],
        required_features: vec![],
    }))
    .unwrap();
    let fragment_desc = |required_features| ShaderDesc {
        label: Some("early-depth-test fragment".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(fragment_source.as_slice()),
        entries: vec![
            ShaderEntryPoint {
                name: "fs_force".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            },
            ShaderEntryPoint {
                name: "fs_conservative".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            },
        ],
        required_features,
    };
    assert!(matches!(
        block_on(device.shader(fragment_desc(vec![]))),
        Err(Error::Shader(_))
    ));
    let fragment =
        block_on(device.shader(fragment_desc(vec![SHADER_FEATURE_EARLY_DEPTH_TEST.into()])))
            .unwrap();

    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let face = StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    let pipeline = |entry: &str| {
        block_on(device.raster_pipeline(RasterPipelineDesc {
            label: Some(format!("early-depth-test {entry} pipeline")),
            layout: layout.clone(),
            vertex: PipelineStage {
                module: vertex.clone(),
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment.clone(),
                entry: entry.into(),
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
                mask: u64::MAX,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        }))
        .unwrap()
    };
    let force_pipeline = pipeline("fs_force");
    let conservative_pipeline = pipeline("fs_conservative");

    let color = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap();
    let depth = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format: TextureFormat::Depth32Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let color_view = device
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
    let depth_view = device
        .create_view(
            &depth,
            &TextureViewDesc {
                format: TextureFormat::Depth32Float,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 2048,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .transition_resources(&[
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &color,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: empty,
                    to: TextureUses {
                        stages: StageMask::COLOR_OUTPUT,
                        accesses: AccessMask::COLOR_WRITE,
                    },
                },
                layout: Some((ImageLayout::Undefined, ImageLayout::ColorAttachment)),
                ownership: None,
            }),
            ResourceTransition::Texture(TextureBarrier {
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
            }),
        ])
        .unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&color_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 1.0, 1.0])),
        store: StoreOp::Store,
    })];
    let mut pass = encoder
        .begin_raster(&RasterPassDesc {
            label: Some("early-depth-test pass".into()),
            colors: &colors,
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&depth_view),
                depth: Some((LoadOp::Clear(0.5), StoreOp::Store)),
                stencil: None,
                depth_read_only: false,
                stencil_read_only: false,
            }),
            render_area: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
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
        height: 8.0,
        min_depth: 0.0,
        max_depth: 1.0,
    })
    .unwrap();
    pass.scissor(Rect {
        x: 0,
        y: 0,
        width: 4,
        height: 8,
    })
    .unwrap();
    pass.set_pipeline(&force_pipeline).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.scissor(Rect {
        x: 4,
        y: 0,
        width: 4,
        height: 8,
    })
    .unwrap();
    pass.set_pipeline(&conservative_pipeline).unwrap();
    pass.draw(0..3, 0..1).unwrap();
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
    let region = TextureRegion {
        texture: &color,
        mip: 0,
        origin: [0, 0, 0],
        extent: [8, 8, 1],
        aspect: TextureAspects::COLOR,
    };
    encoder
        .texture_to_buffer(
            region,
            &readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 8,
            },
        )
        .unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..2048).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 91_004,
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
        _ => panic!("early-depth GPU work was rejected"),
    };
    block_on(device.wait_completion(&point)).unwrap();
    let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };
    let center = 4 * 256 + 2 * 4;
    assert_eq!(
        &bytes[center..center + 4],
        &[0, 0, 255, 255],
        "the forced early test rejects incoming z=0.75 before the shader writes z=0.25"
    );
    let conservative_center = 4 * 256 + 6 * 4;
    assert_eq!(
        &bytes[conservative_center..conservative_center + 4],
        &[0, 255, 0, 255],
        "less_equal conservative depth preserves the valid transition from z=0.75 to z=0.25"
    );
}
