//! Direct typed-RHI validation for Vulkan barycentric shader routes.
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
fn barycentric_extensions_are_admitted_by_route_and_match_interpolation_reference() {
    let device = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    let caps = device.capabilities();
    assert_eq!(
        caps.shader_barycentrics,
        caps.shader_barycentrics_khr || caps.shader_barycentrics_nv
    );
    assert!(
        caps.shader_barycentrics || caps.shader_barycentrics_amd,
        "the test device must expose at least one declared barycentric route"
    );

    let wgsl = ShaderDesc {
        label: Some("unsupported WGSL barycentric route".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(
            b"@fragment fn fs() -> @location(0) vec4f { return vec4f(0.0); }".as_slice(),
        ),
        entries: vec![ShaderEntryPoint {
            name: "fs".into(),
            stage: ShaderStage::Fragment,
            workgroup_size: None,
        }],
        required_features: vec![SHADER_FEATURE_BARYCENTRICS.into()],
    };
    assert!(matches!(
        device.shader_route(&wgsl),
        ShaderRoute::Unsupported
    ));
    let wrong_stage = ShaderDesc {
        label: Some("invalid barycentric stage".into()),
        format: ShaderFormat::SpirV,
        bytes: Arc::from(include_bytes!("assets/barycentric_khr.frag.spv").as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Vertex,
            workgroup_size: None,
        }],
        required_features: vec![SHADER_FEATURE_BARYCENTRICS.into()],
    };
    assert!(matches!(
        device.shader_route(&wrong_stage),
        ShaderRoute::Unsupported
    ));

    let mut routes: Vec<(&'static str, &'static [u8], &'static str)> = Vec::new();
    if caps.shader_barycentrics_khr {
        routes.push((
            "KHR standard barycentrics",
            include_bytes!("assets/barycentric_khr.frag.spv"),
            SHADER_FEATURE_BARYCENTRICS,
        ));
    }
    if caps.shader_barycentrics_nv {
        routes.push((
            "NV standard barycentrics",
            include_bytes!("assets/barycentric_nv.frag.spv"),
            SHADER_FEATURE_BARYCENTRICS,
        ));
    }
    if caps.shader_barycentrics_amd {
        routes.push((
            "AMD explicit-parameter barycentrics",
            include_bytes!("assets/barycentric_amd.frag.spv"),
            SHADER_FEATURE_BARYCENTRICS_AMD,
        ));
    }
    assert!(!routes.is_empty());
    eprintln!(
        "Vulkan barycentric routes exercised: {}",
        routes
            .iter()
            .map(|(label, _, _)| *label)
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut fragments = Vec::with_capacity(routes.len());
    for (label, bytes, requirement) in &routes {
        let desc = ShaderDesc {
            label: Some((*label).into()),
            format: ShaderFormat::SpirV,
            bytes: Arc::from(*bytes),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: vec![(*requirement).into()],
        };
        assert!(matches!(device.shader_route(&desc), ShaderRoute::Direct(_)));
        fragments.push(block_on(device.shader(desc)).unwrap());
    }

    if !caps.shader_barycentrics_khr {
        let unavailable_khr = ShaderDesc {
            label: Some("unavailable KHR barycentrics".into()),
            format: ShaderFormat::SpirV,
            bytes: Arc::from(include_bytes!("assets/barycentric_khr.frag.spv").as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: vec![SHADER_FEATURE_BARYCENTRICS.into()],
        };
        if caps.shader_barycentrics_nv {
            assert!(matches!(
                block_on(device.shader(unavailable_khr)),
                Err(Error::Unsupported(_))
            ));
        }
    }
    if !caps.shader_barycentrics_nv && caps.shader_barycentrics_khr {
        let unavailable_nv = ShaderDesc {
            label: Some("unavailable NV barycentrics".into()),
            format: ShaderFormat::SpirV,
            bytes: Arc::from(include_bytes!("assets/barycentric_nv.frag.spv").as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: vec![SHADER_FEATURE_BARYCENTRICS.into()],
        };
        assert!(matches!(
            block_on(device.shader(unavailable_nv)),
            Err(Error::Unsupported(_))
        ));
    }
    if !caps.shader_barycentrics_amd {
        let unavailable_amd = ShaderDesc {
            label: Some("unavailable AMD barycentrics".into()),
            format: ShaderFormat::SpirV,
            bytes: Arc::from(include_bytes!("assets/barycentric_amd.frag.spv").as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: vec![SHADER_FEATURE_BARYCENTRICS_AMD.into()],
        };
        assert!(matches!(
            device.shader_route(&unavailable_amd),
            ShaderRoute::Unsupported
        ));
    }

    let vertex = block_on(device.shader(ShaderDesc {
        label: Some("barycentric reference vertex".into()),
        format: ShaderFormat::SpirV,
        bytes: Arc::from(include_bytes!("assets/barycentric.vert.spv").as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Vertex,
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
    let pipelines = fragments
        .iter()
        .enumerate()
        .map(|(index, fragment)| {
            block_on(device.raster_pipeline(RasterPipelineDesc {
                label: Some(routes[index].0.into()),
                layout: layout.clone(),
                vertex: PipelineStage {
                    module: vertex.clone(),
                    entry: "main".into(),
                    constants: vec![],
                },
                fragment: Some(PipelineStage {
                    module: fragment.clone(),
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
            .unwrap()
        })
        .collect::<Vec<_>>();

    let layer_count = routes.len() as u32;
    let color = device
        .create_texture(&TextureDesc {
            label: Some("barycentric reference target".into()),
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: layer_count,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap();
    let views = (0..layer_count)
        .map(|layer| {
            device
                .create_view(
                    &color,
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
    let readback_size = u64::from(layer_count) * 4096;
    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("barycentric GPU readback".into()),
            size: readback_size,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let mut encoder = device.encoder(lane).unwrap();
    let render = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &color,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..layer_count,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: render,
            },
            layout: Some((ImageLayout::Undefined, ImageLayout::ColorAttachment)),
            ownership: None,
        })])
        .unwrap();
    for (view, pipeline) in views.iter().zip(&pipelines) {
        let attachments = [Some(ColorAttachment {
            view: AttachmentView::Texture(view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        })];
        let mut pass = encoder
            .begin_raster(&RasterPassDesc {
                label: Some("barycentric interpolation reference".into()),
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
        pass.set_pipeline(pipeline).unwrap();
        pass.draw(0..3, 0..1).unwrap();
        pass.finish().unwrap();
    }
    let copy = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &color,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..layer_count,
            },
            usage: StateTransition {
                from: render,
                to: copy,
            },
            layout: Some((ImageLayout::ColorAttachment, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
    for layer in 0..layer_count {
        encoder
            .texture_to_buffer(
                TextureRegion {
                    texture: &color,
                    mip: 0,
                    origin: [0, 0, layer],
                    extent: [16, 16, 1],
                    aspect: TextureAspects::COLOR,
                },
                &readback,
                TexelCopyBufferLayout {
                    offset: u64::from(layer) * 4096,
                    bytes_per_row: 256,
                    rows_per_image: 16,
                },
            )
            .unwrap();
    }
    let ticket = encoder
        .readback_buffer(&readback, 0..readback_size)
        .unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 91_009,
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
        _ => panic!("barycentric GPU work was rejected"),
    };
    block_on(device.wait_completion(&point)).unwrap();
    let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };

    for (layer, (name, _, _)) in routes.iter().enumerate() {
        let base = layer * 4096;
        for (x, y, expected) in [
            (2usize, 14usize, [219u8, 24, 12]),
            (14, 14, [6, 236, 12]),
            (8, 3, [15, 33, 207]),
            (8, 12, [95, 112, 48]),
        ] {
            let offset = base + y * 256 + x * 4;
            let actual = &bytes[offset..offset + 4];
            for channel in 0..3 {
                assert!(
                    actual[channel].abs_diff(expected[channel]) <= 4,
                    "{name} at ({x},{y}) channel {channel}: expected {:?}, got {:?}",
                    expected,
                    actual
                );
            }
            assert_eq!(actual[3], 255, "{name} alpha at ({x},{y})");
            let sum = u16::from(actual[0]) + u16::from(actual[1]) + u16::from(actual[2]);
            assert!(
                sum.abs_diff(255) <= 2,
                "{name} at ({x},{y}) barycentric weights sum to {sum}"
            );
        }
    }
}
