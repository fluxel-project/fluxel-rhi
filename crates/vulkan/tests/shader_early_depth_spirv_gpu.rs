//! Real-device validation for Direct-SPIR-V `EarlyFragmentTests` admission.
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
fn direct_spirv_early_fragment_tests_suppress_depth_failed_side_effects() {
    let native = VulkanInstance::new().unwrap();
    let instance = native.instance().unwrap();
    let adapter = instance
        .enumerate()
        .unwrap()
        .into_iter()
        .next()
        .expect("a graphics-capable Vulkan adapter");
    eprintln!("Vulkan Direct-SPIR-V early-depth adapter: {}", adapter.name);
    let device = block_on(instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: vec![],
    }))
    .unwrap();
    assert!(device.capabilities().shader_early_depth_test);
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;

    let vertex = block_on(
        device.shader(ShaderDesc {
            label: Some("Direct-SPIR-V early-depth vertex".into()),
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(
                br#"
@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(positions[index], 0.75, 1.0);
}
"#
                .as_slice(),
            ),
            entries: vec![ShaderEntryPoint {
                name: "vs".into(),
                stage: ShaderStage::Vertex,
                workgroup_size: None,
            }],
            required_features: vec![],
        }),
    )
    .unwrap();
    let fragment = block_on(device.shader(ShaderDesc {
        label: Some("Direct-SPIR-V EarlyFragmentTests fragment".into()),
        format: ShaderFormat::SpirV,
        bytes: Arc::from(
            include_bytes!("assets/early_depth_spirv/early_depth.frag.spv").as_slice(),
        ),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Fragment,
            workgroup_size: None,
        }],
        required_features: vec![SHADER_FEATURE_EARLY_DEPTH_TEST.into()],
    }))
    .unwrap();

    let side_effect_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("Direct-SPIR-V early-depth side-effect layout".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 8 * 8 * 4,
                    access: StorageAccess::ReadWrite,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let side_effects = device
        .create_buffer(&BufferDesc {
            label: Some("Direct-SPIR-V early-depth markers".into()),
            size: 8 * 8 * 4,
            usage: BufferUsage::STORAGE
                .union(BufferUsage::COPY_DST)
                .union(BufferUsage::COPY_SRC),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let side_effect_readback = device
        .create_buffer(&BufferDesc {
            label: Some("Direct-SPIR-V early-depth marker readback".into()),
            size: 8 * 8 * 4,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let group = device
        .create_bind_group(&BindGroupDesc {
            label: Some("Direct-SPIR-V early-depth markers".into()),
            layout: &side_effect_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::StorageBufferView {
                    buffer: &side_effects,
                    range: 0..8 * 8 * 4,
                    view: StorageBufferView::Raw,
                },
            }],
        })
        .unwrap();
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&side_effect_layout],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = |label: &str, depth_compare| {
        block_on(device.raster_pipeline(RasterPipelineDesc {
            label: Some(label.into()),
            layout: layout.clone(),
            vertex: PipelineStage {
                module: vertex.clone(),
                entry: "vs".into(),
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
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare,
                stencil_front: StencilFaceState {
                    compare: CompareFunction::Always,
                    fail: StencilOperation::Keep,
                    depth_fail: StencilOperation::Keep,
                    pass: StencilOperation::Keep,
                },
                stencil_back: StencilFaceState {
                    compare: CompareFunction::Always,
                    fail: StencilOperation::Keep,
                    depth_fail: StencilOperation::Keep,
                    pass: StencilOperation::Keep,
                },
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
    let depth_fail = pipeline("Direct-SPIR-V early-depth failure", CompareFunction::Less);
    let depth_pass = pipeline(
        "Direct-SPIR-V early-depth success",
        CompareFunction::Greater,
    );

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
    let color_readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 2048,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .write_buffer(&side_effects, 0, &[0u8; 8 * 8 * 4])
        .unwrap();
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
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &depth,
                    aspects: TextureAspects::DEPTH,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: TextureUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    },
                    to: TextureUses {
                        stages: StageMask::FRAGMENT,
                        accesses: AccessMask::DEPTH_READ | AccessMask::DEPTH_WRITE,
                    },
                },
                layout: Some((ImageLayout::Undefined, ImageLayout::DepthStencilAttachment)),
                ownership: None,
            }),
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &side_effects,
                    range: 0..8 * 8 * 4,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_WRITE,
                    },
                    to: BufferUses {
                        stages: StageMask::FRAGMENT,
                        accesses: AccessMask::SHADER_READ | AccessMask::SHADER_WRITE,
                    },
                },
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
            label: Some("Direct-SPIR-V early-depth pass".into()),
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
    pass.set_pipeline(&depth_fail).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.scissor(Rect {
        x: 4,
        y: 0,
        width: 4,
        height: 8,
    })
    .unwrap();
    pass.set_pipeline(&depth_pass).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();

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
            }),
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &side_effects,
                    range: 0..8 * 8 * 4,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::FRAGMENT,
                        accesses: AccessMask::SHADER_READ | AccessMask::SHADER_WRITE,
                    },
                    to: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                },
                ownership: None,
            }),
        ])
        .unwrap();
    encoder
        .copy_buffer(&side_effects, 0, &side_effect_readback, 0, 8 * 8 * 4)
        .unwrap();
    let side_effect_ticket = encoder
        .readback_buffer(&side_effect_readback, 0..8 * 8 * 4)
        .unwrap();
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &color,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            &color_readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 8,
            },
        )
        .unwrap();
    let color_ticket = encoder.readback_buffer(&color_readback, 0..2048).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 91_007,
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
        _ => panic!("Direct-SPIR-V early-depth GPU work was rejected"),
    };
    block_on(device.wait_completion(&point)).unwrap();
    let color_bytes = unsafe {
        block_on(device.read_after(color_ticket, &point))
            .unwrap()
            .bytes
    };
    let marker_bytes = unsafe {
        block_on(device.read_after(side_effect_ticket, &point))
            .unwrap()
            .bytes
    };
    let y = 4 * 256;
    assert_eq!(&color_bytes[y + 1 * 4..y + 1 * 4 + 4], &[0, 0, 255, 255]);
    assert_eq!(&color_bytes[y + 6 * 4..y + 6 * 4 + 4], &[0, 255, 0, 255]);
    for y in 0..8 {
        for x in 0..8 {
            let offset = (y * 8 + x) * 4;
            let actual = u32::from_le_bytes(marker_bytes[offset..offset + 4].try_into().unwrap());
            let expected = if x < 4 { 0 } else { 1 };
            assert_eq!(
                actual, expected,
                "Direct-SPIR-V early-depth marker at ({x}, {y})"
            );
        }
    }
    eprintln!(
        "Direct-SPIR-V EarlyFragmentTests depth failure suppressed markers; passing half wrote markers"
    );
}
