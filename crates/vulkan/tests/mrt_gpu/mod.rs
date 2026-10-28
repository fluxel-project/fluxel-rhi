//! Real Vulkan GPU readback for multiple raster targets, holes and blending.
#![cfg(all(windows, feature = "hardware-tests"))]
#[path = "resolve.rs"]
mod resolve;
#[path = "transient.rs"]
mod transient;
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
fn two_color_targets_receive_distinct_outputs() {
    draw_targets(false, false, false, false, false);
}

#[test]
fn empty_slot_preserves_target_indices() {
    draw_targets(true, false, false, false, false);
}

#[test]
fn independent_write_masks_preserve_unwritten_channels() {
    draw_targets(false, true, false, false, false);
}

#[test]
fn independent_blend_equations_and_write_masks() {
    draw_targets(false, false, true, false, false);
}

#[test]
fn dynamic_blend_constant_controls_gpu_output() {
    draw_targets(false, false, false, true, false);
}

fn draw_targets(gap: bool, masked: bool, blended: bool, constant: bool, dual: bool) {
    let d = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
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
        include_bytes!("../../../rhi/examples/assets/01_triangle/triangle_vs.spv"),
        "vs_main",
        ShaderStage::Vertex,
    );
    // Compile the self-contained pixel shader with the Windows SDK's dxc tool.
    let dir = std::env::temp_dir().join(format!(
        "fluxel-mrt-{}-{gap}-{masked}-{blended}-{constant}-{dual}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("mrt.hlsl");
    let output = dir.join("mrt.spv");
    let hlsl = format!(
        "struct Output {{ float4 a : SV_Target0; float4 b : SV_Target{}; }}; Output ps_main() {{ Output o; o.a = float4(1,0,0,1); o.b = float4(0,1,0,1); return o; }}",
        if gap { 2 } else { 1 }
    );
    let hlsl = if dual {
        hlsl.replace("float4(0,1,0,1)", "float4(0.25,0.5,0.75,1)")
    } else {
        hlsl
    };
    std::fs::write(&source, hlsl).unwrap();
    let result = std::process::Command::new("dxc")
        .args([
            "-spirv",
            "-fspv-target-env=vulkan1.2",
            "-T",
            "ps_6_0",
            "-E",
            "ps_main",
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
    let pixel_bytes = std::fs::read(&output).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let ps = block_on(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from(pixel_bytes),
        entries: vec![ShaderEntryPoint {
            name: "ps_main".into(),
            stage: ShaderStage::Fragment,
            workgroup_size: None,
        }],
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
            module: vs,
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: ps,
            entry: "ps_main".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![VertexBufferLayout {
            stride: 24,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float32(3),
                },
                VertexAttribute {
                    location: 1,
                    offset: 12,
                    format: VertexFormat::Float32(3),
                },
            ],
        }],
        color_targets: {
            let target = Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: 15,
            });
            if gap {
                vec![target.clone(), None, target]
            } else if dual {
                let blend = BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::Src1,
                        dst_factor: BlendFactor::OneMinusSrc1,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::Zero,
                        operation: BlendOperation::Add,
                    },
                };
                vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: Some(blend),
                    write_mask: 15,
                })]
            } else if constant {
                let blend = BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::Constant,
                        dst_factor: BlendFactor::OneMinusConstant,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::Zero,
                        operation: BlendOperation::Add,
                    },
                };
                vec![
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: Some(blend),
                        write_mask: 15,
                    }),
                    target,
                ]
            } else if blended {
                let add = |src_factor, dst_factor| BlendState {
                    color: BlendComponent {
                        src_factor,
                        dst_factor,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::Zero,
                        operation: BlendOperation::Add,
                    },
                };
                vec![
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: Some(add(BlendFactor::One, BlendFactor::One)),
                        write_mask: 5,
                    }),
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: Some(add(BlendFactor::Zero, BlendFactor::One)),
                        write_mask: 2,
                    }),
                ]
            } else if masked {
                vec![
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: 1,
                    }),
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: 2,
                    }),
                ]
            } else {
                vec![target.clone(), target]
            }
        },
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
            size: 72,
            usage: BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 8192,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    let vertices: [f32; 18] = [
        -0.8, -0.8, 0.5, 1.0, 0.0, 0.0, 0.8, -0.8, 0.5, 1.0, 0.0, 0.0, 0.0, 0.8, 0.5, 1.0, 0.0, 0.0,
    ];
    e.write_buffer(
        &vertex,
        0,
        &vertices
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>(),
    )
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
        (ImageLayout::Undefined, ImageLayout::ColorAttachment),
    )])
    .unwrap();
    let mut attachments = vec![
        Some(ColorAttachment {
            view: AttachmentView::Texture(&colors[0]),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 1.0, 1.0])),
            store: StoreOp::Store,
        }),
        Some(ColorAttachment {
            view: AttachmentView::Texture(&colors[1]),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 1.0, 1.0])),
            store: StoreOp::Store,
        }),
    ];
    if gap {
        attachments.insert(1, None);
    }
    if dual {
        attachments.truncate(1);
    }
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
    if constant {
        pass.blend_constant([0.25, 0.5, 0.75, 1.0]).unwrap();
    }
    pass.set_vertex_buffer(0, &vertex, 0..72).unwrap();
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
    let expected0 = if constant || dual {
        [64, 0, 64, 255]
    } else if masked || blended {
        [255, 0, 255, 255]
    } else {
        [255, 0, 0, 255]
    };
    let expected1 = if blended {
        [0, 0, 255, 255]
    } else if masked {
        [0, 255, 255, 255]
    } else {
        [0, 255, 0, 255]
    };
    assert_eq!(&bytes[center..center + 4], &expected0);
    if !dual {
        assert_eq!(&bytes[4096 + center..4096 + center + 4], &expected1);
    }
}
