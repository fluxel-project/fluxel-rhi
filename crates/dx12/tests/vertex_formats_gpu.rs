//! Real GPU checks for primitive topology and vertex input.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block as block_on;

#[test]
fn compressed_vertex_formats_are_decoded() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let dir = std::env::temp_dir().join(format!("fluxel-vertex-formats-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("vertex.hlsl");
    let out = dir.join("vertex.dxil");
    std::fs::write(&src, "struct O { float4 p : SV_Position; float3 c : LOCATION0; }; O vs_main(float2 p : LOCATION0, float4 c : LOCATION1, uint2 flags : LOCATION2) { O o; o.p=float4(p,0.5,1); o.c=(flags.x==3 && flags.y==5) ? c.rgb : float3(0,0,0); return o; } float4 ps_main(O i) : SV_Target0 { return float4(i.c,1); }").unwrap();
    let status = std::process::Command::new("dxc")
        .args(["-T", "vs_6_0", "-E", "vs_main", "-Fo"])
        .arg(&out)
        .arg(&src)
        .output()
        .expect("hardware test requires dxc in PATH");
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let bytes = std::fs::read(&out).unwrap();
    let result = std::process::Command::new("dxc")
        .args(["-T", "ps_6_0", "-E", "ps_main", "-Fo"])
        .arg(&out)
        .arg(&src)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let pixel = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let vs = block_on(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Dxil,
        bytes: Arc::from(bytes),
        entries: vec![ShaderEntryPoint {
            name: "vs_main".into(),
            stage: ShaderStage::Vertex,
            workgroup_size: None,
        }],
        required_features: vec![],
    }))
    .unwrap();
    let ps = block_on(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Dxil,
        bytes: Arc::from(pixel),
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
            stride: 12,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float16(2),
                },
                VertexAttribute {
                    location: 1,
                    offset: 4,
                    format: VertexFormat::Unorm8(4),
                },
                VertexAttribute {
                    location: 2,
                    offset: 8,
                    format: VertexFormat::Uint8(2),
                },
            ],
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
            size: 36,
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
    let mut data = Vec::new();
    for (x, y) in [(0xbc00u16, 0xbc00u16), (0x3c00, 0xbc00), (0, 0x3c00)] {
        data.extend_from_slice(&x.to_ne_bytes());
        data.extend_from_slice(&y.to_ne_bytes());
        data.extend_from_slice(&[255, 0, 0, 255, 3, 5, 0, 0]);
    }
    e.write_buffer(&vertex, 0, &data).unwrap();
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
    pass.set_vertex_buffer(0, &vertex, 0..36).unwrap();
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
    assert_eq!(&bytes[center..center + 4], &[255, 0, 0, 255]);
}
