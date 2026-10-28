//! Direct base vertex and first instance Core readback checks.
use super::*;
async fn zero_stride(d: &Device<Gl>, _indexed: bool) {
    let shader_source = br#"@vertex fn vs(@location(0) pos:vec2f, @location(1) val:f32, @location(2) fixed:f32, @builtin(instance_index) instance:u32, @builtin(vertex_index) vertex:u32)->@builtin(position) vec4f {
      let ok = val == 42.0 && fixed == 24.0 && instance >= 0u && vertex >= 3u; return vec4f(pos, select(2.0, 0.5, ok), 1.0);
    } @fragment fn fs()->@location(0) vec4f { return vec4f(1.,0.,0.,1.); }"#;
    let shader = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(shader_source.as_slice()),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = d.shader(shader("vs", ShaderStage::Vertex)).await.unwrap();
    let fragment = d.shader(shader("fs", ShaderStage::Fragment)).await.unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: vertex,
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment,
                entry: "fs".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![
                VertexBufferLayout {
                    stride: 8,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![VertexAttribute {
                        location: 0,
                        offset: 0,
                        format: VertexFormat::Float32(2),
                    }],
                },
                VertexBufferLayout {
                    stride: 0,
                    step_mode: VertexStepMode::Instance,
                    attributes: vec![VertexAttribute {
                        location: 1,
                        offset: 0,
                        format: VertexFormat::Float32(1),
                    }],
                },
                VertexBufferLayout {
                    stride: 0,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![VertexAttribute {
                        location: 2,
                        offset: 4,
                        format: VertexFormat::Float32(1),
                    }],
                },
            ],
            color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
            primitive: triangle_list_state(),
            depth_stencil: None,
            multisample: single_sample_state(u64::MAX),
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let view = d
        .create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let buffer = |size, usage| {
        d.create_buffer(&BufferDesc {
            label: None,
            size,
            usage,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let positions = buffer(48, BufferUsage::VERTEX | BufferUsage::COPY_DST);
    let instances = buffer(16, BufferUsage::VERTEX | BufferUsage::COPY_DST);
    let indices = buffer(8, BufferUsage::INDEX | BufferUsage::COPY_DST);
    let out = [
        buffer(256, BufferUsage::COPY_DST | BufferUsage::COPY_SRC),
        buffer(256, BufferUsage::COPY_DST | BufferUsage::COPY_SRC),
    ];
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 280).unwrap();
    // SAFETY: resources outlive the completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    let e = run.commands();
    let mut vertices = Vec::new();
    for p in [
        [0f32, 0.],
        [0., 0.],
        [0., 0.],
        [-1., -1.],
        [3., -1.],
        [-1., 3.],
    ] {
        for v in p {
            vertices.extend_from_slice(&v.to_le_bytes());
        }
    }
    e.write_buffer(&positions, 0, &vertices).unwrap();
    let mut data = Vec::new();
    for v in [42f32, 24., 0., 0.] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    e.write_buffer(&instances, 0, &data).unwrap();
    e.write_buffer(&indices, 0, &[0, 0, 1, 0, 2, 0, 0, 0])
        .unwrap();
    let mut tickets = Vec::new();
    for first in [2, 0] {
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&view),
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
                    width: 8,
                    height: 8,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_vertex_buffer(0, &positions, 0..48).unwrap();
        pass.set_vertex_buffer(1, &instances, 0..16).unwrap();
        pass.set_vertex_buffer(2, &instances, 0..16).unwrap();
        pass.draw(3..6, first..first + 1).unwrap();
        pass.finish().unwrap();
        e.texture_to_buffer(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            &out[tickets.len()],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
        tickets.push(e.readback_buffer(&out[tickets.len()], 0..256).unwrap());
    }
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers this readback.
    for ticket in tickets {
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        assert_eq!(
            &bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4],
            &[255, 0, 0, 255]
        );
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_zero_stride() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let name = native.0.adapter_name().unwrap();
    let broken = name.contains("ANGLE 2.1.22762")
        && name.contains("AMD Radeon 780M Graphics (0x00001900) Direct3D11")
        && name.contains("D3D11-32.0.21028.2002");
    let d = block_on(open(native));
    let _ = broken;
    block_on(zero_stride(&d, false));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_zero_stride() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let name = native.0.adapter_name().unwrap();
    let broken = name.contains("ANGLE 2.1.22762")
        && name.contains("AMD Radeon 780M Graphics (0x00001900) Direct3D11")
        && name.contains("D3D11-32.0.21028.2002");
    let d = block_on(open(native));
    let _ = broken;
    block_on(zero_stride(&d, false));
}
