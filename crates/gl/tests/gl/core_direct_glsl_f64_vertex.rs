//! Desktop GL Float64 vertex attributes through a direct GLSL raster pipeline.

use super::*;

async fn direct_glsl_f64_vertex(d: &Device<Gl>) {
    let shader = |source: &'static [u8], stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Glsl(410),
        bytes: Arc::from(source),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = d
        .shader(shader(
            br#"#version 410 core
layout(location = 0) in dvec2 position;
void main() {
    gl_Position = vec4(vec2(position), 0.5, 1.0);
}
"#,
            ShaderStage::Vertex,
        ))
        .await
        .unwrap();
    let fragment = d
        .shader(shader(
            br#"#version 410 core
layout(location = 0) out vec4 color;
void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }
"#,
            ShaderStage::Fragment,
        ))
        .await
        .unwrap();
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
                entry: "main".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment,
                entry: "main".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![VertexBufferLayout {
                stride: 16,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float64(2),
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
                mask: u64::MAX,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
    let color = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let color_view = d
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
    let vertices = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 48,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 1).unwrap();
    // SAFETY: fixture resources remain live until the completion point below.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    let commands = run.commands();
    let mut vertex_bytes = Vec::with_capacity(48);
    for value in [-1.0_f64, -1.0, 3.0, -1.0, -1.0, 3.0] {
        vertex_bytes.extend_from_slice(&value.to_le_bytes());
    }
    commands.write_buffer(&vertices, 0, &vertex_bytes).unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&color_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
        store: StoreOp::Store,
    })];
    let mut pass = commands
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_vertex_buffer(0, &vertices, 0..48).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    commands
        .texture_to_buffer(
            TextureRegion {
                texture: &color,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 16,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let ticket = commands.readback_buffer(&readback, 0..64).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: the completion point covers the texture readback.
    let pixels = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for pixel in pixels.chunks_exact(4) {
        assert_eq!(pixel, &[255, 0, 0, 255]);
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context with Float64 vertex attributes"]
fn wgl_core_direct_glsl_f64_vertex() {
    use fluxel_host::{Window, WindowConfig};
    use std::{
        future::Future,
        task::{Context, Poll, Wake, Waker},
    };

    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    fn block_on<F: Future>(future: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut context = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::park(),
            }
        }
    }

    let window =
        Window::new(WindowConfig::new("GL Float64 vertex fixture", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let device = block_on(open(native));
    block_on(direct_glsl_f64_vertex(&device));
}
