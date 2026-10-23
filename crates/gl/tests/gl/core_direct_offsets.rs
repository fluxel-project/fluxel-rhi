//! Direct base vertex and first instance Core readback checks.
use super::*;
async fn direct_offsets(d: &Device<Gl>, indexed: bool, indirect_usable: bool) {
    let shader_source = br#"struct Output { @builtin(position) position: vec4f, @location(0) @interpolate(flat) instance: u32 };
    @vertex fn vs(@location(0) pos:vec2f, @location(1) val:f32, @builtin(instance_index) instance:u32, @builtin(vertex_index) vertex:u32)->Output {
      let ok = val == f32(instance) && vertex >= 3u; var result: Output; result.position = vec4f(pos, select(2.0, 0.5, ok), 1.0); result.instance = instance; return result;
    } @fragment fn fs(input:Output)->@location(0) vec4f { return vec4f(f32(input.instance) / 4., 1., 0., 1.); }"#;
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
                    stride: 4,
                    step_mode: VertexStepMode::Instance,
                    attributes: vec![VertexAttribute {
                        location: 1,
                        offset: 0,
                        format: VertexFormat::Float32(1),
                    }],
                },
            ],
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
    let args = d
        .capabilities()
        .indirect_draw
        .then(|| buffer(20, BufferUsage::INDIRECT | BufferUsage::COPY_DST));
    let out: Vec<_> = (0..4)
        .map(|_| buffer(256, BufferUsage::COPY_DST | BufferUsage::COPY_SRC))
        .collect();
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
    for v in [0f32, 1., 2., 3.] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    e.write_buffer(&instances, 0, &data).unwrap();
    e.write_buffer(&indices, 0, &[0, 0, 1, 0, 2, 0, 0, 0])
        .unwrap();
    let mut tickets = Vec::new();
    for (first, indirect) in [(2, false), (2, true), (0, false), (0, true)] {
        if indirect
            && (!indirect_usable
                || !d.capabilities().indirect_draw
                || first != 0 && !d.capabilities().indirect_first_instance)
        {
            continue;
        }
        if indirect {
            let values: Vec<u32> = if indexed {
                vec![3, 1, 0, 3, first]
            } else {
                vec![3, 1, 3, first]
            };
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            e.write_buffer(args.as_ref().unwrap(), 0, &bytes).unwrap();
        }
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
        if indexed {
            pass.set_index_buffer(&indices, 0..8, IndexFormat::Uint16)
                .unwrap();
        }
        if indirect {
            d.raster_extras()
                .unwrap()
                .draw_indirect(&mut pass, args.as_ref().unwrap(), 0, indexed)
                .unwrap();
        } else if indexed {
            pass.draw_indexed(0..3, 3, first..first + 1).unwrap();
        } else {
            pass.draw(3..6, first..first + 1).unwrap();
        }
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
        tickets.push((
            e.readback_buffer(&out[tickets.len()], 0..256).unwrap(),
            first,
        ));
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
    for (ticket, first) in tickets {
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        assert_eq!(
            &bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4],
            &[if first == 2 { 128 } else { 0 }, 255, 0, 255]
        );
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_direct_offsets() {
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
    fn block_on<F: Future>(f: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let name = native.0.adapter_name().unwrap();
    let broken = name.contains("ANGLE 2.1.22762")
        && name.contains("AMD Radeon 780M Graphics (0x00001900) Direct3D11")
        && name.contains("D3D11-32.0.21028.2002");
    let d = block_on(open(native));
    eprintln!(
        "indirect_first_instance={}",
        d.capabilities().indirect_first_instance
    );
    if std::env::var_os("FLUXEL_EXPECT_INDIRECT_FIRST_INSTANCE").is_some() {
        assert!(d.capabilities().indirect_first_instance);
    }
    block_on(direct_offsets(&d, false, !broken));
    block_on(direct_offsets(&d, true, !broken));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_direct_offsets() {
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
    fn block_on<F: Future>(f: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let name = native.0.adapter_name().unwrap();
    let broken = name.contains("ANGLE 2.1.22762")
        && name.contains("AMD Radeon 780M Graphics (0x00001900) Direct3D11")
        && name.contains("D3D11-32.0.21028.2002");
    let d = block_on(open(native));
    block_on(direct_offsets(&d, false, !broken));
    block_on(direct_offsets(&d, true, !broken));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_direct_offsets() {
    use wasm_bindgen::JsCast;
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(8);
    canvas.set_height(8);
    let native = GlInstance::from_webgl2_canvas(canvas.clone()).unwrap();
    let d = open(native).await;
    direct_offsets(&d, false, true).await;
    let gl: web_sys::WebGl2RenderingContext = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap();
    let has_base_vertex = gl
        .get_extension("WEBGL_draw_instanced_base_vertex_base_instance")
        .unwrap()
        .is_some();
    eprintln!("WebGL base vertex extension available: {has_base_vertex}");
    if has_base_vertex {
        direct_offsets(&d, true, true).await;
    }
}
