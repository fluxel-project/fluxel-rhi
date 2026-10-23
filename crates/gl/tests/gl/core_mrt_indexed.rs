//! Core MRT independent color mask and blending GPU readback.
use super::*;

async fn indexed_mrt(d: &Device<Gl>) {
    let source = b"struct O { @builtin(position) p: vec4f }; struct F { @location(0) a: vec4f, @location(2) b: vec4f }; @vertex fn vs(@builtin(vertex_index) i:u32)->O { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); var o:O; o.p=vec4f(p[i],0.5,1.); return o; } @fragment fn fs()->F { var o:F; o.a=vec4f(1.,0.,0.,1.); o.b=vec4f(0.,1.,0.,1.); return o; }";
    let shader = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source.as_slice()),
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
    let component = BlendComponent {
        src_factor: BlendFactor::Constant,
        dst_factor: BlendFactor::OneMinusConstant,
        operation: BlendOperation::Add,
    };
    let first = ColorTargetState {
        format: TextureFormat::Rgba8Unorm,
        blend: None,
        write_mask: 9,
    };
    let second = ColorTargetState {
        format: TextureFormat::Rgba8Unorm,
        blend: Some(BlendState {
            color: component,
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::Zero,
                operation: BlendOperation::Add,
            },
        }),
        write_mask: 15,
    };
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
            vertex_buffers: vec![],
            color_targets: vec![Some(first), None, Some(second)],
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
                mask: u32::MAX as u64,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
    let texture = || {
        d.create_texture(&TextureDesc {
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
        .unwrap()
    };
    let a = texture();
    let b = texture();
    let view = |t: &Texture<Gl>| {
        d.create_view(
            t,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap()
    };
    let av = view(&a);
    let bv = view(&b);
    let attachment = |v| {
        Some(ColorAttachment {
            view: AttachmentView::Texture(v),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 1., 1.])),
            store: StoreOp::Store,
        })
    };
    let colors = [attachment(&av), None, attachment(&bv)];
    let output = || {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let ab = output();
    let bb = output();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 91).unwrap();
    // SAFETY: both textures, buffers, pipeline and adopted context outlive completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin immediate commands"));
    let e = run.commands();
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
    pass.blend_constant([0.25, 0.25, 0.25, 0.25]).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    for (texture, out) in [(&a, &ab), (&b, &bb)] {
        e.texture_to_buffer(
            TextureRegion {
                texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            out,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
    }
    let at = e.readback_buffer(&ab, 0..256).unwrap();
    let bt = e.readback_buffer(&bb, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish immediate commands"),
    };
    d.wait_completion(&point).await.unwrap();
    for (ticket, expected) in [(at, [255u8, 0, 255, 255]), (bt, [0, 64, 191, 255])] {
        // SAFETY: this completion point covers the draw and both readbacks above.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        let actual = &bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4];
        eprintln!("MRT expected {expected:?}, observed {actual:?}");
        for (a, b) in actual.iter().zip(expected) {
            assert!((i16::from(*a) - i16::from(b)).abs() <= 1);
        }
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_mrt_indexed() {
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
    let d = block_on(open(native));
    block_on(indexed_mrt(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_mrt_indexed() {
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
    let d = block_on(open(native));
    use crate::backend::gl::api::{GlFamilyApi as _, GlFamilyProfile};
    // SAFETY: the fixture owns this device and current context on this thread.
    let native = unsafe { d.native_ref() };
    let available = native.0.with_native("inspect indexed blend support", |provider| {
        let discovery = provider.discovery();
        let core = matches!(discovery.context().profile(), GlFamilyProfile::Embedded { major, minor } if major > 3 || (major == 3 && minor >= 2));
        let extension = discovery.extensions().raw_reported_names().any(|name| name == "GL_OES_draw_buffers_indexed" || name == "GL_EXT_draw_buffers_indexed");
        Ok(core || extension)
    }).unwrap();
    if !available {
        eprintln!("SKIP: GLES indexed draw-buffer blend extension unavailable");
        return;
    }
    block_on(indexed_mrt(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_mrt_indexed() {
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
    let gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::WebGl2RenderingContext>()
        .unwrap();
    if gl
        .get_extension("OES_draw_buffers_indexed")
        .unwrap()
        .is_none()
    {
        println!("SKIP: OES_draw_buffers_indexed unavailable");
        return;
    }
    let d = open(native).await;
    indexed_mrt(&d).await;
    assert_eq!(gl.get_error(), web_sys::WebGl2RenderingContext::NO_ERROR);
}
