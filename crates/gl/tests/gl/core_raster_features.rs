//! Public Core immediate raster workload shared by WGL and WebGL2 fixtures.
//! Actual color readback checks both fragment output locations around a hole.
use crate::{Gl, GlInstance};
use fluxel_rhi_core::api::*;
use std::sync::Arc;

// Keep fixture module paths stable while separating the registration catalog.
include!("core_feature_registry.rs");

async fn open(native: GlInstance) -> Device<Gl> {
    let id = InstanceId::allocate().unwrap();
    // SAFETY: the adopted context and its canvas/window remain alive until the device is dropped.
    let instance = unsafe { Instance::<Gl>::from_native(id, native) };
    let device = instance
        .open(DeviceRequest {
            adapter: AdapterId {
                instance: id,
                ordinal: 0,
            },
            required_features: vec![],
        })
        .await
        .unwrap();
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;
        // SAFETY: the fixture owns this device on its context thread.
        let native = unsafe { device.native_ref() };
        native
            .0
            .with_native("fixture renderer identity", |provider| {
                let info = provider.discovery().context();
                eprintln!(
                    "GL_VERSION={}; GL_VENDOR={}; GL_RENDERER={}",
                    info.version(),
                    info.vendor(),
                    info.renderer()
                );
                Ok(())
            })
            .unwrap();
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        // SAFETY: the fixture reads immutable discovery facts from its device.
        let native = unsafe { device.native_ref() };
        let discovery = native.0.owner.discovery_snapshot();
        wasm_bindgen_test::console_log!(
            "WebGL GPU: {}; {:?}",
            discovery.context().renderer(),
            discovery.context().flags()
        );
    }
    device
}

async fn mrt_hole(d: &Device<Gl>) {
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
    let state = ColorTargetState {
        format: TextureFormat::Rgba8Unorm,
        blend: None,
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
            color_targets: vec![Some(state.clone()), None, Some(state)],
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
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
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
    for (ticket, expected) in [(at, [255, 0, 0, 255]), (bt, [0, 255, 0, 255])] {
        // SAFETY: this completion point covers the draw and both readbacks above.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        assert_eq!(&bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4], &expected);
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_mrt_hole() {
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
    block_on(mrt_hole(&d));
    block_on(dynamic::depth_stencil_and_dynamic(&d));
    block_on(native_surface_features::mixed_surface(&d));
    block_on(native_surface_features::integer_clear(&d));
    block_on(native_surface_features::msaa_surface(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_mrt_hole() {
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
    let version = gl
        .get_parameter(web_sys::WebGl2RenderingContext::VERSION)
        .unwrap();
    let renderer = gl
        .get_extension("WEBGL_debug_renderer_info")
        .unwrap()
        .map(|_| gl.get_parameter(0x9246).unwrap())
        .unwrap_or_else(|| {
            gl.get_parameter(web_sys::WebGl2RenderingContext::RENDERER)
                .unwrap()
        });
    web_sys::console::log_1(
        &format!("WEBGL_VERSION={version:?}; WEBGL_RENDERER={renderer:?}").into(),
    );
    let d = open(native).await;
    mrt_hole(&d).await;
    dynamic::depth_stencil_and_dynamic(&d).await;
    let gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::WebGl2RenderingContext>()
        .unwrap();
    assert_eq!(gl.get_error(), web_sys::WebGl2RenderingContext::NO_ERROR);
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_raster_features() {
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
    block_on(mrt_hole(&d));
    block_on(dynamic::depth_stencil_and_dynamic(&d));
}
