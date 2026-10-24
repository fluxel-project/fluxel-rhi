//! Timestamp-query boundaries around native OVR multiview raster passes.
//!
//! `QueryCounter(TIMESTAMP)` is legal before the multiview framebuffer is bound
//! and after it is unbound. The OVR restriction remains enforced for attempts
//! to write a timestamp while the multiview pass is active.
use super::*;

fn report(message: &str) {
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::console_log!("{}", message);
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{message}");
}

async fn multiview_timestamp_boundary(device: &Device<Gl>) {
    if !device.capabilities().multiview {
        report("SKIP multiview timestamp boundary: OVR multiview unavailable");
        return;
    }
    let Some(queries) = device.queries(QueryKind::Timestamp) else {
        report("SKIP multiview timestamp boundary: timestamp capability unavailable");
        return;
    };
    let set = queries
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 2,
        })
        .unwrap();
    let texture = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let view = device
        .create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2Array,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..2,
            },
        )
        .unwrap();
    let buffer = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 32,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device,
        lane: device.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 532).unwrap();
    // SAFETY: the native context and all resources live through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin multiview timestamp boundary"));
    let encoder = run.commands();
    encoder.write_buffer(&buffer, 0, &[0xad; 32]).unwrap();
    queries.reset(encoder, &set, 0..2).unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
        store: StoreOp::Store,
    })];
    let pass = encoder
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
            multiview_mask: Some(3),
            timestamp_writes: Some(PassTimestampWrites {
                set: &set,
                begin: Some(0),
                end: Some(1),
            }),
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    // No timestamp command is issued while the multiview FBO is active.
    pass.finish().unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("multiview timestamp boundary did not issue"),
    };
    device.wait_completion(&point).await.unwrap();
    let plan = common::single_commands_plan(&fixture, 533).unwrap();
    // SAFETY: the timestamp producer completion is established above.
    let mut resolve_session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut resolve_run = resolve_session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin multiview timestamp resolve"));
    let resolve = resolve_run.commands();
    queries.resolve(resolve, &set, 0..2, &buffer, 8).unwrap();
    let ticket = resolve.readback_buffer(&buffer, 0..32).unwrap();
    assert!(matches!(
        resolve_run.close(),
        ImmediateRunOutcome::Issued(_)
    ));
    let resolved = match resolve_session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("multiview timestamp resolve did not issue"),
    };
    device.wait_completion(&resolved).await.unwrap();
    // SAFETY: the resolve completion covers this readback ticket.
    let bytes = unsafe { device.read_after(ticket, &resolved).await }
        .unwrap()
        .bytes;
    assert_eq!(&bytes[..8], &[0xad; 8]);
    assert_eq!(&bytes[24..], &[0xad; 8]);
    let begin = u64::from_ne_bytes(bytes[8..16].try_into().unwrap());
    let end = u64::from_ne_bytes(bytes[16..24].try_into().unwrap());
    if begin == u64::MAX || end == u64::MAX {
        report("SKIP multiview timestamp: disjoint sentinel observed");
        return;
    }
    report(&format!(
        "multiview timestamp values: begin={begin}, end={end}"
    ));
    assert_ne!(begin, 0, "begin timestamp was not written");
    assert_ne!(end, 0, "end timestamp was not written");
    if queries
        .timestamp_clock(device.lanes()[0].id)
        .unwrap()
        .valid_bits
        == 64
    {
        assert!(end >= begin, "timestamp end preceded begin: {begin}, {end}");
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, Wake, Waker};
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
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

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires an explicit WGL hardware-test run"]
fn wgl_core_multiview_timestamp_boundary() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL multiview timestamp", 1024, 768).unwrap()).unwrap();
    // SAFETY: the WGL window remains live through the fixture.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let device = block_on(open(native));
    block_on(multiview_timestamp_boundary(&device));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_multiview_timestamp_boundary() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for multiview timestamp");
    let device = block_on(open(native));
    block_on(multiview_timestamp_boundary(&device));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_multiview_timestamp_boundary() {
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
    let device = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    multiview_timestamp_boundary(&device).await;
}
