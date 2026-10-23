//! Native buffer map lease and GPU copy readback.
use super::*;
async fn mapping(d: &Device<Gl>) {
    let make = |usage, memory| {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage,
            memory,
        })
        .unwrap()
    };
    let upload = make(
        BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Upload,
    );
    let download = make(
        BufferUsage::MAP_READ | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        MemoryClass::Readback,
    );
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    let persistent = {
        // SAFETY: only private backend capability metadata is inspected.
        let buffer = unsafe { download.native_ref() };
        buffer
            .state
            .with_native("inspect mapping fixture", |provider| {
                provider
                    .persistent_buffer_lease(buffer.id, false)
                    .map(|ptr| ptr.is_some())
            })
            .unwrap()
    };
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    let persistent = false;
    eprintln!("native persistent coherent mapping: {persistent}");
    // SAFETY: neither rejected request accesses the buffer.
    assert!(matches!(
        unsafe { d.map_buffer(&upload, 8..8, MapMode::Write).await },
        Err(Error::InvalidRange)
    ));
    assert!(matches!(
        unsafe { d.map_buffer(&upload, 0..4, MapMode::Read).await },
        Err(Error::Unsupported(_))
    ));
    // SAFETY: both native references belong to this device and there is no
    // overlapping GPU work. Dropping an unpolled native future must unmap it.
    let cancelled = <Gl as MappingBackend>::map(
        unsafe { d.native_ref() },
        unsafe { upload.native_ref() },
        0..4,
        MapMode::Write,
    );
    drop(cancelled);
    // SAFETY: no GPU accesses overlap these host leases.
    let mut map = unsafe { d.map_buffer(&upload, 8..40, MapMode::Write).await }.unwrap();
    for (i, byte) in map.bytes_mut().unwrap().iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(7).wrapping_add(3);
    }
    assert_eq!(map.bytes()[1], 10);
    map.flush(3..20).unwrap();
    map.flush(0..32).unwrap();
    assert!(matches!(map.flush(0..33), Err(Error::InvalidRange)));
    // SAFETY: this attempts a second lease solely to verify rejection before access.
    assert!(unsafe { d.map_buffer(&upload, 0..4, MapMode::Write).await }.is_err());
    drop(map);
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 801).unwrap();
    // SAFETY: all resources outlive completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    run.commands()
        .copy_buffer(&upload, 8, &download, 16, 32)
        .unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: the completion point covers the copy.
    let mut read = unsafe { d.map_buffer(&download, 16..48, MapMode::Read).await }.unwrap();
    read.invalidate(2..20).unwrap();
    assert!(matches!(read.invalidate(0..33), Err(Error::InvalidRange)));
    let expected: Vec<u8> = (0..32)
        .map(|i| (i as u8).wrapping_mul(7).wrapping_add(3))
        .collect();
    assert_eq!(read.bytes(), expected);
    // SAFETY: the prior copy completed and the host does not access either
    // mapped range while the following GPU copy executes.
    let mut update = unsafe { d.map_buffer(&upload, 8..40, MapMode::Write).await }.unwrap();
    update.bytes_mut().unwrap().fill(0xab);
    update.flush(0..32).unwrap();
    drop(update);
    let plan = common::single_commands_plan(&fixture, 802).unwrap();
    // SAFETY: resources remain alive until this copy completes.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    run.commands()
        .copy_buffer(&upload, 8, &download, 16, 32)
        .unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    read.invalidate(4..12).unwrap();
    assert_eq!(&read.bytes()[4..12], &[0xab; 8]);
    // Coherent persistent mappings can expose the entire completed GPU write;
    // snapshot profiles only refresh the requested range.
    if persistent {
        assert_eq!(read.bytes(), &[0xab; 32]);
    } else {
        assert_eq!(&read.bytes()[..4], &expected[..4]);
        assert_eq!(&read.bytes()[12..], &expected[12..]);
    }
    drop(read);
    // SAFETY: the previous copy completed; the host leaves this lease idle
    // while the GPU overwrites the same allocation below.
    let mut lease = unsafe { d.map_buffer(&upload, 8..40, MapMode::Write).await }.unwrap();
    lease.bytes_mut().unwrap().fill(0x42);
    lease.flush(0..32).unwrap();
    let plan = common::single_commands_plan(&fixture, 803).unwrap();
    // SAFETY: all referenced resources outlive the GPU copy.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    run.commands()
        .copy_buffer(&download, 16, &upload, 8, 32)
        .unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    drop(lease);
    let plan = common::single_commands_plan(&fixture, 804).unwrap();
    // SAFETY: the preceding copy completed and resources outlive this run.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    run.commands()
        .copy_buffer(&upload, 8, &download, 16, 32)
        .unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers the readback allocation.
    let read = unsafe { d.map_buffer(&download, 16..48, MapMode::Read).await }.unwrap();
    assert_eq!(read.bytes(), &[0xab; 32]);
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_mapping() {
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
    let window = Window::new(WindowConfig::new("GL mapping fixture", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let name = native.0.adapter_name().unwrap();
    let broken = name.contains("ANGLE 2.1.22762")
        && name.contains("AMD Radeon 780M Graphics (0x00001900) Direct3D11")
        && name.contains("D3D11-32.0.21028.2002");
    let d = block_on(open(native));
    let _ = broken;
    block_on(mapping(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_mapping() {
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
    let _ = broken;
    block_on(mapping(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_mapping() {
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
    let d = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    mapping(&d).await;
}
