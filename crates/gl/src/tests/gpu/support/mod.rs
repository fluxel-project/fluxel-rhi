//! Shared context setup, identity evidence and wake-driven GL test executor.
use super::*;

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::{
        sync::Arc,
        task::{Context, Poll, Wake, Waker},
    };
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

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) async fn webgl2_device(extent: [u32; 2]) -> super::Device<super::Gl> {
    use wasm_bindgen::JsCast;
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(extent[0]);
    canvas.set_height(extent[1]);
    open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await
}

pub(super) async fn open(native: GlInstance) -> Device<Gl> {
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
        use crate::backend::api::GlFamilyApi as _;
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

/// Canonical unculled triangle-list state; non-default raster cases stay explicit.
pub(super) fn triangle_list_state() -> PrimitiveState {
    PrimitiveState {
        topology: PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: FrontFace::Ccw,
        cull_mode: CullMode::None,
        unclipped_depth: false,
        polygon_mode: PolygonMode::Fill,
        conservative: false,
    }
}

/// A one-sample pipeline retaining the fixture's exact sample-mask spelling.
pub(super) fn single_sample_state(mask: u64) -> MultisampleState {
    MultisampleState {
        count: 1,
        mask,
        alpha_to_coverage_enabled: false,
    }
}

/// A target with no blending and all four color write bits enabled.
pub(super) fn unblended_target(format: TextureFormat) -> ColorTargetState {
    ColorTargetState {
        format,
        blend: None,
        write_mask: 15,
    }
}
