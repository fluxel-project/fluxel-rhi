//! Browser integration coverage for WebGL2 canvas reconfiguration and leases.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]

#[allow(dead_code)]
#[path = "webgl2_support.rs"]
mod support;

use fluxel_rhi::{
    api::*,
    gl::{Gl, GlInstance},
};
use wasm_bindgen::{JsCast, closure::Closure};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn config(extent: [u32; 2]) -> SurfaceConfig {
    SurfaceConfig {
        extent,
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    }
}

async fn open_device(native: GlInstance) -> Device<Gl> {
    // SAFETY: callers retain the canvas-backed owner for the test lifetime.
    let instance = unsafe { Instance::<Gl>::from_native(InstanceId::allocate().unwrap(), native) };
    let adapter = instance.enumerate().unwrap().into_iter().next().unwrap();
    instance
        .open(DeviceRequest {
            adapter: adapter.id,
            required_features: vec![],
        })
        .await
        .unwrap()
}

#[wasm_bindgen_test(async)]
async fn canvas_resize_reconfigures_generation_and_preserves_exclusive_leases() {
    let canvas = support::canvas(16, 16);
    let native = GlInstance::from_webgl2_canvas(canvas.clone()).expect("WebGL2 instance");
    let device = open_device(native).await;
    let mut surface = Gl::surface(&device).expect("canvas surface");

    surface.configure(&config([16, 16])).await.unwrap();
    let first_generation = surface.generation();
    let first = surface.acquire().await.unwrap();
    assert!(first.is_current());

    // Reconfiguration cannot steal the default framebuffer while a lease is
    // live; dropping it releases exactly that acquisition for the host resize.
    assert!(matches!(
        surface.configure(&config([32, 24])).await,
        Err(Error::InvalidState)
    ));
    drop(first);

    surface.configure(&config([32, 24])).await.unwrap();
    assert!(surface.generation() > first_generation);
    assert_eq!([canvas.width(), canvas.height()], [32, 24]);

    let resized = surface.acquire().await.unwrap();
    assert!(resized.is_current());
    drop(resized);
    canvas.remove();
}

#[wasm_bindgen_test(async)]
async fn externally_resized_canvas_requires_reconfiguration_before_acquire() {
    let canvas = support::canvas(16, 16);
    let native = GlInstance::from_webgl2_canvas(canvas.clone()).expect("WebGL2 instance");
    let device = open_device(native).await;
    let mut surface = Gl::surface(&device).expect("canvas surface");
    surface.configure(&config([16, 16])).await.unwrap();

    // The host owns browser canvas lifecycle. An out-of-band backing-store
    // resize must be observed as stale until the adapter reconfigures RHI.
    canvas.set_width(32);
    canvas.set_height(24);
    assert!(matches!(
        surface.acquire().await,
        Err(Error::SurfaceOutdated)
    ));

    surface.configure(&config([32, 24])).await.unwrap();
    let image = surface.acquire().await.unwrap();
    assert!(image.is_current());
    drop(image);
    canvas.remove();
}

#[wasm_bindgen_test(async)]
async fn host_context_loss_notification_invalidates_live_surface_leases() {
    let canvas = support::canvas(16, 16);
    let native = GlInstance::from_webgl2_canvas(canvas.clone()).expect("WebGL2 instance");
    let host_lifecycle = native.clone();
    let device = open_device(native).await;
    let mut surface = Gl::surface(&device).expect("canvas surface");
    surface.configure(&config([16, 16])).await.unwrap();
    let image = surface.acquire().await.unwrap();
    assert!(image.is_current());

    host_lifecycle.notify_context_lost();

    assert!(!image.is_current());
    assert!(matches!(device.wait_idle().await, Err(Error::DeviceLost)));
    drop(image);
    canvas.remove();
}

#[wasm_bindgen_test(async)]
async fn context_loss_during_completion_poll_reports_device_lost() {
    let canvas = support::canvas(16, 16);
    let native = GlInstance::from_webgl2_canvas(canvas.clone()).expect("WebGL2 instance");
    let device = open_device(native).await;
    let mut surface = Gl::surface(&device).expect("canvas surface");
    surface.configure(&config([16, 16])).await.unwrap();
    let image = surface.acquire().await.unwrap();

    let raw: web_sys::WebGl2RenderingContext = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap();
    let extension = raw
        .get_extension("WEBGL_lose_context")
        .unwrap()
        .expect("WEBGL_lose_context extension");
    // WebGL extension objects have no global constructor for instanceof checks;
    // the successful named lookup supplies the WebGLLoseContext contract.
    let lose_context: web_sys::WebglLoseContext = extension.unchecked_into();
    // The zero-delay loss runs after wait_idle has created and registered its
    // real GLsync. The browser's next nonblocking poll must classify WAIT_FAILED
    // on the lost context as DeviceLost, without deleting the dead GLsync.
    let callback = Closure::once_into_js(move || lose_context.lose_context());
    web_sys::window()
        .unwrap()
        .set_timeout_with_callback_and_timeout_and_arguments_0(callback.unchecked_ref(), 0)
        .unwrap();

    assert!(matches!(device.wait_idle().await, Err(Error::DeviceLost)));
    assert!(!image.is_current());
    drop(image);
    canvas.remove();
}
