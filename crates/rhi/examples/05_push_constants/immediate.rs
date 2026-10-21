//! GL execution for the sphere-ring dynamic-uniform fallback.
//!
//! GL has no public push constants. The existing WebGPU WGSL and one aligned
//! uniform record per draw preserve the sixteen-sphere workload. The backend
//! issues each command on the context owner thread and presents the acquired
//! lease; it stores no CPU command stream or deferred submission.

#[cfg(any(windows, all(target_os = "linux", feature = "native-gles-egl")))]
use std::{cell::RefCell, rc::Rc};

use fluxel_rhi::api::*;
use fluxel_rhi::gl::Gl;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
use fluxel_rhi::gl::GlInstance;

use super::{PushWorkload, utils};

fn issue_frame(
    device: &Device<Gl>,
    lane: LaneId,
    workload: &mut PushWorkload<Gl>,
    image: SurfaceImage<Gl>,
    extent: [u32; 2],
) -> Result<()> {
    let (_, outcome) = utils::execute_frame(
        device,
        lane,
        image,
        CompletionRequest::None,
        |encoder, image| workload.issue(encoder, image, extent),
    )?;
    utils::validate_execution_outcome(&outcome)?;
    workload.commit_frame();
    Ok(())
}

#[cfg(any(windows, all(target_os = "linux", feature = "native-gles-egl")))]
pub(super) fn run() -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    };
    let workload = Rc::new(RefCell::new(None));
    let dropped_workload = Rc::clone(&workload);
    utils::gl_api::run_native(
        "Push constants",
        config.clone(),
        move || *dropped_workload.borrow_mut() = None,
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("GL graphics lane"))?;
            if workload.borrow().is_none() {
                *workload.borrow_mut() = Some(utils::block_on(PushWorkload::<Gl>::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?);
            }
            issue_frame(
                device,
                lane,
                workload.borrow_mut().as_mut().ok_or(Error::InvalidState)?,
                image,
                extent,
            )
        },
    )
}

/// Runs the same WGSL sphere ring on a caller-owned WebGL2 canvas.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgl2_push_constants(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    async fn run(canvas: web_sys::HtmlCanvasElement, frames: u32) -> Result<()> {
        if frames == 0 {
            return Err(Error::InvalidRange);
        }
        let native = GlInstance::from_webgl2_canvas(canvas.clone())?;
        // SAFETY: this newly created owner remains retained by the typed
        // instance, device, and surface for the browser thread's whole run.
        let instance = unsafe { Instance::<Gl>::from_native(InstanceId::allocate()?, native) };
        let adapter = instance
            .enumerate()?
            .into_iter()
            .next()
            .ok_or(Error::Unsupported("WebGL2 adapter"))?;
        let device = instance
            .open(DeviceRequest {
                adapter: adapter.id,
                required_features: vec![],
            })
            .await?;
        let lane = device
            .lanes()
            .iter()
            .find(|lane| lane.class == LaneClass::Graphics)
            .map(|lane| lane.id)
            .ok_or(Error::Unsupported("WebGL2 graphics lane"))?;
        let extent = [canvas.width(), canvas.height()];
        if extent.contains(&0) {
            return Err(Error::SurfaceLost);
        }
        let config = SurfaceConfig {
            extent,
            format: TextureFormat::Rgba8Unorm,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 1,
        };
        let mut surface = Gl::surface(&device)?;
        surface.configure(&config).await?;
        let mut workload = PushWorkload::<Gl>::new(&device, lane, config.format, extent).await?;
        let scope = utils::GpuScope::new(&device);
        for _ in 0..frames {
            let image = surface.acquire().await?;
            issue_frame(&device, lane, &mut workload, image, extent)?;
        }
        scope.finish().await;
        Ok(())
    }
    run(canvas, frames)
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}
