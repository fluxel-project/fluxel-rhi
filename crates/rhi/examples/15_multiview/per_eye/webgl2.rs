//! WebGL2 immediate surface loop for the sequential stereo fallback.

use super::*;

/// Runs the ordinary two-pass stereo fallback on WebGL2's immediate context.
/// Native multiview remains unavailable and is not emulated through command replay.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub async fn run_webgl2_multiview(canvas: web_sys::HtmlCanvasElement, frames: u32) -> Result<()> {
    use fluxel_rhi::gl::{Gl, GlInstance};

    if frames == 0 {
        return Err(Error::InvalidRange);
    }
    let native = GlInstance::from_webgl2_canvas(canvas.clone())?;
    // SAFETY: the canvas-backed owner is retained by the typed instance,
    // device, and surface while every GL operation stays on this browser thread.
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
    if device.capabilities().multiview {
        return Err(Error::InvalidState);
    }
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
    let format = TextureFormat::Rgba8Unorm;
    let mut surface = Gl::surface(&device)?;
    surface
        .configure(&SurfaceConfig {
            extent,
            format,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 1,
        })
        .await?;
    let mut scene = StereoFallback::<Gl>::new(&device, format, extent).await?;
    let scope = utils::GpuScope::new(&device);
    for _ in 0..frames {
        let image = surface.acquire().await?;
        let (uploaded, _outcome) = utils::execute_frame(
            &device,
            lane,
            image,
            CompletionRequest::None,
            |encoder, image| scene.encode(encoder, image),
        )?;
        if uploaded {
            scene.mark_uploaded();
        }
    }
    scope.finish().await;
    Ok(())
}
