//! WebGL2 immediate execution for the sequential fallback.

use super::*;

/// Runs the sequential fallback through WebGL2's owner-thread immediate SPI.
/// No worker or secondary-command capability is advertised or emulated.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub async fn run_webgl2_multithreaded_recording(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> Result<()> {
    use fluxel_rhi::gl::{Gl, GlInstance};

    if frames == 0 {
        return Err(Error::InvalidRange);
    }
    let native = GlInstance::from_webgl2_canvas(canvas.clone())?;
    // SAFETY: the browser canvas remains owned by this future; all GL access
    // stays on the JavaScript owner thread through the typed device/surface.
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
    let mut scene = Sequential::<Gl>::new(&device, format).await?;
    let scope = utils::GpuScope::new(&device);
    for _ in 0..frames {
        let image = surface.acquire().await?;
        let (uploaded, outcome) = utils::execute_frame(
            &device,
            lane,
            image,
            CompletionRequest::End,
            |encoder, image| scene.encode(&device, image, encoder, extent),
        )?;
        let point = utils::execution_end_checkpoint(&outcome)?;
        if uploaded {
            scene.mark_uploaded();
        }
        device.wait_completion(&point).await?;
    }
    scope.finish().await;
    Ok(())
}
