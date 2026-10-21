//! WebGPU submission and presentation for the sequential fallback.

use super::*;

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_multithreaded_recording(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    use fluxel_rhi::webgpu::{WebGpu, WebGpuInstance};
    async fn run(canvas: web_sys::HtmlCanvasElement, frames: u32) -> Result<()> {
        let d = WebGpuInstance::new()?.open_default().await?;
        let lane = d
            .lanes()
            .iter()
            .find(|x| x.class == LaneClass::Graphics)
            .map(|x| x.id)
            .ok_or(Error::Unsupported("WebGPU graphics lane"))?;
        let extent = [canvas.width(), canvas.height()];
        let mut surface = WebGpu::surface_from_canvas(&d, canvas)?;
        let format = surface
            .capabilities()?
            .formats
            .into_iter()
            .next()
            .ok_or(Error::Unsupported("WebGPU canvas format"))?;
        surface
            .configure(&SurfaceConfig {
                extent,
                format,
                present_mode: PresentMode::Fifo,
                alpha_mode: CompositeAlphaMode::Opaque,
                color_space: ColorSpace::Srgb,
                usage: SurfaceUsage::RENDER_ATTACHMENT,
                maximum_frame_latency: 2,
            })
            .await?;
        let mut scene = Sequential::<WebGpu>::new(&d, format).await?;
        for _ in 0..frames {
            let image = surface.acquire().await?;
            let mut encoder = d.encoder(lane)?;
            let uploaded_geometry = scene.encode(&d, &image, &mut encoder, extent)?;
            let work = encoder.finish()?;
            utils::submit_and_present_async(&d, lane, work, image).await?;
            if uploaded_geometry {
                scene.mark_uploaded();
            }
        }
        Ok(())
    }
    run(canvas, frames)
        .await
        .map_err(|e| wasm_bindgen::JsValue::from_str(&format!("{e:?}")))
}
