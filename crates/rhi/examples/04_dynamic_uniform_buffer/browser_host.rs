//! Browser canvas exports for the dynamic-uniform cube grid.
//!
//! WebGPU records deferred work and WebGL2 issues the same workload through
//! its immediate session. Both routes consume `dynamic_uniform.wgsl`; the
//! backend selects the native lowering rather than the example maintaining
//! platform-specific shader sources.

#[cfg(feature = "webgpu")]
use super::{DynamicUniforms, utils};
#[cfg(feature = "webgpu")]
use fluxel_rhi::api::*;
use fluxel_rhi::backend::*;

/// Draws the dynamic uniform cube grid into a host-owned WebGPU canvas.
#[cfg(feature = "webgpu")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_dynamic_uniforms(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    use fluxel_rhi::webgpu::{WebGpu, WebGpuInstance};

    let run = async {
        if frames == 0 {
            return Err(Error::InvalidRange);
        }
        let device = WebGpuInstance::new()?.open_default().await?;
        let lane = device.lanes()[0].id;
        let extent = [canvas.width(), canvas.height()];
        let mut surface = WebGpu::surface_from_canvas(&device, canvas)?;
        let format = surface.capabilities()?.formats[0];
        let config = SurfaceConfig {
            extent,
            format,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 2,
        };
        surface.configure(&config).await?;
        let workload = DynamicUniforms::<WebGpu>::new(&device, lane, format, extent).await?;
        for frame in 0..frames {
            let image = surface.acquire().await?;
            let work = workload.record(&device, &image, extent, frame as f32 / 60.0)?;
            utils::submit_and_present_async(&device, lane, work, image).await?;
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

/// Draws the dynamic-uniform workload through a WebGL2 immediate session.
///
/// The GL backend owns the browser context and lowers the shared WGSL source
/// to the WebGL2 shader ABI. The public export accepts only the canvas.
#[cfg(feature = "webgl2")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgl2_dynamic_uniforms(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    super::immediate::run_webgl2(canvas, frames)
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}
