//! SaschaWillems offscreen mirror scene lowered through the RHI.
//!
//! This entry assembles the scene and records its two passes. The host owns
//! the window and presentation lifetime; the renderer remains responsible for
//! scheduling and resource retirement outside this standalone example.
mod common;
#[path = "common/gltf.rs"]
mod gltf;
use fluxel_rhi::api::*;
use std::sync::Arc;

const DRAGON: &[u8] = include_bytes!("assets/models/chinesedragon.gltf");
const PLANE: &[u8] = include_bytes!("assets/models/plane.gltf");
const PVS: &[u8] = include_bytes!("assets/07_offscreen/phong.vs.dxil");
const PPS: &[u8] = include_bytes!("assets/07_offscreen/phong.ps.dxil");
const MVS: &[u8] = include_bytes!("assets/07_offscreen/mirror.vs.dxil");
const MPS: &[u8] = include_bytes!("assets/07_offscreen/mirror.ps.dxil");
const PVS_SPIRV: &[u8] = include_bytes!("assets/07_offscreen/phong.vs.spv");
const PPS_SPIRV: &[u8] = include_bytes!("assets/07_offscreen/phong.ps.spv");
const MVS_SPIRV: &[u8] = include_bytes!("assets/07_offscreen/mirror.vs.spv");
const MPS_SPIRV: &[u8] = include_bytes!("assets/07_offscreen/mirror.ps.spv");
pub trait OffscreenBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + DeferredBackend
{
}
impl<T> OffscreenBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + DeferredBackend
{
}

mod offscreen {
    pub(crate) mod resources;
    mod workload;
    pub(crate) use workload::Offscreen;
}
use offscreen::Offscreen;

/// Draws the mirror scene into a host-owned browser canvas.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_offscreen(
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
        let mut demo = Offscreen::<WebGpu>::new(&device, lane, format, extent).await?;
        for frame in 0..frames {
            let image = surface.acquire().await?;
            let work = demo.record(&device, &image, extent, frame as f32 / 60.0)?;
            common::submit_and_present_async(&device, lane, work, image).await?;
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

#[cfg(windows)]
fn run<B: OffscreenBackend, H: common::ExampleHost<B>>(host: H) -> Result<()> {
    let c = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut x: Option<Offscreen<B>> = None;
    let start = std::time::Instant::now();
    common::run_example(
        host,
        "Offscreen mirror rendering",
        c.clone(),
        move |d, _, i, e| {
            let l = d
                .lanes()
                .iter()
                .find(|x| x.class == LaneClass::Graphics)
                .map(|x| x.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if x.is_none() {
                x = Some(common::block_on(Offscreen::new(d, l, c.format, e))?)
            }
            let w = x.as_mut().ok_or(Error::InvalidState)?.record(
                d,
                &i,
                e,
                start.elapsed().as_secs_f32(),
            )?;
            common::submit_and_present(d, l, w, i)
        },
    )
}
#[cfg(windows)]
fn main() {
    let r = match common::windows_api::selected_backend() {
        Ok(common::windows_api::SelectedBackend::Dx12) => {
            run::<fluxel_rhi::dx12::Dx12, _>(common::windows_api::Dx12Host)
        }
        #[cfg(feature = "vulkan")]
        Ok(common::windows_api::SelectedBackend::Vulkan) => {
            run::<fluxel_rhi::vulkan::Vulkan, _>(common::windows_api::VulkanHost)
        }
        Err(error) => Err(error),
    };
    if let Err(e) = r {
        eprintln!("07_offscreen: {e:?}");
        std::process::exit(1)
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("07_offscreen requires Windows");
}
