//! SaschaWillems offscreen mirror scene lowered through the RHI.
//!
//! This entry assembles the scene and records its two passes. The host owns
//! the window and presentation lifetime; the renderer remains responsible for
//! scheduling and resource retirement outside this standalone example.
#[path = "../utils/gltf.rs"]
mod gltf;
#[path = "../utils/mod.rs"]
mod utils;
use fluxel_rhi::api::*;
use std::sync::Arc;

const DRAGON: &[u8] = include_bytes!("../assets/models/chinesedragon.gltf");
const PLANE: &[u8] = include_bytes!("../assets/models/plane.gltf");
const PVS: &[u8] = include_bytes!("../assets/07_offscreen/phong.vs.dxil");
const PPS: &[u8] = include_bytes!("../assets/07_offscreen/phong.ps.dxil");
const MVS: &[u8] = include_bytes!("../assets/07_offscreen/mirror.vs.dxil");
const MPS: &[u8] = include_bytes!("../assets/07_offscreen/mirror.ps.dxil");
const PVS_SPIRV: &[u8] = include_bytes!("../assets/07_offscreen/phong.vs.spv");
const PPS_SPIRV: &[u8] = include_bytes!("../assets/07_offscreen/phong.ps.spv");
const MVS_SPIRV: &[u8] = include_bytes!("../assets/07_offscreen/mirror.vs.spv");
const MPS_SPIRV: &[u8] = include_bytes!("../assets/07_offscreen/mirror.ps.spv");
pub trait OffscreenWorkloadBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend
{
}
impl<T> OffscreenWorkloadBackend for T where
    T: Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend
{
}
pub trait OffscreenBackend: OffscreenWorkloadBackend + DeferredBackend {}
impl<T> OffscreenBackend for T where T: OffscreenWorkloadBackend + DeferredBackend {}

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl"),
    all(target_arch = "wasm32", feature = "webgl2")
))]
#[path = "immediate.rs"]
mod immediate;

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
pub(crate) use immediate::run_gl;

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
pub(crate) fn main_gl() {
    if let Err(error) = run_gl() {
        eprintln!("07_offscreen: {error:?}");
        std::process::exit(1);
    }
}

/// Draws the shared two-pass mirror scene through WebGL2 immediate commands.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgl2_offscreen(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    immediate::run_webgl2(canvas, frames)
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

#[path = "resources.rs"]
mod resources;
#[path = "workload.rs"]
mod workload;
pub use workload::Offscreen;

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
            utils::submit_and_present_async(&device, lane, work, image).await?;
            demo.commit_frame();
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

#[cfg(any(windows, all(target_os = "macos", feature = "metal")))]
fn run<B: OffscreenBackend, H: utils::ExampleHost<B>>(host: H) -> Result<()> {
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
    utils::run_example(
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
                x = Some(utils::block_on(Offscreen::new(d, l, c.format, e))?)
            }
            let w = x.as_mut().ok_or(Error::InvalidState)?.record(
                d,
                &i,
                e,
                start.elapsed().as_secs_f32(),
            )?;
            utils::submit_and_present(d, l, w, i)?;
            x.as_mut().ok_or(Error::InvalidState)?.commit_frame();
            Ok(())
        },
    )
}
#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    if let Err(error) = run::<fluxel_rhi::metal::Metal, _>(utils::metal_api::MetalHost) {
        eprintln!("07_offscreen: {error:?}");
        std::process::exit(1);
    }
}
#[cfg(windows)]
fn main() {
    #[cfg(feature = "native-gl-wgl")]
    if std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .any(|args| args[0] == "--backend" && args[1] == "gl")
    {
        main_gl();
        return;
    }
    let r: Result<()> = match utils::windows_api::selected_backend() {
        #[cfg(feature = "dx12")]
        Ok(utils::windows_api::SelectedBackend::Dx12) => {
            run::<fluxel_rhi::dx12::Dx12, _>(utils::windows_api::Dx12Host)
        }
        #[cfg(feature = "vulkan")]
        Ok(utils::windows_api::SelectedBackend::Vulkan) => {
            run::<fluxel_rhi::vulkan::Vulkan, _>(utils::windows_api::VulkanHost)
        }
        Err(error) => Err(error),
    };
    if let Err(e) = r {
        eprintln!("07_offscreen: {e:?}");
        std::process::exit(1)
    }
}
#[cfg(all(target_os = "linux", feature = "native-gles-egl"))]
fn main() {
    main_gl();
}

#[cfg(not(any(
    windows,
    all(target_os = "macos", feature = "metal"),
    all(target_os = "linux", feature = "native-gles-egl")
)))]
fn main() {
    eprintln!("07_offscreen requires a supported native or browser backend");
}
