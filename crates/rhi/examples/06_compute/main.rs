//! Three real storage-texture filters shown beside their KTX source image.
#[path = "scene.rs"]
mod compute;
#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
#[path = "immediate.rs"]
mod immediate;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "raster_fallback.rs"]
mod raster_fallback;
#[path = "../utils/mod.rs"]
mod utils;

use compute::Compute;
#[cfg(any(windows, all(target_os = "macos", feature = "metal")))]
use compute::ComputeBackend;
use fluxel_rhi::api::*;

/// Runs the same source/filter comparison with fragment shaders on WebGL2.
/// WebGL2 has no compute stage or storage textures, so its capability bits
/// remain false and this path performs an explicitly visible raster fallback.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgl2_compute_fallback(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    raster_fallback::run(canvas, frames)
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

/// Runs the storage-texture filters in a host-owned browser canvas.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_compute(
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
        let mut demo = Compute::<WebGpu>::new(&device, lane, format).await?;
        for _ in 0..frames {
            let image = surface.acquire().await?;
            let work = demo.record(&device, &image, extent)?;
            utils::submit_and_present_async(&device, lane, work, image).await?;
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

#[cfg(any(windows, all(target_os = "macos", feature = "metal")))]
fn run<B: ComputeBackend + DeferredBackend, H: utils::ExampleHost<B>>(host: H) -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut demo: Option<Compute<B>> = None;
    utils::run_example(
        host,
        "Storage texture filters",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if demo.is_none() {
                demo = Some(utils::block_on(Compute::new(device, lane, config.format))?);
            }
            let work = demo
                .as_mut()
                .ok_or(Error::InvalidState)?
                .record(device, &image, extent)?;
            utils::submit_and_present(device, lane, work, image)
        },
    )
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    if let Err(error) = run::<fluxel_rhi::metal::Metal, _>(utils::metal_api::MetalHost) {
        eprintln!("06_compute: {error:?}");
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
        if let Err(error) = immediate::run() {
            eprintln!("06_compute: {error:?}");
            std::process::exit(1);
        }
        return;
    }
    let result: Result<()> = match utils::windows_api::selected_backend() {
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
    if let Err(error) = result {
        eprintln!("06_compute: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(all(target_os = "linux", feature = "native-gles-egl"))]
fn main() {
    if let Err(error) = immediate::run() {
        eprintln!("06_compute: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(any(
    windows,
    all(target_os = "macos", feature = "metal"),
    all(target_os = "linux", feature = "native-gles-egl")
)))]
fn main() {
    eprintln!("06_compute requires a supported native or browser backend");
}
