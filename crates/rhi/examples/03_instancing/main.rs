//! Instanced glTF rock field with planet, stars, depth, and 8,192 instances.
//!
//! The workload lives in `instancing/`; this entry owns platform startup and
//! does not own GPU scheduling, resource retirement, or window lifetime.
#[path = "../utils/gltf.rs"]
mod gltf;
#[path = "scene.rs"]
mod instancing;
#[path = "../utils/mod.rs"]
mod utils;

use fluxel_rhi::api::*;
use instancing::Instancing;

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl"),
    all(target_arch = "wasm32", feature = "webgl2")
))]
#[path = "immediate.rs"]
mod immediate;

/// Draws the instanced rock field into a host-owned browser canvas.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_instancing(
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
        let mut scene = Instancing::<WebGpu>::new(&device, lane, format, extent).await?;
        for frame in 0..frames {
            let image = surface.acquire().await?;
            let work = scene.record(&device, &image, extent, frame as f32 / 60.0)?;
            utils::submit_and_present_async(&device, lane, work, image).await?;
            scene.mark_submitted();
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

/// Draws the instanced rock field through the immediate WebGL2 backend.
///
/// The scene consumes the same WGSL assets as WebGPU. The GL backend lowers
/// those entries to the browser's GLSL ES route on the context owner thread.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgl2_instancing(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    immediate::run_webgl2(canvas, frames)
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

#[cfg(any(windows, all(target_os = "macos", feature = "metal")))]
fn run<B: instancing::InstancingBackend + DeferredBackend, H: utils::ExampleHost<B>>(
    host: H,
) -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut scene: Option<Instancing<B>> = None;
    let started = std::time::Instant::now();
    utils::run_example(
        host,
        "Instanced rock field",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if scene.is_none() {
                scene = Some(utils::block_on(Instancing::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?);
            }
            let work = scene.as_mut().ok_or(Error::InvalidState)?.record(
                device,
                &image,
                extent,
                started.elapsed().as_secs_f32(),
            )?;
            utils::submit_and_present(device, lane, work, image)?;
            scene.as_mut().ok_or(Error::InvalidState)?.mark_submitted();
            Ok(())
        },
    )
}

#[cfg(windows)]
fn main() {
    #[cfg(feature = "native-gl-wgl")]
    if std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .any(|args| args[0] == "--backend" && args[1] == "gl")
    {
        if let Err(error) = immediate::run_native() {
            eprintln!("03_instancing: {error:?}");
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
        eprintln!("03_instancing: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    if let Err(error) = run::<fluxel_rhi::metal::Metal, _>(utils::metal_api::MetalHost) {
        eprintln!("03_instancing: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(all(target_os = "linux", feature = "native-gles-egl"))]
fn main() {
    if let Err(error) = immediate::run_native() {
        eprintln!("03_instancing: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(any(
    windows,
    all(target_os = "macos", feature = "metal"),
    all(target_os = "linux", feature = "native-gles-egl")
)))]
fn main() {
    eprintln!("03_instancing requires Windows and DX12");
}
