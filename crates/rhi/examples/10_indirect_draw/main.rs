//! Native indexed-indirect plant scene.
//!
//! Scene resources, KTX parsing, pipelines, and frame recording live in the
//! sibling modules, leaving this entry point responsible only for platform
//! startup and not for renderer scheduling or resource retirement.
#[path = "../utils/gltf.rs"]
mod gltf;
mod ktx;
mod pipeline;
mod scene;
#[path = "../utils/mod.rs"]
mod utils;
mod workload;

use fluxel_rhi::api::*;

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl"),
    all(target_arch = "wasm32", feature = "webgl2")
))]
mod immediate;

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
fn main_gl() {
    if let Err(error) = immediate::run() {
        eprintln!("10_indirect_draw: {error:?}");
        std::process::exit(1);
    }
}

/// Draws the indexed-indirect plant scene into a host-owned browser canvas.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_indirect(
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
        let mut demo = workload::Workload::<WebGpu>::new(&device, lane, format, extent).await?;
        for frame in 0..frames {
            let image = surface.acquire().await?;
            let work = demo.record(&device, &image, extent, frame as f32 / 60.0)?;
            utils::submit_and_present_async(&device, lane, work, image).await?;
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

/// Draws the same plant scene with ordinary indexed draws when WebGL2 lacks
/// indirect-draw commands. The device continues to report `indirect_draw=false`.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgl2_indirect_fallback(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    immediate::run_webgl2(canvas, frames)
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

#[cfg(any(windows, all(target_os = "macos", feature = "metal")))]
fn run<B: workload::IndirectBackend, H: utils::ExampleHost<B>>(host: H) -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut workload: Option<workload::Workload<B>> = None;
    let started = std::time::Instant::now();
    utils::run_example(
        host,
        "Indirect draw",
        config.clone(),
        move |device: &Device<B>, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if workload.is_none() {
                workload = Some(utils::block_on(workload::Workload::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?);
            }
            let work = workload.as_mut().ok_or(Error::InvalidState)?.record(
                device,
                &image,
                extent,
                started.elapsed().as_secs_f32(),
            )?;
            utils::submit_and_present(device, lane, work, image)
        },
    )
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    if let Err(error) = run::<fluxel_rhi::metal::Metal, _>(utils::metal_api::MetalHost) {
        eprintln!("10_indirect_draw: {error:?}");
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
        eprintln!("10_indirect_draw: {error:?}");
        std::process::exit(1);
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
    eprintln!("10_indirect_draw requires a supported native or browser backend");
}
