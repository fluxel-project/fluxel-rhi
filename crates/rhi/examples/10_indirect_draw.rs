// The native indexed-indirect plant scene.
// Scene resources, KTX parsing, pipelines, and frame recording live in the
// sibling modules so this entry point only owns platform startup.
mod common;
#[path = "common/gltf.rs"]
mod gltf;
#[path = "indirect_draw/ktx.rs"]
mod ktx;
#[path = "indirect_draw/pipeline.rs"]
mod pipeline;
#[path = "indirect_draw/scene.rs"]
mod scene;
#[path = "indirect_draw/workload.rs"]
mod workload;

use fluxel_rhi::api::*;

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
            common::submit_and_present_async(&device, lane, work, image).await?;
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

#[cfg(windows)]
fn run<B: workload::IndirectBackend, H: common::ExampleHost<B>>(host: H) -> Result<()> {
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
    common::run_example(
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
                workload = Some(common::block_on(workload::Workload::new(
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
            common::submit_and_present(device, lane, work, image)
        },
    )
}

#[cfg(windows)]
fn main() {
    let result = match common::windows_api::selected_backend() {
        Ok(common::windows_api::SelectedBackend::Dx12) => {
            run::<fluxel_rhi::dx12::Dx12, _>(common::windows_api::Dx12Host)
        }
        #[cfg(feature = "vulkan")]
        Ok(common::windows_api::SelectedBackend::Vulkan) => {
            run::<fluxel_rhi::vulkan::Vulkan, _>(common::windows_api::VulkanHost)
        }
        Err(error) => Err(error),
    };
    if let Err(error) = result {
        eprintln!("10_indirect_draw: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("10_indirect_draw requires Windows");
}
