//! Native host loop for the multithreaded-recording example.
//!
//! The host owns surface configuration and presentation; scene recording stays
//! in the parent module.
use super::*;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn run_metal() -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut scene: Option<sequential::Sequential<fluxel_rhi::metal::Metal>> = None;
    utils::run_example(
        utils::metal_api::MetalHost,
        "Sequential UFO command fallback",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if scene.is_none() {
                scene = Some(utils::block_on(sequential::Sequential::new(
                    device,
                    config.format,
                ))?);
            }
            let scene = scene.as_mut().ok_or(Error::InvalidState)?;
            let mut encoder = device.encoder(lane)?;
            let uploaded = scene.encode(device, &image, &mut encoder, extent)?;
            let work = encoder.finish()?;
            utils::submit_and_present(device, lane, work, image)?;
            if uploaded {
                scene.mark_uploaded();
            }
            Ok(())
        },
    )
}

#[cfg(windows)]
fn run_host<
    B: SecondaryExampleBackend + ExecutionBackend<Mode = DeferredExecution>,
    H: utils::ExampleHost<B>,
>(
    host: H,
) -> Result<()>
where
    B::NativeDevice: Send + Sync,
    B::NativeRasterPipeline: Send + Sync,
    B::NativeBuffer: Send + Sync,
    B::NativeBindGroup: Send + Sync,
    B::NativeSecondaryWork: Send,
{
    let cfg = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut scene: Option<Multithreaded<B>> = None;
    utils::run_example(
        host,
        "Multi threaded UFO command buffers",
        cfg.clone(),
        move |d, _, image, extent| {
            let lane = d
                .lanes()
                .iter()
                .find(|x| x.class == LaneClass::Graphics)
                .map(|x| x.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if scene.is_none() {
                scene = Some(utils::block_on(Multithreaded::new(d, lane, cfg.format))?)
            }
            let (work, uploaded_geometry) = scene
                .as_mut()
                .ok_or(Error::InvalidState)?
                .record(d, &image, extent)?;
            utils::submit_and_present(d, lane, work, image)?;
            if uploaded_geometry {
                scene.as_mut().ok_or(Error::InvalidState)?.mark_uploaded();
            }
            Ok(())
        },
    )
}

/// Runs the native Cargo example target.
#[cfg(windows)]
pub(super) fn run() {
    let result: Result<()> = match utils::windows_api::selected_backend() {
        #[cfg(feature = "dx12")]
        Ok(utils::windows_api::SelectedBackend::Dx12) => {
            run_host::<fluxel_rhi::dx12::Dx12, _>(utils::windows_api::Dx12Host)
        }
        #[cfg(feature = "vulkan")]
        Ok(utils::windows_api::SelectedBackend::Vulkan) => {
            run_host::<fluxel_rhi::vulkan::Vulkan, _>(utils::windows_api::VulkanHost)
        }
        Err(error) => Err(error),
    };
    if let Err(e) = result {
        eprintln!("13_multithreaded_recording: {e:?}");
        std::process::exit(1)
    }
}
/// Reports the unavailable native target on unsupported hosts.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn run() {
    if let Err(error) = run_metal() {
        eprintln!("13_multithreaded_recording: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(any(windows, all(target_os = "macos", feature = "metal"))))]
pub(super) fn run() {
    eprintln!(
        "13_multithreaded_recording requires Windows and a backend with secondary raster commands"
    )
}
