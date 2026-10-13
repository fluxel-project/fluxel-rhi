//! Native host loop for the multithreaded-recording example.
//!
//! The host owns surface configuration and presentation; scene recording stays
//! in the parent module.
use super::*;

#[cfg(windows)]
fn run_host<B: SecondaryExampleBackend, H: common::ExampleHost<B>>(host: H) -> Result<()>
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
    common::run_example(
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
                scene = Some(common::block_on(Multithreaded::new(d, lane, cfg.format))?)
            }
            let (work, uploaded_geometry) = scene
                .as_mut()
                .ok_or(Error::InvalidState)?
                .record(d, &image, extent)?;
            common::submit_and_present(d, lane, work, image)?;
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
    let result = match common::windows_api::selected_backend() {
        Ok(common::windows_api::SelectedBackend::Dx12) => {
            run_host::<fluxel_rhi::dx12::Dx12, _>(common::windows_api::Dx12Host)
        }
        #[cfg(feature = "vulkan")]
        Ok(common::windows_api::SelectedBackend::Vulkan) => {
            run_host::<fluxel_rhi::vulkan::Vulkan, _>(common::windows_api::VulkanHost)
        }
        Err(error) => Err(error),
    };
    if let Err(e) = result {
        eprintln!("13_multithreaded_recording: {e:?}");
        std::process::exit(1)
    }
}
/// Reports the unavailable native target on unsupported hosts.
#[cfg(not(windows))]
pub(super) fn run() {
    eprintln!(
        "13_multithreaded_recording requires Windows and a backend with secondary raster commands"
    )
}
