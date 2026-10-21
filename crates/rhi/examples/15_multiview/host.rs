//! Native host setup for deferred multiview and Metal's sequential stereo fallback.
//!
//! This module owns only surface configuration and per-frame dispatch. The
//! scene and fallback encode their shared workloads in the sibling modules.

use super::*;

#[cfg(windows)]
pub(super) fn run<
    B: MultiviewBackend + ExecutionBackend<Mode = DeferredExecution>,
    H: utils::ExampleHost<B>,
>(
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
    let mut scene: Option<Multiview<B>> = None;
    utils::run_example(
        host,
        "Multiview rendering",
        config.clone(),
        move |device, _, image, extent| {
            require_multiview(device, 2)?;
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if scene
                .as_ref()
                .is_none_or(|example| example.extent() != extent)
            {
                scene = Some(utils::block_on(Multiview::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?);
            }
            let example = scene.as_mut().ok_or(Error::InvalidState)?;
            let (work, uploaded_geometry) = example.record(device, &image)?;
            utils::submit_and_present(device, lane, work, image)?;
            if uploaded_geometry {
                example.mark_uploaded();
            }
            Ok(())
        },
    )
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn run_metal_stereo_fallback() -> Result<()> {
    type B = fluxel_rhi::metal::Metal;
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut scene: Option<per_eye::StereoFallback<B>> = None;
    utils::run_example(
        utils::metal_api::MetalHost,
        "Sequential stereo fallback",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if scene
                .as_ref()
                .is_none_or(|fallback| fallback.extent() != extent)
            {
                scene = Some(utils::block_on(per_eye::StereoFallback::new(
                    device,
                    config.format,
                    extent,
                ))?);
            }
            let fallback = scene.as_mut().ok_or(Error::InvalidState)?;
            let mut encoder = device.encoder(lane)?;
            let uploaded = fallback.encode(&mut encoder, &image)?;
            let work = encoder.finish()?;
            utils::submit_and_present(device, lane, work, image)?;
            if uploaded {
                fallback.mark_uploaded();
            }
            Ok(())
        },
    )
}
