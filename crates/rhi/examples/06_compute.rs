//! Three real storage-texture filters shown beside their KTX source image.
mod common;
#[path = "compute/mod.rs"]
mod compute;

use compute::Compute;
use fluxel_rhi::api::*;

#[cfg(windows)]
fn main() {
    use fluxel_rhi::dx12::Dx12;

    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut demo: Option<Compute<Dx12>> = None;
    let result = common::run_example(
        common::windows_api::Dx12Host,
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
                demo = Some(common::block_on(Compute::new(device, lane, config.format))?);
            }
            let work = demo
                .as_mut()
                .ok_or(Error::InvalidState)?
                .record(device, &image, extent)?;
            common::submit_and_present(device, lane, work, image)
        },
    );
    if let Err(error) = result {
        eprintln!("06_compute: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("06_compute requires Windows and DX12");
}
