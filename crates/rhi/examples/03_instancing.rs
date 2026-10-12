//! Original glTF rock field, planet, stars, depth, and 8,192 instances via RHI.
//! The workload lives in `instancing/`; this entry owns the native host only.
mod common;
#[path = "common/gltf.rs"]
mod gltf;
mod instancing;

use fluxel_rhi::api::*;
use instancing::Instancing;

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
    let mut scene: Option<Instancing<Dx12>> = None;
    let started = std::time::Instant::now();
    let result = common::run_example(
        common::windows_api::Dx12Host,
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
                scene = Some(common::block_on(Instancing::new(
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
            common::submit_and_present(device, lane, work, image)
        },
    );
    if let Err(error) = result {
        eprintln!("03_instancing: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("03_instancing requires Windows and DX12");
}
