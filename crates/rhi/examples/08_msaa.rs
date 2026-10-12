//! The SaschaWillems multisampled Voyager scene, lowered through RHI.
//!
//! The scene renders the original `voyager.gltf` mesh into a multisampled
//! color/depth pair and resolves the color attachment into the acquired frame.
mod common;
#[path = "common/gltf.rs"]
mod gltf;
mod msaa;

use fluxel_rhi::api::*;
use msaa::{Msaa, MsaaBackend};

pub(crate) const VOYAGER: &[u8] = include_bytes!("assets/models/voyager.gltf");
pub(crate) const VS: &[u8] = include_bytes!("assets/08_msaa/msaa.vs.dxil");
pub(crate) const PS: &[u8] = include_bytes!("assets/08_msaa/msaa.ps.dxil");
pub(crate) const VS_SPIRV: &[u8] = include_bytes!("assets/08_msaa/msaa.vs.spv");
pub(crate) const PS_SPIRV: &[u8] = include_bytes!("assets/08_msaa/msaa.ps.spv");
#[cfg(windows)]
fn run<B: MsaaBackend, H: common::ExampleHost<B>>(host: H) -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut workload: Option<Msaa<B>> = None;
    let start = std::time::Instant::now();
    common::run_example(
        host,
        "Multisample anti-aliasing",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if workload.is_none() {
                workload = Some(common::block_on(Msaa::new(
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
                start.elapsed().as_secs_f32(),
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
        eprintln!("08_msaa: {error:?}");
        std::process::exit(1)
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("08_msaa requires Windows");
}
