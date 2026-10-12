//! Runtime RGBA8 mip-chain generation through native linear blits.
mod common;
#[path = "common/gltf.rs"]
mod gltf;
mod texture_mipmap;

use fluxel_rhi::api::*;
use texture_mipmap::Tunnel;

pub(crate) const MODEL: &[u8] = include_bytes!("assets/models/tunnel_cylinder.gltf");
pub(crate) const IMAGE: &[u8] = include_bytes!("assets/textures/metalplate_nomips_rgba.ktx");
pub(crate) const VS: &[u8] = include_bytes!("assets/09_texture_mipmap/tunnel.vs.dxil");
pub(crate) const PS: &[u8] = include_bytes!("assets/09_texture_mipmap/tunnel.ps.dxil");
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
    let mut demo: Option<Tunnel<Dx12>> = None;
    let result = common::run_example(
        common::windows_api::Dx12Host,
        "GPU generated mipmaps",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if demo.is_none() {
                demo = Some(common::block_on(Tunnel::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?);
            }
            let work = demo
                .as_mut()
                .ok_or(Error::InvalidState)?
                .record(device, &image, extent)?;
            common::submit_and_present(device, lane, work, image)
        },
    );
    if let Err(error) = result {
        eprintln!("09_texture_mipmap: {error:?}");
        std::process::exit(1);
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("09_texture_mipmap requires Windows and DX12");
}
