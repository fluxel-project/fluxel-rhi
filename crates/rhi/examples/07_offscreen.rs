//! The SaschaWillems offscreen mirror scene, lowered through RHI.
mod common;
#[path = "common/gltf.rs"]
mod gltf;
use fluxel_rhi::api::*;
use std::sync::Arc;

const DRAGON: &[u8] = include_bytes!("assets/models/chinesedragon.gltf");
const PLANE: &[u8] = include_bytes!("assets/models/plane.gltf");
const PVS: &[u8] = include_bytes!("assets/07_offscreen/phong.vs.dxil");
const PPS: &[u8] = include_bytes!("assets/07_offscreen/phong.ps.dxil");
const MVS: &[u8] = include_bytes!("assets/07_offscreen/mirror.vs.dxil");
const MPS: &[u8] = include_bytes!("assets/07_offscreen/mirror.ps.dxil");
pub trait OffscreenBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + DeferredBackend
{
}
impl<T> OffscreenBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + DeferredBackend
{
}

mod offscreen {
    pub(crate) mod resources;
    mod workload;
    pub(crate) use workload::Offscreen;
}
use offscreen::Offscreen;

#[cfg(windows)]
fn main() {
    use fluxel_rhi::dx12::Dx12;
    let c = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut x: Option<Offscreen<Dx12>> = None;
    let start = std::time::Instant::now();
    let r = common::run_example(
        common::windows_api::Dx12Host,
        "Offscreen mirror rendering",
        c.clone(),
        move |d, _, i, e| {
            let l = d
                .lanes()
                .iter()
                .find(|x| x.class == LaneClass::Graphics)
                .map(|x| x.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if x.is_none() {
                x = Some(common::block_on(Offscreen::new(d, l, c.format, e))?)
            }
            let w = x.as_mut().ok_or(Error::InvalidState)?.record(
                d,
                &i,
                e,
                start.elapsed().as_secs_f32(),
            )?;
            common::submit_and_present(d, l, w, i)
        },
    );
    if let Err(e) = r {
        eprintln!("07_offscreen: {e:?}");
        std::process::exit(1)
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("07_offscreen requires Windows and DX12");
}
