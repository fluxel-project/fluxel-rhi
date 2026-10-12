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
const PVS_SPIRV: &[u8] = include_bytes!("assets/07_offscreen/phong.vs.spv");
const PPS_SPIRV: &[u8] = include_bytes!("assets/07_offscreen/phong.ps.spv");
const MVS_SPIRV: &[u8] = include_bytes!("assets/07_offscreen/mirror.vs.spv");
const MPS_SPIRV: &[u8] = include_bytes!("assets/07_offscreen/mirror.ps.spv");
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
fn run<B: OffscreenBackend, H: common::ExampleHost<B>>(host: H) -> Result<()> {
    let c = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut x: Option<Offscreen<B>> = None;
    let start = std::time::Instant::now();
    common::run_example(
        host,
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
    )
}
#[cfg(windows)]
fn main() {
    let r = match common::windows_api::selected_backend() {
        Ok(common::windows_api::SelectedBackend::Dx12) => {
            run::<fluxel_rhi::dx12::Dx12, _>(common::windows_api::Dx12Host)
        }
        #[cfg(feature = "vulkan")]
        Ok(common::windows_api::SelectedBackend::Vulkan) => {
            run::<fluxel_rhi::vulkan::Vulkan, _>(common::windows_api::VulkanHost)
        }
        Err(error) => Err(error),
    };
    if let Err(e) = r {
        eprintln!("07_offscreen: {e:?}");
        std::process::exit(1)
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("07_offscreen requires Windows");
}
