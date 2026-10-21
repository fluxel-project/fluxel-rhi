//! Immediate WGL adapter for the shared runtime-mipmap tunnel workload.
//!
//! Upload, per-level barriers, linear framebuffer blits, and the display pass
//! execute directly on the context owner thread in one immediate command step.

use std::cell::RefCell;
use std::rc::Rc;

use fluxel_rhi::api::*;
use fluxel_rhi::gl::Gl;

use super::{Tunnel, utils};

fn issue_frame(
    device: &Device<Gl>,
    lane: LaneId,
    tunnel: &mut Tunnel<Gl>,
    image: SurfaceImage<Gl>,
    extent: [u32; 2],
) -> Result<()> {
    let (commit, _) = utils::execute_frame(
        device,
        lane,
        image,
        CompletionRequest::None,
        |commands, image| tunnel.encode(commands, device, image, extent),
    )?;
    tunnel.commit_frame(commit);
    Ok(())
}

#[cfg(any(windows, all(target_os = "linux", feature = "native-gles-egl")))]
pub(super) fn run() -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    };
    let tunnel = Rc::new(RefCell::new(None));
    let dropped_tunnel = Rc::clone(&tunnel);
    utils::gl_api::run_native(
        "GPU generated mipmaps",
        config.clone(),
        move || *dropped_tunnel.borrow_mut() = None,
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("GL graphics lane"))?;
            if tunnel.borrow().is_none() {
                *tunnel.borrow_mut() = Some(utils::block_on(Tunnel::<Gl>::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?);
            }
            issue_frame(
                device,
                lane,
                tunnel.borrow_mut().as_mut().ok_or(Error::InvalidState)?,
                image,
                extent,
            )
        },
    )
}

#[cfg(not(any(windows, all(target_os = "linux", feature = "native-gles-egl"))))]
pub(super) fn run() -> Result<()> {
    Err(Error::Unsupported("09 GL WGL host"))
}
