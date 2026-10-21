//! GL immediate execution path for the compute-filter example.
//!
//! The shared execution interface runs compute and display on the owner thread.
//! The GL backend issues each command directly while the workload owns resources.

use std::cell::RefCell;
use std::rc::Rc;

use fluxel_rhi::api::*;
use fluxel_rhi::gl::Gl;

use super::{compute::Compute, utils};

fn issue_frame(
    device: &Device<Gl>,
    lane: LaneId,
    workload: &mut Compute<Gl>,
    image: SurfaceImage<Gl>,
    extent: [u32; 2],
) -> Result<()> {
    let (initialized, _) = utils::execute_frame(
        device,
        lane,
        image,
        CompletionRequest::None,
        |commands, image| workload.encode(commands, device, image, extent),
    )?;
    workload.commit_frame(initialized);
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
    let workload = Rc::new(RefCell::new(None));
    let dropped_workload = Rc::clone(&workload);
    utils::gl_api::run_native(
        "Storage texture filters",
        config.clone(),
        move || *dropped_workload.borrow_mut() = None,
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("GL graphics lane"))?;
            if workload.borrow().is_none() {
                *workload.borrow_mut() = Some(utils::block_on(Compute::<Gl>::new(
                    device,
                    lane,
                    config.format,
                ))?);
            }
            let mut workload = workload.borrow_mut();
            issue_frame(
                device,
                lane,
                workload.as_mut().ok_or(Error::InvalidState)?,
                image,
                extent,
            )
        },
    )
}

#[cfg(not(any(windows, all(target_os = "linux", feature = "native-gles-egl"))))]
pub(super) fn run() -> Result<()> {
    Err(Error::Unsupported("06 GL compute host"))
}
