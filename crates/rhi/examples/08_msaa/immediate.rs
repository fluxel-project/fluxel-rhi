//! Immediate WGL adapter for the shared multisampled Voyager workload.
//!
//! GL raster verbs execute on the context owner thread. The adapter only
//! creates the one-lane immediate plan and hands the acquired image to present.

use std::cell::RefCell;
use std::rc::Rc;

use fluxel_rhi::api::*;
use fluxel_rhi::gl::Gl;

use super::{Msaa, utils};

fn issue_frame(
    device: &Device<Gl>,
    lane: LaneId,
    workload: &mut Msaa<Gl>,
    image: SurfaceImage<Gl>,
    extent: [u32; 2],
    time: f32,
) -> Result<()> {
    let (commit, _) = utils::execute_frame(
        device,
        lane,
        image,
        CompletionRequest::None,
        |commands, image| workload.encode(commands, device, image, extent, time),
    )?;
    workload.commit_frame(commit);
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
    let start = std::time::Instant::now();
    let workload = Rc::new(RefCell::new(None));
    let dropped_workload = Rc::clone(&workload);
    utils::gl_api::run_native(
        "Multisample anti-aliasing",
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
                *workload.borrow_mut() = Some(utils::block_on(Msaa::<Gl>::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?);
            }
            issue_frame(
                device,
                lane,
                workload.borrow_mut().as_mut().ok_or(Error::InvalidState)?,
                image,
                extent,
                start.elapsed().as_secs_f32(),
            )
        },
    )
}

#[cfg(not(any(windows, all(target_os = "linux", feature = "native-gles-egl"))))]
pub(super) fn run() -> Result<()> {
    Err(Error::Unsupported("08 GL WGL host"))
}
