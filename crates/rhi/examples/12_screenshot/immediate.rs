//! Immediate WGL path for the shared screenshot workload.
//!
//! Texture capture uses the GL backend's direct readback route at the command
//! call site. The immediate session then supplies Core's completion evidence
//! before the ticket is exposed as a file.

use std::{
    cell::RefCell,
    rc::Rc,
    sync::Arc,
    sync::atomic::{AtomicU64, Ordering},
};

use fluxel_rhi::{api::*, gl::Gl};

use super::{Screenshot, ScreenshotWorkloadBackend, utils};

static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);

fn run_error(outcome: ImmediateRunOutcome<Gl>) -> Error {
    match outcome {
        ImmediateRunOutcome::Issued(_) => Error::InvalidState,
        ImmediateRunOutcome::Failed { error, .. }
        | ImmediateRunOutcome::Indeterminate { error, .. } => error,
    }
}

fn issue_frame(
    device: &Device<Gl>,
    lane: LaneId,
    screenshot: &mut Screenshot<Gl>,
    image: SurfaceImage<Gl>,
    capture: bool,
) -> Result<bool> {
    let ordinal = NEXT_PLAN
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::OutOfMemory)?;
    // SAFETY: both plan steps use this context-owned lane and are issued in
    // order by the WGL owner thread.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal,
            },
            Arc::from([
                StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane,
                    kind: StepKind::Present,
                    waits: vec![],
                },
            ]),
        )
    };
    // SAFETY: all commands execute on the current WGL context; the acquired
    // surface lease is retained until the following Present step.
    let mut session = unsafe { device.immediate(plan, CompletionRequest::End)? };
    let mut commands = session.begin_commands().map_err(run_error)?;
    let ticket = screenshot.encode(commands.commands(), &image, capture)?;
    match commands.close() {
        ImmediateRunOutcome::Issued(_) => {}
        outcome => return Err(run_error(outcome)),
    }
    match session.present(PresentBinding::new(image)) {
        ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_)) => {}
        ImmediatePresentResult::Run(outcome) => return Err(run_error(outcome)),
        ImmediatePresentResult::Rejected { error, .. } => return Err(error),
    }
    let point = utils::immediate_end_checkpoint(session.finish())?;
    if let Some(ticket) = ticket {
        utils::block_on(screenshot.save_ppm_after(device, ticket, &point, "screenshot.ppm"))?;
        Ok(true)
    } else {
        utils::block_on(device.wait_completion(&point))?;
        Ok(false)
    }
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
    let capture_requested = std::env::var_os("FLUXEL_RHI_SCREENSHOT").is_some();
    let screenshot = Rc::new(RefCell::new(None::<(Screenshot<Gl>, [u32; 2])>));
    let dropped = Rc::clone(&screenshot);
    let captured = Rc::new(RefCell::new(false));
    let captured_frame = Rc::clone(&captured);
    utils::gl_api::run_native(
        "Screenshot readback",
        config.clone(),
        move || *dropped.borrow_mut() = None,
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("GL graphics lane"))?;
            if screenshot
                .borrow()
                .as_ref()
                .is_none_or(|(_, previous_extent)| *previous_extent != extent)
            {
                *screenshot.borrow_mut() = Some((
                    utils::block_on(Screenshot::<Gl>::new_with_surface_format(
                        device,
                        lane,
                        extent,
                        TextureFormat::Rgba8Unorm,
                    ))?,
                    extent,
                ));
            }
            let mut slot = screenshot.borrow_mut();
            let (workload, _) = slot.as_mut().ok_or(Error::InvalidState)?;
            let capture = capture_requested && !*captured_frame.borrow();
            let did_capture = issue_frame(device, lane, workload, image, capture)?;
            if did_capture {
                *captured_frame.borrow_mut() = true;
            }
            Ok(())
        },
    )
}

#[cfg(not(any(windows, all(target_os = "linux", feature = "native-gles-egl"))))]
pub(super) fn run() -> Result<()> {
    Err(Error::Unsupported("12 GL WGL host"))
}

const _: fn() = || {
    fn assert_workload<B: ScreenshotWorkloadBackend>() {}
    assert_workload::<Gl>();
};
