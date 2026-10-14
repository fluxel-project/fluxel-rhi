//! Immediate WGL adapter for the shared offscreen mirror workload.
//!
//! The workload issues its offscreen pass, explicit texture transitions, and
//! surface pass through one owner-thread command encoder. GL presentation is
//! the next immediate step; this adapter does not create deferred work.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use fluxel_rhi::api::*;
use fluxel_rhi::gl::Gl;

use super::{Offscreen, utils};

static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);

fn immediate_error(outcome: ImmediateRunOutcome<Gl>) -> Error {
    match outcome {
        ImmediateRunOutcome::Issued(_) => Error::InvalidState,
        ImmediateRunOutcome::Failed { error, .. }
        | ImmediateRunOutcome::Indeterminate { error, .. } => error,
    }
}

fn issue_frame(
    device: &Device<Gl>,
    lane: LaneId,
    workload: &mut Offscreen<Gl>,
    image: SurfaceImage<Gl>,
    extent: [u32; 2],
    time: f32,
) -> Result<()> {
    let ordinal = NEXT_PLAN
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::OutOfMemory)?;
    // SAFETY: commands and presentation target the one adopted GL lane in order.
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
    // SAFETY: the acquired image remains leased through the immediate present step.
    let mut session = unsafe { device.immediate(plan, CompletionRequest::None)? };
    let mut commands = session.begin_commands().map_err(immediate_error)?;
    workload.encode(commands.commands(), device, &image, extent, time)?;
    match commands.close() {
        ImmediateRunOutcome::Issued(_) => workload.commit_frame(),
        outcome => return Err(immediate_error(outcome)),
    }
    match session.present(PresentBinding::new(image)) {
        ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_)) => {}
        ImmediatePresentResult::Run(outcome) => return Err(immediate_error(outcome)),
        ImmediatePresentResult::Rejected { error, .. } => return Err(error),
    }
    match session.finish() {
        ImmediateFinishOutcome::Issued(_) => Ok(()),
        ImmediateFinishOutcome::Failed(outcome) => Err(immediate_error(outcome)),
        ImmediateFinishOutcome::AlreadyReported => Err(Error::InvalidState),
    }
}

pub(super) fn run_gl() -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        // WGL exposes the tested RGBA default framebuffer format; the scene's
        // visible color semantics are identical to the BGRA native routes.
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
    utils::gl_api::run_wgl(
        "Offscreen mirror rendering",
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
                *workload.borrow_mut() = Some(utils::block_on(Offscreen::<Gl>::new(
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
