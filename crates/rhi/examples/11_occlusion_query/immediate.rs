//! Immediate WGL adapter for the shared occlusion-query workload.
//!
//! The query result is resolved by GL's blocking result query before the
//! frame's host completion fence. This preserves the Core readback contract
//! without turning immediate commands into a deferred command list.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use fluxel_rhi::api::*;
use fluxel_rhi::gl::Gl;

use super::{OcclusionWorkloadBackend, Workload, utils};

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
    workload: &mut Workload<Gl>,
    image: SurfaceImage<Gl>,
    extent: [u32; 2],
) -> Result<()> {
    let ordinal = NEXT_PLAN
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::OutOfMemory)?;
    // SAFETY: both plan steps target this one context-owned graphics lane and
    // execute in the order in which the adapter calls them.
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
    // SAFETY: every command below is issued directly on the current WGL
    // context; the acquired surface image stays leased through Present.
    let mut session = unsafe { device.immediate(plan, CompletionRequest::End)? };
    let mut commands = session.begin_commands().map_err(immediate_error)?;
    let ticket = workload.encode(device, commands.commands(), &image, extent)?;
    match commands.close() {
        ImmediateRunOutcome::Issued(_) => {}
        outcome => return Err(immediate_error(outcome)),
    }
    match session.present(PresentBinding::new(image)) {
        ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_)) => {}
        ImmediatePresentResult::Run(outcome) => return Err(immediate_error(outcome)),
        ImmediatePresentResult::Rejected { error, .. } => return Err(error),
    }
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => prefix
            .checkpoints
            .into_iter()
            .next()
            .map(|checkpoint| checkpoint.point)
            .ok_or(Error::InvalidState)?,
        ImmediateFinishOutcome::Failed(outcome) => return Err(immediate_error(outcome)),
        ImmediateFinishOutcome::AlreadyReported => return Err(Error::InvalidState),
    };
    workload.consume(device, ticket, &point)
}

#[cfg(windows)]
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
    utils::gl_api::run_wgl(
        "Occlusion queries",
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
                *workload.borrow_mut() = Some(utils::block_on(Workload::<Gl>::new(
                    device,
                    lane,
                    config.format,
                ))?);
            }
            let mut workload = workload.borrow_mut();
            let workload = workload.as_mut().ok_or(Error::InvalidState)?;
            issue_frame(device, lane, workload, image, extent)
        },
    )
}

#[cfg(not(windows))]
pub(super) fn run() -> Result<()> {
    Err(Error::Unsupported("11 GL WGL host"))
}

const _: fn() = || {
    fn assert_workload<B: OcclusionWorkloadBackend>() {}
    assert_workload::<Gl>();
};
