//! Immediate WGL adapter for the shared runtime-mipmap tunnel workload.
//!
//! Upload, per-level barriers, linear framebuffer blits, and the display pass
//! execute directly on the context owner thread in one immediate command step.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use fluxel_rhi::api::*;
use fluxel_rhi::gl::Gl;

use super::{Tunnel, utils};

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
    tunnel: &mut Tunnel<Gl>,
    image: SurfaceImage<Gl>,
    extent: [u32; 2],
) -> Result<()> {
    let ordinal = NEXT_PLAN
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::OutOfMemory)?;
    // SAFETY: command and present steps use the same adopted GL lane, in order.
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
    // SAFETY: this owner-thread session directly issues the command step and
    // retains the acquired image until the ordered present step consumes it.
    let mut session = unsafe { device.immediate(plan, CompletionRequest::None)? };
    let mut commands = session.begin_commands().map_err(immediate_error)?;
    let commit = tunnel.encode(commands.commands(), device, &image, extent)?;
    match commands.close() {
        ImmediateRunOutcome::Issued(_) => tunnel.commit_frame(commit),
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
    let tunnel = Rc::new(RefCell::new(None));
    let dropped_tunnel = Rc::clone(&tunnel);
    utils::gl_api::run_wgl(
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

#[cfg(not(windows))]
pub(super) fn run() -> Result<()> {
    Err(Error::Unsupported("09 GL WGL host"))
}
