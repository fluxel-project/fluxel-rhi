//! WGL route for the sequential stereo visual fallback.
//!
//! The scene encoder calls execute on the WGL owner thread. This module only
//! drives Core's immediate session; it does not retain commands for replay.

use std::{
    cell::RefCell,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use fluxel_rhi::{api::*, gl::Gl};

use super::{per_eye::StereoFallback, utils};

static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);

fn outcome_error(outcome: ImmediateRunOutcome<Gl>) -> Error {
    match outcome {
        ImmediateRunOutcome::Issued(_) => Error::InvalidState,
        ImmediateRunOutcome::Failed { error, .. }
        | ImmediateRunOutcome::Indeterminate { error, .. } => error,
    }
}

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
    let scene = Rc::new(RefCell::new(None::<StereoFallback<Gl>>));
    let dropped = Rc::clone(&scene);
    utils::gl_api::run_wgl(
        "Sequential stereo fallback",
        config.clone(),
        move || *dropped.borrow_mut() = None,
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("GL graphics lane"))?;
            if scene
                .borrow()
                .as_ref()
                .is_none_or(|example| example.extent() != extent)
            {
                let fallback =
                    utils::block_on(StereoFallback::<Gl>::new(device, config.format, extent))
                        .map_err(|error| {
                            eprintln!("multiview fallback setup failed: {error:?}");
                            error
                        })?;
                *scene.borrow_mut() = Some(fallback);
            }
            let ordinal = NEXT_PLAN
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| Error::OutOfMemory)?;
            // SAFETY: this plan is device-local and has one command step and
            // one present step on the WGL context-owning thread.
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
            // SAFETY: every encoder verb and present call below executes while
            // the host callback owns the current WGL context.
            let mut session = unsafe { device.immediate(plan, CompletionRequest::End)? };
            let mut immediate = session.begin_commands().map_err(outcome_error)?;
            let mut scene = scene.borrow_mut();
            let scene = scene.as_mut().ok_or(Error::InvalidState)?;
            let uploaded = scene
                .encode(immediate.commands(), &image)
                .map_err(|error| {
                    eprintln!("multiview fallback encoding failed: {error:?}");
                    error
                })?;
            match immediate.close() {
                ImmediateRunOutcome::Issued(_) => {}
                outcome => return Err(outcome_error(outcome)),
            }
            if uploaded {
                scene.mark_uploaded();
            }
            match session.present(PresentBinding::new(image)) {
                ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_)) => {}
                ImmediatePresentResult::Run(outcome) => return Err(outcome_error(outcome)),
                ImmediatePresentResult::Rejected { error, .. } => return Err(error),
            }
            let point = utils::immediate_end_checkpoint(session.finish())?;
            utils::block_on(device.wait_completion(&point))
        },
    )
}
