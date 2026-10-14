//! Native GL immediate execution for the sequential fallback.

use super::*;

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
pub fn run_gl_multithreaded_recording() -> Result<()> {
    use std::{
        cell::RefCell,
        rc::Rc,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
    };

    use fluxel_rhi::gl::Gl;

    static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);
    fn outcome_error(outcome: ImmediateRunOutcome<Gl>) -> Error {
        match outcome {
            ImmediateRunOutcome::Issued(_) => Error::InvalidState,
            ImmediateRunOutcome::Failed { error, .. }
            | ImmediateRunOutcome::Indeterminate { error, .. } => error,
        }
    }

    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    };
    let scene = Rc::new(RefCell::new(None::<Sequential<Gl>>));
    let dropped = Rc::clone(&scene);
    utils::gl_api::run_native(
        "Sequential UFO recording fallback",
        config.clone(),
        move || *dropped.borrow_mut() = None,
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("GL graphics lane"))?;
            if scene.borrow().is_none() {
                let next = utils::block_on(Sequential::<Gl>::new(device, config.format))?;
                *scene.borrow_mut() = Some(next);
            }
            let ordinal = NEXT_PLAN
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| Error::OutOfMemory)?;
            // SAFETY: this single-threaded WGL callback owns the context and
            // issues both plan steps consecutively on its graphics lane.
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
            // SAFETY: immediate encoding and presentation happen on the
            // current WGL context before the callback returns.
            let mut session = unsafe { device.immediate(plan, CompletionRequest::End)? };
            let mut commands = session.begin_commands().map_err(outcome_error)?;
            let uploaded_geometry = scene
                .borrow_mut()
                .as_mut()
                .ok_or(Error::InvalidState)?
                .encode(device, &image, commands.commands(), extent)?;
            match commands.close() {
                ImmediateRunOutcome::Issued(_) => {}
                outcome => return Err(outcome_error(outcome)),
            }
            if uploaded_geometry {
                scene
                    .borrow_mut()
                    .as_mut()
                    .ok_or(Error::InvalidState)?
                    .mark_uploaded();
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
