//! Immediate-session close, presentation, and checkpoint facts for GL.
//!
//! GL executes every command verb when it is called. This module therefore
//! creates no queue and no replay list: it only turns an issued prefix into a
//! truthful close result and, when requested, a fence-based host observation.

use fluxel_rhi_core::api::{
    self, Checkpoint, CheckpointOutcome, CompletionCoverage, CompletionPoint, CompletionRequest,
    Error, ImmediateClose, ImmediateCloseReason, NativeUncertainty, PresentBinding, StepId,
    StepWait, SubmissionPlan, WaitInstallOutcome,
};
use fluxel_rhi_core::backend::ImmediateBackend;

use super::completion::GlCompletion;
use super::surface;
use super::types::GlEncoder;
use super::{Gl, GlDevice};

// SAFETY: GL encoders issue their verbs synchronously on the adopted context's
// owner thread. Each method below reports the actual issued prefix and creates
// only host completion evidence; it neither stores commands nor schedules them.
unsafe impl ImmediateBackend for Gl {
    fn install_waits(
        _: &GlDevice,
        lane: api::LaneId,
        waits: &[StepWait],
    ) -> WaitInstallOutcome<Self> {
        // Core validates the full plan and its lanes before entering this SPI.
        // This backend exposes exactly one lane, so all valid predecessors
        // precede this call on the same native context. Ordering does not replace
        // the renderer's explicit resource visibility transitions.
        if waits
            .iter()
            .any(|wait| wait.previous.plan.device != lane.device)
        {
            WaitInstallOutcome::Rejected(Error::WrongLane)
        } else {
            WaitInstallOutcome::Ready { issued: false }
        }
    }

    fn close_immediate(
        mut encoder: GlEncoder,
        reason: ImmediateCloseReason,
    ) -> ImmediateClose<Self> {
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        if encoder.statistics_query_active || encoder.indeterminate {
            let result = encoder
                .state
                .with_native("close GL pipeline statistics", |provider| {
                    provider.abort_pipeline_statistics()
                });
            if result.is_err() {
                encoder.indeterminate = true;
            }
            if encoder.statistics_query_active && !encoder.indeterminate {
                return ImmediateClose::Failed {
                    error: Error::InvalidState,
                    issued: encoder.issued,
                };
            }
        }
        if encoder.indeterminate {
            // SAFETY: at least one direct native call returned after entering
            // the provider boundary, so its acceptance cannot be reclassified
            // as a clean rejection.
            return ImmediateClose::Indeterminate {
                error: Error::Backend("indeterminate immediate GL command".into()),
                uncertain: unsafe { NativeUncertainty::from_native(encoder.device, ()) },
            };
        }
        match reason {
            ImmediateCloseReason::Finished => {
                #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
                if let Some(browser) = encoder.browser.take() {
                    let issued = encoder.issued;
                    if let Err(error) = browser.close_scope() {
                        return if issued {
                            ImmediateClose::Indeterminate {
                                error,
                                // SAFETY: the immediate prefix was already
                                // issued, and failure to close its owner scope
                                // leaves native acceptance uncertain.
                                uncertain: unsafe {
                                    NativeUncertainty::from_native(encoder.device, ())
                                },
                            }
                        } else {
                            ImmediateClose::Failed {
                                error,
                                issued: false,
                            }
                        };
                    }
                }
                ImmediateClose::Issued
            }
            ImmediateCloseReason::Aborted => ImmediateClose::Failed {
                error: Error::InvalidState,
                issued: encoder.issued,
            },
        }
    }

    fn present_immediate(
        _: &GlDevice,
        lane: api::LaneId,
        waits: &[StepWait],
        mut binding: PresentBinding<Self>,
    ) -> std::result::Result<ImmediateClose<Self>, (Error, PresentBinding<Self>)> {
        if waits
            .iter()
            .any(|wait| wait.previous.plan.device != lane.device)
        {
            return Err((Error::WrongLane, binding));
        }
        // SAFETY: Core validated the exclusive, current image before entering
        // this SPI. `try_handoff` is the irreversible boundary: only after it
        // succeeds may a swap failure be reported as indeterminate.
        if let Err(error) = unsafe { binding.try_handoff() } {
            return Err((error, binding));
        }
        // SAFETY: the handoff above keeps the native image in `binding` until
        // this call returns; presentation only reads its private lease state.
        let image = unsafe { binding.image().native_ref() };
        match surface::present_handed_off(image) {
            Ok(()) => Ok(ImmediateClose::Issued),
            Err(error) => Ok(ImmediateClose::Indeterminate {
                error,
                // SAFETY: swap was entered after irrevocable handoff and may
                // have reached the driver even when it returned an error.
                uncertain: unsafe { NativeUncertainty::from_native(lane.device, ()) },
            }),
        }
    }

    fn checkpoints(
        device: &GlDevice,
        plan: &SubmissionPlan<Self>,
        issued: &[StepId],
        request: &CompletionRequest,
    ) -> CheckpointOutcome<Self> {
        if issued.is_empty() || matches!(request, CompletionRequest::None) {
            CheckpointOutcome::Covered(Vec::new())
        } else {
            checkpoint_after_issued(device, plan, issued)
        }
    }

    fn emergency_checkpoint(
        device: &GlDevice,
        plan: &SubmissionPlan<Self>,
        issued: &[StepId],
    ) -> CheckpointOutcome<Self> {
        if issued.is_empty() {
            CheckpointOutcome::Covered(Vec::new())
        } else {
            checkpoint_after_issued(device, plan, issued)
        }
    }

    fn abandon_session(device: &GlDevice, _: &SubmissionPlan<Self>, issued: &[StepId]) {
        if issued.is_empty() {
            return;
        }
        // An unclosed session cannot hand a point back to Core. Capture a
        // terminal fence anyway and immediately drop it: GlCompletion moves
        // the unobserved lease into its owner-thread orphan set, where later
        // completion observations retire it. That keeps a real GPU boundary
        // without retaining a CPU command stream or calling GL from a worker.
        // If fence creation itself fails, the native context remains the
        // conservative quarantine owner until loss/teardown.
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            let _ = GlCompletion::after_issued(device.0.clone());
        }
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            let _ = device.0.owner.completion_after_current_commands();
        }
    }
}

fn checkpoint_after_issued(
    device: &GlDevice,
    plan: &SubmissionPlan<Gl>,
    issued: &[StepId],
) -> CheckpointOutcome<Gl> {
    let Some(frontier) = issued.last().copied() else {
        return CheckpointOutcome::Covered(Vec::new());
    };
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    let completion = GlCompletion::after_issued(device.0.clone());
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    let completion = device
        .0
        .owner
        .completion_after_current_commands()
        .map(|value| GlCompletion::from_browser(device.0.clone(), value));
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    let completion: std::result::Result<GlCompletion, Error> =
        Err(Error::Unsupported("GL platform completion"));
    match completion {
        Ok(completion) => CheckpointOutcome::Covered(vec![Checkpoint {
            // SAFETY: after_issued created this fence on the exact adopted
            // device after every direct command in `issued` was called.
            point: unsafe { CompletionPoint::from_native(plan.id().device, completion) },
            covers: CompletionCoverage {
                lane_frontiers: vec![(plan.steps()[frontier.index as usize].lane, frontier)],
            },
        }]),
        Err(_) => CheckpointOutcome::Indeterminate(
            // SAFETY: creating/flushing a fence failed after the caller had
            // already issued the prefix, so its native completion boundary is
            // inherently uncertain and must not be reported as a clean error.
            unsafe { NativeUncertainty::from_native(plan.id().device, ()) },
        ),
    }
}
