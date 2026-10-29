//! Deferred and immediate submission SPI witnesses for Core contract tests.
use super::*;

impl DeferredBackend for NativeShared {
    type NativeRecordedWork = ();
    fn finish_encoder(_: (), _: LaneId) -> Result<()> {
        Ok(())
    }
    // SAFETY: this mock never issues native work and returns the owning
    // packet unchanged; there is no accepted prefix to retain or retire.
    unsafe fn submit(_: &Arc<()>, packet: SubmitPacket<Self>) -> SubmitOutcome<Self> {
        SubmitOutcome::Rejected {
            error: Error::Unsupported("mock"),
            work: RetryableSubmit::Fresh(packet),
        }
    }
    // SAFETY: this mock also rejects a tail without touching its accepted
    // frontier, so the caller retains its original ownership and proof.
    unsafe fn resume(_: &Arc<()>, tail: DeferredTail<Self>) -> SubmitOutcome<Self> {
        SubmitOutcome::Rejected {
            error: Error::Unsupported("mock"),
            work: RetryableSubmit::Tail(tail),
        }
    }
}
// SAFETY: this fixture reports every issued prefix deterministically.
unsafe impl ImmediateBackend for GlLocal {
    fn install_waits(_: &Rc<()>, _: LaneId, _: &[StepWait]) -> WaitInstallOutcome<Self> {
        if GL_WAIT_REJECT.with(|v| v.replace(false)) {
            WaitInstallOutcome::Rejected(Error::Backend("wait rejected".into()))
        } else {
            WaitInstallOutcome::Ready { issued: false }
        }
    }
    fn close_immediate(_: Rc<()>, reason: ImmediateCloseReason) -> ImmediateClose<Self> {
        if matches!(reason, ImmediateCloseReason::Aborted)
            && GL_CLOSE_FAILURE.with(|mode| mode.get()) == 3
        {
            GL_CLOSE_FAILURE.with(|mode| mode.set(0));
            return ImmediateClose::Issued;
        }
        if matches!(reason, ImmediateCloseReason::Finished) {
            match GL_CLOSE_FAILURE.with(|mode| mode.replace(0)) {
                1 => {
                    return ImmediateClose::Failed {
                        error: Error::Backend("partial close".into()),
                        issued: true,
                    };
                }
                2 => {
                    return ImmediateClose::Indeterminate {
                        error: Error::Backend("uncertain close".into()),
                        uncertain: NativeUncertainty {
                            device: GL_DEVICE.with(|id| id.get().unwrap()),
                            native: (),
                        },
                    };
                }
                _ => {}
            }
        }
        match reason {
            ImmediateCloseReason::Finished => ImmediateClose::Issued,
            ImmediateCloseReason::Aborted => ImmediateClose::Failed {
                error: Error::InvalidState,
                issued: false,
            },
        }
    }
    fn present_immediate(
        _: &Rc<()>,
        _: LaneId,
        _: &[StepWait],
        _: PresentBinding<Self>,
    ) -> std::result::Result<ImmediateClose<Self>, (Error, PresentBinding<Self>)> {
        Ok(ImmediateClose::Issued)
    }
    fn checkpoints(
        _: &Rc<()>,
        plan: &SubmissionPlan<Self>,
        issued: &[StepId],
        request: &CompletionRequest,
    ) -> CheckpointOutcome<Self> {
        GL_CHECKPOINTS.with(|calls| calls.set(calls.get() + 1));
        if GL_CHECKPOINT_UNKNOWN.with(|v| v.replace(false)) {
            CheckpointOutcome::Indeterminate(NativeUncertainty {
                device: GL_DEVICE.with(|id| id.get().unwrap()),
                native: (),
            })
        } else if GL_EMPTY_STEP_CHECKPOINTS.with(|v| v.get())
            && matches!(request, CompletionRequest::Steps(steps) if steps.len() == 1)
        {
            CheckpointOutcome::Covered(Vec::new())
        } else if GL_EMPTY_STEP_CHECKPOINTS.with(|v| v.get())
            && matches!(request, CompletionRequest::Steps(steps) if steps.len() > 1)
        {
            let Some(&frontier) = issued.last() else {
                return CheckpointOutcome::Covered(Vec::new());
            };
            CheckpointOutcome::Covered(vec![Checkpoint {
                // SAFETY: this mock point belongs to the fixture device.
                point: unsafe {
                    CompletionPoint::from_native(
                        GL_DEVICE.with(|id| id.get().unwrap()),
                        Rc::new(()),
                    )
                },
                covers: CompletionCoverage {
                    lane_frontiers: vec![(plan.steps()[frontier.index as usize].lane, frontier)],
                },
            }])
        } else {
            CheckpointOutcome::Covered(Vec::new())
        }
    }
    fn emergency_checkpoint(
        _: &Rc<()>,
        _: &SubmissionPlan<Self>,
        _: &[StepId],
    ) -> CheckpointOutcome<Self> {
        GL_CHECKPOINTS.with(|calls| calls.set(calls.get() + 1));
        if GL_EMERGENCY_UNKNOWN.with(|v| v.replace(false)) {
            CheckpointOutcome::Indeterminate(NativeUncertainty {
                device: GL_DEVICE.with(|id| id.get().unwrap()),
                native: (),
            })
        } else {
            CheckpointOutcome::Covered(Vec::new())
        }
    }
    fn abandon_session(_: &Rc<()>, _: &SubmissionPlan<Self>, issued: &[StepId]) {
        ABANDONED_STEPS.with(|count| count.set(issued.len()));
        GL_ABANDONED.with(|count| count.set(count.get() + 1));
    }
}
