//! Immediate-mode execution contract for backends without deferred command lists.
//!
//! The session reports an exact issued prefix on close or abandonment. It does
//! not add a scheduler, command cache, or hidden synchronization policy.
#![allow(missing_docs, dead_code, unused_variables, unused_imports)]
use super::*;
use std::{
    future::Future,
    marker::PhantomData,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

pub enum ImmediateClose<B: Backend> {
    Issued,
    Failed {
        error: Error,
        issued: bool,
    },
    Indeterminate {
        error: Error,
        uncertain: NativeUncertainty<B>,
    },
}
pub enum WaitInstallOutcome<B: Backend> {
    /// Whether installation itself issued a native queue/context operation.
    Ready { issued: bool },
    /// No native effect from this wait was issued; the slot may be retried.
    Rejected(Error),
    /// Native state may have changed. The backend retains its recovery state.
    Indeterminate {
        error: Error,
        uncertain: NativeUncertainty<B>,
    },
}
pub enum ImmediateCloseReason {
    Finished,
    Aborted,
}
pub enum CheckpointOutcome<B: Backend> {
    Covered(Vec<Checkpoint<B>>),
    Indeterminate(NativeUncertainty<B>),
}
pub enum ImmediateRunOutcome<B: Backend> {
    Issued(StepId),
    Failed {
        error: Error,
        prefix: IssuedPrefix<B>,
    },
    Indeterminate {
        error: Error,
        prefix: Vec<StepId>,
        uncertain: NativeUncertainty<B>,
    },
}
pub enum ImmediatePresentResult<B: Backend> {
    Run(ImmediateRunOutcome<B>),
    /// A clean rejection returns the same exclusive lease for retry/abandon.
    Rejected {
        error: Error,
        binding: PresentBinding<B>,
    },
}
pub enum ImmediateFinishOutcome<B: Backend> {
    Issued(IssuedPrefix<B>),
    Failed(ImmediateRunOutcome<B>),
    /// An earlier explicit close/failure already returned its true prefix.
    AlreadyReported,
}
/// Immediate-only SPI. Its command methods issue native calls synchronously.
/// No full CPU command list, replay stream, or owner-thread dispatcher exists.
/// # Safety
/// A failed or abandoned close must truthfully report whether work was issued.
/// Every emergency checkpoint must cover the complete reported prefix; if the
/// backend cannot prove coverage it must return Indeterminate.
pub unsafe trait ImmediateBackend: CommandBackend + CompletionBackend {
    /// Core calls this after validating that every wait names an earlier step
    /// of this plan and that every plan lane belongs to the device. Backends
    /// must still implement or reject cross-lane synchronization. A wait does
    /// not replace the resource visibility transitions supplied by the caller.
    /// Rejected means no native effect was issued. Any ambiguous native issue
    /// must return Indeterminate, never a plain error.
    fn install_waits(
        device: &Self::NativeDevice,
        lane: LaneId,
        waits: &[StepWait],
    ) -> WaitInstallOutcome<Self>;
    fn close_immediate(
        encoder: Self::NativeEncoder,
        reason: ImmediateCloseReason,
    ) -> ImmediateClose<Self>;
    /// Err is a clean native rejection with the still-live lease. Once native
    /// handoff is possible, return an issued/indeterminate close instead.
    fn present_immediate(
        device: &Self::NativeDevice,
        lane: LaneId,
        waits: &[StepWait],
        binding: PresentBinding<Self>,
    ) -> std::result::Result<ImmediateClose<Self>, (Error, PresentBinding<Self>)>;
    /// None creates no normal checkpoint. End covers every issued step, with
    /// multiple lane frontiers if needed. Steps covers the requested positions,
    /// possibly using later conservative proof. Failure must produce complete
    /// emergency coverage for the issued prefix or report Indeterminate.
    fn checkpoints(
        device: &Self::NativeDevice,
        plan: &SubmissionPlan<Self>,
        issued: &[StepId],
        request: &CompletionRequest,
    ) -> CheckpointOutcome<Self>;
    fn emergency_checkpoint(
        device: &Self::NativeDevice,
        plan: &SubmissionPlan<Self>,
        issued: &[StepId],
    ) -> CheckpointOutcome<Self>;
    /// An unclosed session is an executor error. Quarantine actual native GL
    /// issue effects, never silently assume completion or destroy resources.
    fn abandon_session(device: &Self::NativeDevice, plan: &SubmissionPlan<Self>, issued: &[StepId]);
}
pub struct ImmediateSession<B: ImmediateBackend> {
    pub(crate) device: Device<B>,
    plan: SubmissionPlan<B>,
    cursor: usize,
    issued_count: usize,
    history_count: usize,
    completion: CompletionRequest,
    failed: bool,
    terminal: Option<ImmediateRunOutcome<B>>,
    failure_effect: Option<StepEffect>,
    step_checkpoints: Vec<(StepId, Vec<Checkpoint<B>>)>,
    closed: bool,
}
impl<B: ImmediateBackend> Device<B> {
    /// # Safety
    /// Renderer retains every referenced object until true GPU completion and
    /// explicitly finishes or aborts the session after any native issue.
    pub unsafe fn immediate(
        &self,
        plan: SubmissionPlan<B>,
        completion: CompletionRequest,
    ) -> Result<ImmediateSession<B>> {
        // Validate the reusable topology once, before any native operation.
        // A wait orders execution; resource visibility still requires the
        // explicit transitions supplied by the renderer.
        plan.validate_for_execution(self)?;
        completion.validate(&plan)?;
        Ok(ImmediateSession {
            device: self.clone(),
            plan,
            cursor: 0,
            issued_count: 0,
            history_count: 0,
            completion,
            failed: false,
            terminal: None,
            failure_effect: None,
            step_checkpoints: Vec::new(),
            closed: false,
        })
    }
}
impl<B: ImmediateBackend> ImmediateSession<B> {
    fn requested_step_checkpoint(&mut self, step: StepId) -> Option<ImmediateRunOutcome<B>> {
        let CompletionRequest::Steps(requested) = &self.completion else {
            return None;
        };
        if !requested.contains(&step) {
            return None;
        }
        let request = CompletionRequest::Steps(Arc::from([step]));
        let issued = self.prefix(self.issued_count);
        match B::checkpoints(&self.device.native, &self.plan, &issued, &request) {
            CheckpointOutcome::Covered(checkpoints) => {
                self.step_checkpoints.push((step, checkpoints));
                None
            }
            CheckpointOutcome::Indeterminate(uncertain) => {
                // The step was already closed as issued. Completion observer
                // installation failed after that fact, so its effect is not a
                // clean failed slot and the true issued prefix must escape.
                self.failure_effect = Some(StepEffect::Unknown);
                self.failed = true;
                Some(ImmediateRunOutcome::Indeterminate {
                    error: Error::InvalidState,
                    prefix: self.take_issued_steps(),
                    uncertain,
                })
            }
        }
    }
    fn prefix(&self, count: usize) -> Vec<StepId> {
        (0..count).map(|index| self.plan.step(index)).collect()
    }
    fn take_issued_steps(&mut self) -> Vec<StepId> {
        let steps = self.prefix(self.issued_count);
        self.issued_count = 0;
        steps
    }
    fn slot(&self, kind: StepKind) -> Result<(StepId, &StepDesc)> {
        if self.failed {
            return Err(Error::InvalidState);
        }
        let desc = self
            .plan
            .steps
            .get(self.cursor)
            .ok_or(Error::InvalidState)?;
        if desc.kind != kind {
            return Err(Error::InvalidState);
        }
        Ok((self.plan.step(self.cursor), desc))
    }
    fn record(&mut self, step: StepId, outcome: ImmediateClose<B>) -> ImmediateRunOutcome<B> {
        match outcome {
            ImmediateClose::Issued => {
                self.issued_count += 1;
                self.history_count += 1;
                self.cursor += 1;
                self.requested_step_checkpoint(step)
                    .unwrap_or(ImmediateRunOutcome::Issued(step))
            }
            ImmediateClose::Failed { error, issued } => {
                self.failure_effect = Some(if issued {
                    StepEffect::Partial
                } else {
                    StepEffect::None
                });
                // A close failure can still have issued native commands. Keep
                // that prefix rather than returning an ordinary retryable
                // error, because later work may already observe its effects.
                if issued {
                    self.issued_count += 1;
                    self.history_count += 1;
                }
                self.fail(error)
            }
            ImmediateClose::Indeterminate { error, uncertain } => {
                self.failure_effect = Some(StepEffect::Unknown);
                self.failed = true;
                self.history_count += 1;
                ImmediateRunOutcome::Indeterminate {
                    error,
                    prefix: self.take_issued_steps(),
                    uncertain,
                }
            }
        }
    }
    pub(crate) fn failure_effect(&self) -> Option<StepEffect> {
        self.failure_effect
    }
    pub(crate) fn fail(&mut self, error: Error) -> ImmediateRunOutcome<B> {
        // The current operation failed before issuing unless its caller has
        // already recorded a more precise effect. Emergency checkpoint
        // uncertainty concerns the prefix, not the failing slot itself.
        self.failure_effect.get_or_insert(StepEffect::None);
        self.failed = true;
        let steps = self.take_issued_steps();
        // Once a session has issued any prefix, failure must carry completion
        // evidence for that exact prefix. Otherwise resource retirement could
        // race GPU work whose issuing status is no longer recoverable.
        match B::emergency_checkpoint(&self.device.native, &self.plan, &steps) {
            CheckpointOutcome::Covered(checkpoints) => ImmediateRunOutcome::Failed {
                error,
                prefix: IssuedPrefix { steps, checkpoints },
            },
            CheckpointOutcome::Indeterminate(uncertain) => ImmediateRunOutcome::Indeterminate {
                error,
                prefix: steps,
                uncertain,
            },
        }
    }
    pub fn begin_commands(
        &mut self,
    ) -> std::result::Result<ImmediateEncoder<'_, B>, ImmediateRunOutcome<B>> {
        let (step, desc) = match self.slot(StepKind::Commands) {
            Ok(v) => v,
            Err(e) => return Err(self.fail(e)),
        };
        let lane = desc.lane;
        let encoder = match CommandEncoder::new(&self.device, lane) {
            Ok(v) => v,
            Err(e) => return Err(self.fail(e)),
        };
        let wait_issued = match B::install_waits(&self.device.native, lane, &desc.waits) {
            WaitInstallOutcome::Ready { issued } => issued,
            WaitInstallOutcome::Rejected(error) => return Err(self.fail(error)),
            WaitInstallOutcome::Indeterminate { error, uncertain } => {
                self.failure_effect = Some(StepEffect::Unknown);
                self.failed = true;
                self.history_count += 1;
                return Err(ImmediateRunOutcome::Indeterminate {
                    error,
                    prefix: self.take_issued_steps(),
                    uncertain,
                });
            }
        };
        Ok(ImmediateEncoder {
            session: self,
            step,
            encoder: Some(encoder),
            wait_issued,
        })
    }
    pub fn present(&mut self, binding: PresentBinding<B>) -> ImmediatePresentResult<B> {
        let (step, desc) = match self.slot(StepKind::Present) {
            Ok(v) => v,
            Err(error) => return ImmediatePresentResult::Rejected { error, binding },
        };
        if binding.image.device != self.device.id {
            return ImmediatePresentResult::Rejected {
                error: Error::WrongDevice,
                binding,
            };
        }
        if binding.handed_off {
            return ImmediatePresentResult::Rejected {
                error: Error::InvalidState,
                binding,
            };
        }
        if !binding.image.is_current() {
            return ImmediatePresentResult::Rejected {
                error: Error::SurfaceOutdated,
                binding,
            };
        }
        match B::present_immediate(&self.device.native, desc.lane, &desc.waits, binding) {
            Ok(outcome) => ImmediatePresentResult::Run(self.record(step, outcome)),
            Err((error, binding)) => ImmediatePresentResult::Rejected { error, binding },
        }
    }
    pub fn finish(mut self) -> ImmediateFinishOutcome<B> {
        self.closed = true;
        if let Some(outcome) = self.terminal.take() {
            return ImmediateFinishOutcome::Failed(outcome);
        }
        if self.failed {
            return ImmediateFinishOutcome::AlreadyReported;
        }
        if self.cursor != self.plan.steps.len() {
            return ImmediateFinishOutcome::Failed(self.fail(Error::InvalidState));
        }
        let steps = self.take_issued_steps();
        // A completion request is useful only if it covers every issued step.
        // If the backend cannot prove coverage, expose indeterminacy rather
        // than fabricating a completion point from a later or unrelated event.
        let completion = match &self.completion {
            CompletionRequest::Steps(requested) => {
                let mut checkpoints = Vec::new();
                for step in requested.iter() {
                    let Some((_, checkpoints_for_step)) = self
                        .step_checkpoints
                        .iter()
                        .find(|(recorded, points)| recorded == step && !points.is_empty())
                    else {
                        // Preserve the established conservative Steps path for
                        // backends that returned no close-time point.
                        return match B::checkpoints(
                            &self.device.native,
                            &self.plan,
                            &steps,
                            &self.completion,
                        ) {
                            CheckpointOutcome::Covered(checkpoints) => {
                                ImmediateFinishOutcome::Issued(IssuedPrefix { steps, checkpoints })
                            }
                            CheckpointOutcome::Indeterminate(uncertain) => {
                                ImmediateFinishOutcome::Failed(ImmediateRunOutcome::Indeterminate {
                                    error: Error::InvalidState,
                                    prefix: steps,
                                    uncertain,
                                })
                            }
                        };
                    };
                    checkpoints.extend(checkpoints_for_step.iter().map(|checkpoint| Checkpoint {
                        point: checkpoint.point.clone(),
                        covers: CompletionCoverage {
                            lane_frontiers: checkpoint.covers.lane_frontiers.clone(),
                        },
                    }));
                }
                CheckpointOutcome::Covered(checkpoints)
            }
            _ => B::checkpoints(&self.device.native, &self.plan, &steps, &self.completion),
        };
        match completion {
            CheckpointOutcome::Covered(checkpoints) => {
                ImmediateFinishOutcome::Issued(IssuedPrefix { steps, checkpoints })
            }
            CheckpointOutcome::Indeterminate(uncertain) => {
                ImmediateFinishOutcome::Failed(ImmediateRunOutcome::Indeterminate {
                    error: Error::InvalidState,
                    prefix: steps,
                    uncertain,
                })
            }
        }
    }
    pub fn abort(mut self, error: Error) -> Option<ImmediateRunOutcome<B>> {
        self.closed = true;
        if let Some(outcome) = self.terminal.take() {
            return Some(outcome);
        }
        if self.failed {
            return None;
        }
        Some(self.fail(error))
    }
}

impl<B: ImmediateBackend> Drop for ImmediateSession<B> {
    fn drop(&mut self) {
        if !self.closed {
            // Drop cannot return a completion token. The backend quarantines
            // orphaned issued work until idle/loss; Renderer still owns every
            // resource lifetime and must explicitly recover/await.
            let history = self.prefix(self.history_count);
            B::abandon_session(&self.device.native, &self.plan, &history);
        }
    }
}

impl<B: ImmediateBackend + ImmediateSparseBackend> ImmediateSession<B> {
    pub(crate) fn sparse(&mut self, binds: &[SparseBind<B>]) -> ImmediateRunOutcome<B> {
        let (step, desc) = match self.slot(StepKind::SparseBind) {
            Ok(v) => v,
            Err(e) => return self.fail(e),
        };
        if binds.iter().any(|b| {
            b.resource.device != self.device.id
                || b.allocation
                    .as_ref()
                    .is_some_and(|a| a.device != self.device.id)
        }) {
            return self.fail(Error::WrongDevice);
        }
        let outcome = B::sparse_immediate(&self.device.native, desc.lane, &desc.waits, binds);
        self.record(step, outcome)
    }
}
pub struct ImmediateEncoder<'a, B: ImmediateBackend> {
    session: &'a mut ImmediateSession<B>,
    step: StepId,
    encoder: Option<CommandEncoder<B>>,
    wait_issued: bool,
}
impl<B: ImmediateBackend> ImmediateEncoder<'_, B> {
    pub fn commands(&mut self) -> &mut CommandEncoder<B> {
        self.encoder.as_mut().expect("active encoder")
    }
    pub fn close(mut self) -> ImmediateRunOutcome<B> {
        let encoder = self.encoder.take().expect("active encoder");
        let reason = if encoder.poisoned || encoder.in_pass {
            ImmediateCloseReason::Aborted
        } else {
            ImmediateCloseReason::Finished
        };
        let outcome = B::close_immediate(encoder.native.expect("native encoder"), reason);
        let outcome = match outcome {
            ImmediateClose::Failed { error, issued } => ImmediateClose::Failed {
                error,
                issued: issued || self.wait_issued,
            },
            other => other,
        };
        self.session.record(self.step, outcome)
    }
}
impl<B: ImmediateBackend> Drop for ImmediateEncoder<'_, B> {
    fn drop(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            let outcome = B::close_immediate(
                encoder.native.expect("native encoder"),
                ImmediateCloseReason::Aborted,
            );
            let outcome = match outcome {
                ImmediateClose::Failed { error, issued } => ImmediateClose::Failed {
                    error,
                    issued: issued || self.wait_issued,
                },
                other => other,
            };
            let terminal = self.session.record(self.step, outcome);
            // A dropped encoder is always an error, even if its native close
            // reports that its already issued commands were accepted.
            let terminal = match terminal {
                ImmediateRunOutcome::Issued(_) => {
                    self.session.failure_effect = Some(StepEffect::Partial);
                    self.session.fail(Error::InvalidState)
                }
                other => other,
            };
            self.session.terminal = Some(terminal);
        }
    }
}
