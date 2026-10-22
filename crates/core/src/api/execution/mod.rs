//! Optional statically dispatched execution facade over the native execution
//! protocols. It stores deferred native recordings, never CPU command streams.
//! Immediate commands execute inside the supplied closure on the context thread.
//!
//! The caller remains responsible for the compiled plan, synchronization and
//! keeping referenced resources alive until the backend's completion evidence.
use super::*;

#[cfg(test)]
mod tests;

/// Opts a backend into the common execution interface by selecting its native
/// execution model. Existing backend and encoder traits remain unchanged.
pub trait ExecutionBackend: CommandBackend + CompletionBackend {
    /// The backend's execution protocol.
    type Mode: ExecutionMode<Self>;
}

/// Borrowed facts from a model-specific result, without consuming recovery
/// payloads or interpreting an issue prefix as GPU completion.
pub trait ExecutionOutcome<B: Backend> {
    /// Portable execution observations.
    fn facts(&self) -> ExecutionFacts<'_, B>;
    /// The operation's failure, if one was reported.
    fn error(&self) -> Option<&Error>;
}

/// Static implementation of the common execution operations.
pub trait ExecutionMode<B: ExecutionBackend>: Sized {
    /// State owned by an open execution.
    type State;
    /// The original, model-specific owning result.
    type Outcome: ExecutionOutcome<B>;
    /// Starts execution under the caller's resource lifetime obligations.
    ///
    /// # Safety
    /// The plan must be realizable and all referenced resources retained until
    /// completion, including partial or uncertain native effects.
    unsafe fn begin(
        device: &Device<B>,
        plan: SubmissionPlan<B>,
        completion: CompletionRequest,
    ) -> Result<Self::State>;
    /// Encodes the next commands step. Immediate encoders call native APIs now.
    fn commands<F: FnOnce(&mut CommandEncoder<B>) -> Result<()>>(
        state: &mut Self::State,
        step: StepId,
        encode: F,
    ) -> Result<()>;
    /// Binds or performs the next presentation step.
    fn present(
        state: &mut Self::State,
        step: StepId,
        binding: PresentBinding<B>,
    ) -> std::result::Result<(), PresentError<B>>;
    /// Ends execution and returns all native execution facts and recovery state.
    ///
    /// # Safety
    /// The caller must satisfy the obligations of `begin` through completion.
    unsafe fn finish(state: Self::State) -> Self::Outcome;
}

/// A presentation error; a clean rejection returns the unconsumed lease.
pub struct PresentError<B: Backend> {
    /// The failure reported to the caller.
    pub error: Error,
    /// Present only if native handoff did not consume the lease.
    pub binding: Option<PresentBinding<B>>,
}

/// One execution of an already compiled plan.
#[must_use = "finish the execution and preserve its completion evidence"]
pub struct Execution<B: ExecutionBackend> {
    state: <B::Mode as ExecutionMode<B>>::State,
}

impl<B: ExecutionBackend> Device<B> {
    /// Opens an execution without changing the underlying backend's timing.
    ///
    /// # Safety
    /// The caller supplies a realizable plan and retains referenced resources
    /// through actual completion, including on partial or uncertain failure.
    pub unsafe fn begin_execution(
        &self,
        plan: SubmissionPlan<B>,
        completion: CompletionRequest,
    ) -> Result<Execution<B>> {
        // SAFETY: the caller assumes the plan and resource obligations above.
        let state = unsafe { B::Mode::begin(self, plan, completion)? };
        Ok(Execution { state })
    }
}

impl<B: ExecutionBackend> Execution<B> {
    /// Executes/records the next commands step using the shared command API.
    /// On error, call `finish` to recover native execution facts.
    pub fn commands<F: FnOnce(&mut CommandEncoder<B>) -> Result<()>>(
        &mut self,
        step: StepId,
        encode: F,
    ) -> Result<()> {
        B::Mode::commands(&mut self.state, step, encode)
    }
    /// Performs/binds the next presentation step. Immediate success means the
    /// native handoff has occurred; deferred success only binds the lease and
    /// the handoff outcome is known after `finish` submits the packet.
    /// After any error, finish the execution to recover native effects.
    pub fn present(
        &mut self,
        step: StepId,
        binding: PresentBinding<B>,
    ) -> std::result::Result<(), PresentError<B>> {
        B::Mode::present(&mut self.state, step, binding)
    }
    /// Returns the underlying protocol's owning result. No host wait is added.
    ///
    /// # Safety
    /// All resource and synchronization obligations of `begin_execution`
    /// continue until true GPU completion, even after an error.
    pub unsafe fn finish(self) -> <B::Mode as ExecutionMode<B>>::Outcome {
        // SAFETY: the caller maintains the execution obligations.
        unsafe { B::Mode::finish(self.state) }
    }
}

/// A clean rejection of pre-recorded native work, returned unchanged so it can
/// be rebound or explicitly abandoned by its owner.
pub struct RecordedCommandsError<B: DeferredBackend> {
    /// Why the current plan slot or recordings were rejected.
    pub error: Error,
    /// The original one-shot native recordings, in their original order.
    pub work: Vec<RecordedWork<B>>,
}

impl<B: DeferredBackend + ExecutionBackend<Mode = DeferredExecution>> Execution<B> {
    /// Binds a sparse mapping step into the deferred submission packet.
    pub fn sparse(&mut self, step: StepId, binds: Vec<SparseBind<B>>) -> Result<()> {
        let state = &mut self.state;
        let index = state.steps.len();
        if state.error.is_some()
            || state
                .plan
                .steps
                .get(index)
                .is_none_or(|desc| desc.kind != StepKind::SparseBind)
            || step != state.plan.step(index)
        {
            return Err(Error::InvalidState);
        }
        if binds.iter().any(|bind| {
            bind.resource.device != state.device.id
                || bind
                    .allocation
                    .as_ref()
                    .is_some_and(|a| a.device != state.device.id)
        }) {
            return Err(Error::WrongDevice);
        }
        state.steps.push(BoundStep {
            step,
            payload: StepPayload::SparseBind(binds),
        });
        Ok(())
    }
    /// Binds already-recorded native work to the next commands step.
    ///
    /// This accepts recordings produced ahead of time or in parallel without
    /// translating or replaying their commands. On rejection all recordings are
    /// returned unchanged. The caller retains referenced resources through the
    /// actual completion of any accepted or uncertain submission.
    pub fn recorded_commands(
        &mut self,
        step: StepId,
        work: Vec<RecordedWork<B>>,
    ) -> std::result::Result<(), RecordedCommandsError<B>> {
        let state = &mut self.state;
        let index = state.steps.len();
        let error = if state.error.is_some() {
            Some(Error::InvalidState)
        } else if let Some(desc) = state.plan.steps.get(index) {
            if step != state.plan.step(index) || desc.kind != StepKind::Commands {
                Some(Error::InvalidState)
            } else if work
                .iter()
                .any(|recording| recording.device() != state.device.id())
            {
                Some(Error::WrongDevice)
            } else if work.iter().any(|recording| recording.lane() != desc.lane) {
                Some(Error::WrongLane)
            } else {
                None
            }
        } else {
            Some(Error::InvalidState)
        };
        if let Some(error) = error {
            return Err(RecordedCommandsError { error, work });
        }
        state.steps.push(BoundStep {
            step,
            payload: StepPayload::Commands(work),
        });
        Ok(())
    }
}

/// Deferred-native implementation of the facade.
pub struct DeferredExecution;
/// Immediate-native implementation of the facade.
pub struct ImmediateExecution;

/// Deferred state stores native recordings and bound leases, not commands.
pub struct DeferredExecutionState<B: DeferredBackend> {
    device: Device<B>,
    plan: SubmissionPlan<B>,
    completion: CompletionRequest,
    steps: Vec<BoundStep<B>>,
    error: Option<Error>,
}

/// An execution that failed before native submission retains all bound work.
pub enum DeferredExecutionOutcome<B: DeferredBackend> {
    /// Native submission's unmodified owning outcome.
    Submitted(SubmitOutcome<B>),
    /// No submission occurred. Recorded work and presentation leases remain owned.
    Unsubmitted {
        error: Error,
        plan: SubmissionPlan<B>,
        steps: Vec<BoundStep<B>>,
        completion: CompletionRequest,
    },
}

impl<B: DeferredBackend> ExecutionOutcome<B> for DeferredExecutionOutcome<B> {
    fn facts(&self) -> ExecutionFacts<'_, B> {
        match self {
            Self::Submitted(outcome) => outcome.facts(),
            Self::Unsubmitted { .. } => ExecutionFacts {
                status: ExecutionStatus::Rejected,
                failed_step: None,
                accepted: Some(&[]),
                affected: Some(&[]),
                checkpoints: None,
                presents: None,
            },
        }
    }
    fn error(&self) -> Option<&Error> {
        match self {
            Self::Unsubmitted { error, .. } => Some(error),
            Self::Submitted(
                SubmitOutcome::Rejected { error, .. }
                | SubmitOutcome::PartiallyAccepted { error, .. }
                | SubmitOutcome::Indeterminate { error, .. },
            ) => Some(error),
            Self::Submitted(SubmitOutcome::Accepted(_)) => None,
        }
    }
}

impl<B: ExecutionBackend + DeferredBackend> ExecutionMode<B> for DeferredExecution {
    type State = DeferredExecutionState<B>;
    type Outcome = DeferredExecutionOutcome<B>;
    unsafe fn begin(
        device: &Device<B>,
        plan: SubmissionPlan<B>,
        completion: CompletionRequest,
    ) -> Result<Self::State> {
        plan.validate_for_execution(device)?;
        if !device.caps.sparse
            && plan
                .steps
                .iter()
                .any(|step| step.kind == StepKind::SparseBind)
        {
            return Err(Error::Unsupported(
                "sparse steps are not supported by this execution facade",
            ));
        }
        completion.validate(&plan)?;
        Ok(DeferredExecutionState {
            device: device.clone(),
            plan,
            completion,
            steps: Vec::new(),
            error: None,
        })
    }
    fn commands<F: FnOnce(&mut CommandEncoder<B>) -> Result<()>>(
        state: &mut Self::State,
        step: StepId,
        encode: F,
    ) -> Result<()> {
        if state.error.is_some() {
            return Err(Error::InvalidState);
        }
        let index = state.steps.len();
        let desc = state.plan.steps.get(index).ok_or(Error::InvalidState)?;
        if step != state.plan.step(index) || desc.kind != StepKind::Commands {
            return Err(Error::InvalidState);
        }
        let result: Result<()> = (|| {
            let mut encoder = state.device.encoder(desc.lane)?;
            encode(&mut encoder)?;
            let work = encoder.finish()?;
            state.steps.push(BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            });
            Ok(())
        })();
        if let Err(error) = &result {
            state.error = Some(error.clone());
        }
        result
    }
    fn present(
        state: &mut Self::State,
        step: StepId,
        binding: PresentBinding<B>,
    ) -> std::result::Result<(), PresentError<B>> {
        let index = state.steps.len();
        if state.error.is_some()
            || state
                .plan
                .steps
                .get(index)
                .is_none_or(|d| d.kind != StepKind::Present)
            || step != state.plan.step(index)
        {
            return Err(PresentError {
                error: Error::InvalidState,
                binding: Some(binding),
            });
        }
        state.steps.push(BoundStep {
            step,
            payload: StepPayload::Present(binding),
        });
        Ok(())
    }
    unsafe fn finish(state: Self::State) -> Self::Outcome {
        if state.error.is_some() || state.steps.len() != state.plan.steps.len() {
            return DeferredExecutionOutcome::Unsubmitted {
                error: state.error.unwrap_or(Error::InvalidState),
                plan: state.plan,
                steps: state.steps,
                completion: state.completion,
            };
        }
        let packet = SubmitPacket {
            plan: state.plan,
            steps: state.steps,
            completion: state.completion,
        };
        // SAFETY: caller retains resources and provides valid synchronization.
        DeferredExecutionOutcome::Submitted(unsafe { state.device.queue().submit(packet) })
    }
}

/// Immediate state owns a live session and any failure returned by it.
pub struct ImmediateExecutionState<B: ImmediateBackend> {
    session: ImmediateSession<B>,
    plan: SubmissionPlan<B>,
    cursor: usize,
    failure: Option<ImmediateRunOutcome<B>>,
    error: Option<Error>,
    failed_slot: Option<StepId>,
    rejected_present: Option<(StepId, Error)>,
}

/// The session result and, if different, the original command encoding error.
pub struct ImmediateExecutionOutcome<B: ImmediateBackend> {
    /// Number of steps fully issued before a possible failing step.
    pub completed_steps: usize,
    /// The unsuccessful slot and its known native effect, if identifiable.
    pub failed_step: Option<FailedStep>,
    /// Original encoding error or unretried clean presentation rejection.
    pub error: Option<Error>,
    /// Successful presentation handoffs. On failure this is only the known
    /// completed prefix; absence is not proof about an uncertain current step.
    pub presents: Vec<PresentReceipt>,
    /// Unmodified session finish/failure with native completion or uncertainty.
    pub outcome: ImmediateFinishOutcome<B>,
}

impl<B: ImmediateBackend> ExecutionOutcome<B> for ImmediateExecutionOutcome<B> {
    fn facts(&self) -> ExecutionFacts<'_, B> {
        let mut facts = self.outcome.facts();
        facts.presents = Some(&self.presents);
        facts.failed_step = self.failed_step;
        if facts.accepted.is_none() {
            if let Some(prefix) = facts.affected {
                if self.completed_steps <= prefix.len() {
                    facts.accepted = Some(&prefix[..self.completed_steps]);
                }
            }
        }
        facts
    }
    fn error(&self) -> Option<&Error> {
        self.error.as_ref().or_else(|| match &self.outcome {
            ImmediateFinishOutcome::Failed(
                ImmediateRunOutcome::Failed { error, .. }
                | ImmediateRunOutcome::Indeterminate { error, .. },
            ) => Some(error),
            _ => None,
        })
    }
}

impl<B: ExecutionBackend + ImmediateBackend> ExecutionMode<B> for ImmediateExecution {
    type State = ImmediateExecutionState<B>;
    type Outcome = ImmediateExecutionOutcome<B>;
    unsafe fn begin(
        device: &Device<B>,
        plan: SubmissionPlan<B>,
        completion: CompletionRequest,
    ) -> Result<Self::State> {
        if plan
            .steps
            .iter()
            .any(|step| step.kind == StepKind::SparseBind)
        {
            return Err(Error::Unsupported(
                "sparse steps are not supported by this execution facade",
            ));
        }
        // SAFETY: the caller retains resources and explicitly finishes.
        let session = unsafe { device.immediate(plan.clone(), completion)? };
        Ok(ImmediateExecutionState {
            session,
            plan,
            cursor: 0,
            failure: None,
            error: None,
            failed_slot: None,
            rejected_present: None,
        })
    }
    fn commands<F: FnOnce(&mut CommandEncoder<B>) -> Result<()>>(
        state: &mut Self::State,
        step: StepId,
        encode: F,
    ) -> Result<()> {
        if state.failure.is_some()
            || state.error.is_some()
            || state
                .plan
                .steps
                .get(state.cursor)
                .is_none_or(|d| d.kind != StepKind::Commands)
            || step != state.plan.step(state.cursor)
        {
            return Err(Error::InvalidState);
        }
        let mut encoder = match state.session.begin_commands() {
            Ok(encoder) => encoder,
            Err(outcome) => {
                let error = run_error(&outcome);
                state.failed_slot = Some(step);
                state.failure = Some(outcome);
                return Err(error);
            }
        };
        if let Err(error) = encode(encoder.commands()) {
            // Dropping closes the active native encoder and retains the native
            // result in the session; keep the original callback error as well.
            drop(encoder);
            state.failed_slot = Some(step);
            state.error = Some(error.clone());
            return Err(error);
        }
        match encoder.close() {
            ImmediateRunOutcome::Issued(_) => {
                state.cursor += 1;
                Ok(())
            }
            outcome => {
                let error = run_error(&outcome);
                state.failed_slot = Some(step);
                state.failure = Some(outcome);
                Err(error)
            }
        }
    }
    fn present(
        state: &mut Self::State,
        step: StepId,
        binding: PresentBinding<B>,
    ) -> std::result::Result<(), PresentError<B>> {
        if state.failure.is_some()
            || state.error.is_some()
            || state
                .plan
                .steps
                .get(state.cursor)
                .is_none_or(|d| d.kind != StepKind::Present)
            || step != state.plan.step(state.cursor)
        {
            return Err(PresentError {
                error: Error::InvalidState,
                binding: Some(binding),
            });
        }
        match state.session.present(binding) {
            ImmediatePresentResult::Rejected { error, binding } => {
                state.rejected_present = Some((step, error.clone()));
                Err(PresentError {
                    error,
                    binding: Some(binding),
                })
            }
            ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_)) => {
                state.rejected_present = None;
                state.cursor += 1;
                Ok(())
            }
            ImmediatePresentResult::Run(outcome) => {
                let error = run_error(&outcome);
                state.rejected_present = None;
                state.failed_slot = Some(step);
                state.failure = Some(outcome);
                Err(PresentError {
                    error,
                    binding: None,
                })
            }
        }
    }
    unsafe fn finish(state: Self::State) -> Self::Outcome {
        let presents = state.plan.steps[..state.cursor]
            .iter()
            .enumerate()
            .filter_map(|(index, desc)| {
                (desc.kind == StepKind::Present).then(|| PresentReceipt {
                    step: Some(state.plan.step(index)),
                    accepted: true,
                })
            })
            .collect();
        let failure_effect = state.session.failure_effect();
        let result = state.session.finish();
        let outcome = match state.failure {
            Some(failure) => ImmediateFinishOutcome::Failed(failure),
            None => result,
        };
        let failed_step = state
            .failed_slot
            .map(|step| FailedStep {
                step,
                effect: failure_effect.unwrap_or(StepEffect::Unknown),
            })
            .or_else(|| {
                state.rejected_present.as_ref().map(|(step, _)| FailedStep {
                    step: *step,
                    effect: StepEffect::None,
                })
            });
        ImmediateExecutionOutcome {
            completed_steps: state.cursor,
            failed_step,
            error: state
                .error
                .or_else(|| state.rejected_present.map(|(_, error)| error)),
            presents,
            outcome,
        }
    }
}

fn run_error<B: Backend>(outcome: &ImmediateRunOutcome<B>) -> Error {
    match outcome {
        ImmediateRunOutcome::Failed { error, .. }
        | ImmediateRunOutcome::Indeterminate { error, .. } => error.clone(),
        ImmediateRunOutcome::Issued(_) => Error::InvalidState,
    }
}
