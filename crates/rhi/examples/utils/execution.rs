//! Shared example execution of one compiled Commands -> Present plan.
//!
//! The host retains scene resources through GPU completion. On failure this
//! standalone harness keeps the owning outcome until an idle drain succeeds;
//! browser failures trap because Drop cannot perform an asynchronous drain.
use super::submission::frame_plan;
use fluxel_rhi::api::*;

pub fn execute_frame<B: ExecutionBackend, T>(
    device: &Device<B>,
    lane: LaneId,
    image: SurfaceImage<B>,
    completion: CompletionRequest,
    encode: impl FnOnce(&mut CommandEncoder<B>, &SurfaceImage<B>) -> Result<T>,
) -> Result<(T, <B::Mode as ExecutionMode<B>>::Outcome)> {
    if lane.device != device.id() || image.device() != device.id() {
        return Err(Error::WrongDevice);
    }
    let plan = frame_plan::<B>(device.id(), lane)?;
    let commands = plan.step(0);
    let present = plan.step(1);
    // SAFETY: example hosts keep scene resources alive through their GPU drain;
    // the plan is a single ordered lane with explicit resource transitions.
    let mut execution = unsafe { device.begin_execution(plan, completion)? };
    let mut value = None;
    let command_result = execution.commands(commands, |encoder| {
        value = Some(encode(encoder, &image)?);
        Ok(())
    });
    let mut present_error = None;
    let error = match command_result {
        Ok(()) => match execution.present(present, PresentBinding::new(image)) {
            Ok(()) => None,
            Err(failure) => {
                let error = failure.error.clone();
                present_error = Some(failure);
                Some(error)
            }
        },
        Err(error) => Some(error),
    };
    // SAFETY: scene resources and the owning result remain alive until the
    // host proves completion; errors drain before dropping result or binding.
    let outcome = unsafe { execution.finish() };
    let error = error.or_else(|| outcome.error().cloned()).or_else(|| {
        (outcome.facts().status != ExecutionStatus::Accepted).then_some(Error::InvalidState)
    });
    if let Some(error) = error {
        // A clean rejection has no native effects; otherwise hold both owning
        // values through a real drain, or terminate before their destructors.
        if outcome.facts().status != ExecutionStatus::Rejected {
            #[cfg(not(target_arch = "wasm32"))]
            if let Err(drain_error) = super::block_on(device.wait_idle()) {
                super::lifecycle::unsettled(&format!("execution drain failed: {drain_error}"));
            }
            #[cfg(target_arch = "wasm32")]
            super::lifecycle::unsettled("execution failed before browser GPU drain");
        }
        drop(present_error);
        drop(outcome);
        return Err(error);
    }
    Ok((value.ok_or(Error::InvalidState)?, outcome))
}

pub fn validate_execution_outcome<B: ExecutionBackend>(
    outcome: &impl ExecutionOutcome<B>,
) -> Result<()> {
    if let Some(error) = outcome.error() {
        return Err(error.clone());
    }
    if outcome.facts().status != ExecutionStatus::Accepted {
        return Err(Error::InvalidState);
    }
    Ok(())
}

pub fn execution_end_checkpoint<B: ExecutionBackend>(
    outcome: &impl ExecutionOutcome<B>,
) -> Result<CompletionPoint<B>> {
    validate_execution_outcome(outcome)?;
    let facts = outcome.facts();
    let points = facts.checkpoints.ok_or(Error::InvalidState)?;
    if points.len() != 1 {
        return Err(Error::InvalidState);
    }
    Ok(points[0].point.clone())
}
