//! Facade behavior under the existing immediate and deferred mock SPIs.
use super::*;
use crate::api::tests::type_contract_tests::{
    GL_ABANDONED, GL_CHECKPOINTS, GL_CLOSE_FAILURE, GL_DEVICE, GL_TRANSITIONS, GlLocal,
    NativeShared,
};
use std::{cell::Cell, rc::Rc, sync::Arc};

impl ExecutionBackend for GlLocal {
    type Mode = ImmediateExecution;
}
impl ExecutionBackend for NativeShared {
    type Mode = DeferredExecution;
}

fn capabilities(id: DeviceId) -> DeviceCapabilities {
    let mut caps = DeviceCapabilities::default();
    caps.lanes = vec![LaneInfo {
        id: LaneId {
            device: id,
            ordinal: 0,
        },
        class: LaneClass::Graphics,
        routes: vec![],
    }]
    .into();
    caps
}
fn plan<B: Backend>(device: &Device<B>, kinds: &[StepKind]) -> SubmissionPlan<B> {
    let id = PlanId {
        device: device.id(),
        ordinal: 1,
    };
    let lane = device.lanes()[0].id;
    let steps: Arc<[StepDesc]> = kinds
        .iter()
        .map(|&kind| StepDesc {
            lane,
            kind,
            waits: vec![],
        })
        .collect();
    // SAFETY: the test supplies a device-local, ordered, single-lane plan.
    unsafe { SubmissionPlan::from_compiled(id, steps) }
}
fn immediate() -> Device<GlLocal> {
    let id = DeviceId::allocate().unwrap();
    GL_DEVICE.with(|device| device.set(Some(id)));
    GL_CLOSE_FAILURE.with(|mode| mode.set(0));
    GL_CHECKPOINTS.with(|calls| calls.set(0));
    GL_TRANSITIONS.with(|calls| calls.set(0));
    GL_ABANDONED.with(|calls| calls.set(0));
    // SAFETY: the mock native device and capability lane share this ID.
    unsafe { Device::from_native(id, Rc::new(()), capabilities(id)) }
}

#[test]
fn immediate_native_call_occurs_before_finish_without_step_checkpoint() {
    let device = immediate();
    let plan = plan(&device, &[StepKind::Commands, StepKind::Commands]);
    // SAFETY: the mock records no resources.
    let mut execution =
        unsafe { device.begin_execution(plan.clone(), CompletionRequest::None) }.unwrap();
    execution
        .commands(plan.step(0), |encoder| encoder.transition_resources(&[]))
        .unwrap();
    GL_TRANSITIONS.with(|calls| assert_eq!(calls.get(), 1));
    GL_CHECKPOINTS.with(|calls| assert_eq!(calls.get(), 0));
    execution
        .commands(plan.step(1), |encoder| encoder.transition_resources(&[]))
        .unwrap();
    GL_TRANSITIONS.with(|calls| assert_eq!(calls.get(), 2));
    GL_CHECKPOINTS.with(|calls| assert_eq!(calls.get(), 0));
    // SAFETY: the mock records no resources.
    let _outcome = unsafe { execution.finish() };
    GL_CHECKPOINTS.with(|calls| assert_eq!(calls.get(), 1));
}

#[test]
fn immediate_failed_and_indeterminate_close_preserve_known_prefix() {
    for (failure, expected) in [
        (1, ExecutionStatus::Failed),
        (2, ExecutionStatus::Indeterminate),
    ] {
        let device = immediate();
        let plan = plan(&device, &[StepKind::Commands, StepKind::Commands]);
        // SAFETY: the mock records no resources.
        let mut execution =
            unsafe { device.begin_execution(plan.clone(), CompletionRequest::None) }.unwrap();
        execution.commands(plan.step(0), |_| Ok(())).unwrap();
        GL_CLOSE_FAILURE.with(|mode| mode.set(failure));
        assert!(execution.commands(plan.step(1), |_| Ok(())).is_err());
        // SAFETY: the mock records no resources.
        let outcome = unsafe { execution.finish() };
        assert_eq!(outcome.completed_steps, 1);
        let facts = outcome.facts();
        assert_eq!(facts.status, expected);
        assert_eq!(facts.accepted.unwrap(), &[plan.step(0)]);
        if failure == 1 {
            assert_eq!(facts.affected.unwrap(), &[plan.step(0), plan.step(1)]);
        }
        assert!(outcome.error().is_some());
        GL_CHECKPOINTS.with(|calls| assert_eq!(calls.get(), usize::from(failure == 1)));
    }
}

#[test]
fn dropped_immediate_execution_is_abandoned_without_idle_or_checkpoint() {
    let device = immediate();
    let plan = plan(&device, &[StepKind::Commands]);
    // SAFETY: this mock uses no GPU resources.
    let mut execution =
        unsafe { device.begin_execution(plan.clone(), CompletionRequest::None) }.unwrap();
    execution.commands(plan.step(0), |_| Ok(())).unwrap();
    drop(execution);
    GL_ABANDONED.with(|calls| assert_eq!(calls.get(), 1));
    GL_CHECKPOINTS.with(|calls| assert_eq!(calls.get(), 0));
}
fn deferred() -> Device<NativeShared> {
    let id = DeviceId::allocate().unwrap();
    // SAFETY: the mock native device and capability lane share this ID.
    unsafe { Device::from_native(id, Arc::new(()), capabilities(id)) }
}

#[test]
fn immediate_callback_runs_now_and_original_failure_survives_close() {
    let device = immediate();
    let plan = plan(&device, &[StepKind::Commands, StepKind::Commands]);
    // SAFETY: no GPU resources are referenced by this mock execution.
    let mut execution =
        unsafe { device.begin_execution(plan.clone(), CompletionRequest::None) }.unwrap();
    let calls = Cell::new(0);
    execution
        .commands(plan.step(0), |_| {
            calls.set(calls.get() + 1);
            Ok(())
        })
        .unwrap();
    assert_eq!(calls.get(), 1);
    let original = Error::Backend("original encoding failure".into());
    assert_eq!(
        execution.commands(plan.step(1), |_| {
            calls.set(calls.get() + 1);
            Err(original.clone())
        }),
        Err(original.clone())
    );
    assert_eq!(calls.get(), 2);
    // SAFETY: no GPU resources are referenced.
    let outcome = unsafe { execution.finish() };
    assert_eq!(outcome.completed_steps, 1);
    assert_eq!(outcome.error(), Some(&original));
    let facts = outcome.facts();
    assert_eq!(facts.accepted.unwrap(), &[plan.step(0)]);
    assert_eq!(facts.status, ExecutionStatus::Failed);
}

#[test]
fn failed_callback_cannot_become_success_even_if_abort_issued() {
    let device = immediate();
    let plan = plan(&device, &[StepKind::Commands]);
    // SAFETY: the mock records no GPU resources.
    let mut execution =
        unsafe { device.begin_execution(plan.clone(), CompletionRequest::None) }.unwrap();
    GL_CLOSE_FAILURE.with(|mode| mode.set(3));
    let original = Error::Backend("callback failed".into());
    assert_eq!(
        execution.commands(plan.step(0), |_| Err(original.clone())),
        Err(original.clone())
    );
    // SAFETY: the mock records no GPU resources.
    let outcome = unsafe { execution.finish() };
    assert_eq!(outcome.error(), Some(&original));
    assert_eq!(outcome.facts().status, ExecutionStatus::Failed);
    assert_eq!(outcome.facts().accepted.unwrap(), &[]);
    assert_eq!(outcome.facts().affected.unwrap(), &[plan.step(0)]);
}

#[test]
fn deferred_failure_never_submits_and_preserves_recorded_steps() {
    let device = deferred();
    let plan = plan(&device, &[StepKind::Commands, StepKind::Commands]);
    // SAFETY: the mock records no resources and never issues GPU work.
    let mut execution =
        unsafe { device.begin_execution(plan.clone(), CompletionRequest::None) }.unwrap();
    execution.commands(plan.step(0), |_| Ok(())).unwrap();
    let original = Error::Backend("encode failed".into());
    assert_eq!(
        execution.commands(plan.step(1), |_| Err(original.clone())),
        Err(original.clone())
    );
    // SAFETY: the mock has no GPU resource obligations.
    let outcome = unsafe { execution.finish() };
    assert_eq!(outcome.error(), Some(&original));
    match outcome {
        DeferredExecutionOutcome::Unsubmitted { steps, .. } => assert_eq!(steps.len(), 1),
        DeferredExecutionOutcome::Submitted(_) => panic!("failed encoding was submitted"),
    }
}

#[test]
fn recorded_commands_rejection_preserves_recordings_and_allows_retry() {
    let device = deferred();
    let foreign = deferred();
    let plan = plan(&device, &[StepKind::Commands]);
    let lane = device.lanes()[0].id;
    let foreign_work = foreign
        .encoder(foreign.lanes()[0].id)
        .unwrap()
        .finish()
        .unwrap();
    // SAFETY: the mock has no GPU resource obligations.
    let mut execution =
        unsafe { device.begin_execution(plan.clone(), CompletionRequest::None) }.unwrap();
    let rejected = match execution.recorded_commands(plan.step(0), vec![foreign_work]) {
        Err(rejected) => rejected,
        Ok(()) => panic!("foreign work was accepted"),
    };
    assert_eq!(rejected.error, Error::WrongDevice);
    assert_eq!(rejected.work.len(), 1);
    assert_eq!(rejected.work[0].device(), foreign.id());
    let work = vec![
        device.encoder(lane).unwrap().finish().unwrap(),
        device.encoder(lane).unwrap().finish().unwrap(),
    ];
    let rejected = match execution.recorded_commands(plan.step(1), work) {
        Err(rejected) => rejected,
        Ok(()) => panic!("wrong slot was accepted"),
    };
    assert_eq!(rejected.error, Error::InvalidState);
    assert_eq!(rejected.work.len(), 2);
    assert!(
        execution
            .recorded_commands(plan.step(0), rejected.work)
            .is_ok()
    );
    // SAFETY: the mock has no GPU resource obligations.
    let outcome = unsafe { execution.finish() };
    match outcome {
        DeferredExecutionOutcome::Submitted(SubmitOutcome::Rejected {
            work: RetryableSubmit::Fresh(packet),
            ..
        }) => match &packet.steps[0].payload {
            StepPayload::Commands(work) => assert_eq!(work.len(), 2),
            _ => panic!("recordings were not bound to the commands step"),
        },
        _ => panic!("mock submission did not preserve the recorded packet"),
    }
}

#[test]
fn recorded_commands_wrong_lane_preserves_work() {
    let device = deferred();
    let plan = plan(&device, &[StepKind::Commands]);
    // SAFETY: the mock uses no GPU resources and issues no native work.
    let mut execution =
        unsafe { device.begin_execution(plan.clone(), CompletionRequest::None) }.unwrap();
    let other_lane = LaneId {
        device: device.id(),
        ordinal: 7,
    };
    let work = RecordedWork {
        device: device.id(),
        lane: other_lane,
        native: (),
    };
    let rejected = match execution.recorded_commands(plan.step(0), vec![work]) {
        Err(rejected) => rejected,
        Ok(()) => panic!("wrong lane was accepted"),
    };
    assert_eq!(rejected.error, Error::WrongLane);
    assert_eq!(rejected.work[0].lane(), other_lane);
    // SAFETY: the mock uses no GPU resources and issues no native work.
    assert!(matches!(
        unsafe { execution.finish() },
        DeferredExecutionOutcome::Unsubmitted { .. }
    ));
}

#[test]
fn invalid_order_is_rejected_before_invoking_the_callback() {
    let device = immediate();
    let plan = plan(&device, &[StepKind::Commands, StepKind::Commands]);
    // SAFETY: this mock references no GPU resources.
    let mut execution =
        unsafe { device.begin_execution(plan.clone(), CompletionRequest::None) }.unwrap();
    let invoked = Cell::new(false);
    assert_eq!(
        execution.commands(plan.step(1), |_| {
            invoked.set(true);
            Ok(())
        }),
        Err(Error::InvalidState)
    );
    assert!(!invoked.get());
    execution.commands(plan.step(0), |_| Ok(())).unwrap();
    execution.commands(plan.step(1), |_| Ok(())).unwrap();
    // SAFETY: this mock references no GPU resources.
    let outcome = unsafe { execution.finish() };
    assert_eq!(outcome.facts().status, ExecutionStatus::Accepted);
}

#[test]
fn unsupported_sparse_plan_fails_before_commands_can_issue() {
    let device = immediate();
    let plan = plan(&device, &[StepKind::Commands, StepKind::SparseBind]);
    // SAFETY: this fixture has no resources; failure happens before native issue.
    let result = unsafe { device.begin_execution(plan, CompletionRequest::None) };
    assert!(matches!(result, Err(Error::Unsupported(_))));
}

#[test]
fn rejected_present_keeps_lease_for_retry() {
    let device = immediate();
    let plan = plan(&device, &[StepKind::Commands, StepKind::Present]);
    // SAFETY: this mock references no GPU resources.
    let mut execution =
        unsafe { device.begin_execution(plan.clone(), CompletionRequest::None) }.unwrap();
    let binding = PresentBinding::new(SurfaceImage {
        device: device.id(),
        surface: SurfaceId::allocate().unwrap(),
        generation: 0,
        native: (),
    });
    let rejected = match execution.present(plan.step(1), binding) {
        Err(rejected) => rejected,
        Ok(()) => panic!("out-of-order presentation unexpectedly accepted"),
    };
    assert_eq!(rejected.error, Error::InvalidState);
    execution.commands(plan.step(0), |_| Ok(())).unwrap();
    assert!(
        execution
            .present(plan.step(1), rejected.binding.unwrap())
            .is_ok()
    );
    // SAFETY: this mock references no GPU resources.
    let outcome = unsafe { execution.finish() };
    assert_eq!(outcome.facts().status, ExecutionStatus::Accepted);
    let facts = outcome.facts();
    let presents = facts.presents.unwrap();
    assert_eq!(presents.len(), 1);
    assert_eq!(presents[0].step, Some(plan.step(1)));
    assert!(presents[0].accepted);
}
