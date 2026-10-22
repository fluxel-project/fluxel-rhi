//! Contract tests for Metal plan acceptance and asynchronous completion state.

use super::*;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

#[test]
fn checkpoint_frontier_is_a_snapshot_of_the_accepted_lane_prefix() {
    let device = api::DeviceId::allocate().unwrap();
    let plan = api::PlanId { device, ordinal: 9 };
    let graphics = LaneId { device, ordinal: 0 };
    let transfer = LaneId { device, ordinal: 1 };
    let first = api::StepId { plan, index: 0 };
    let second = api::StepId { plan, index: 1 };
    let later = api::StepId { plan, index: 2 };
    let mut frontiers = Vec::new();

    record_frontier(&mut frontiers, graphics, first);
    let checkpoint_snapshot = frontiers.clone();
    record_frontier(&mut frontiers, transfer, second);
    record_frontier(&mut frontiers, graphics, later);

    assert_eq!(checkpoint_snapshot, vec![(graphics, first)]);
    assert_eq!(frontiers, vec![(transfer, second), (graphics, later)]);
}

#[test]
fn completion_future_transitions_from_pending_to_terminal_once() {
    let completion = Completion::new();
    let mut future = CompletionFuture(vec![completion.clone()]);
    let mut context = Context::from_waker(std::task::Waker::noop());

    assert!(matches!(
        Pin::new(&mut future).poll(&mut context),
        Poll::Pending
    ));
    completion.publish(Ok(()));
    completion.publish(Err(Error::DeviceLost));
    assert!(matches!(
        Pin::new(&mut future).poll(&mut context),
        Poll::Ready(Ok(()))
    ));
}

#[test]
fn completion_future_preserves_device_loss_result() {
    let completion = Completion::new();
    completion.publish(Err(Error::DeviceLost));
    let mut future = CompletionFuture(vec![completion]);
    let mut context = Context::from_waker(std::task::Waker::noop());

    assert!(matches!(
        Pin::new(&mut future).poll(&mut context),
        Poll::Ready(Err(Error::DeviceLost))
    ));
}

#[test]
fn aggregate_completion_waits_for_all_callbacks_and_preserves_failure() {
    let first = Completion::new();
    let second = Completion::new();
    let marker = Completion::new();
    let mut future = CompletionFuture(vec![first.clone(), second.clone(), marker.clone()]);
    let mut context = Context::from_waker(std::task::Waker::noop());
    marker.publish(Ok(()));
    second.publish(Err(Error::DeviceLost));
    assert!(matches!(
        Pin::new(&mut future).poll(&mut context),
        Poll::Pending
    ));
    first.publish(Ok(()));
    assert!(matches!(
        Pin::new(&mut future).poll(&mut context),
        Poll::Ready(Err(Error::DeviceLost))
    ));
}

#[test]
fn checkpoint_join_waits_for_dependency_failure_processing() {
    let command = Completion::new();
    let marker = Completion::new();
    let checkpoint = Completion::new();
    join_into(&checkpoint, vec![command.clone(), marker.clone()]);
    marker.publish(Ok(()));
    assert!(checkpoint.result().is_none());
    command.publish(Err(Error::DeviceLost));
    assert!(matches!(checkpoint.result(), Some(Err(Error::DeviceLost))));
}

#[test]
fn chained_executions_propagate_producer_failure_without_unrelated_work() {
    let producer = Completion::new();
    let consumer_native = Completion::new();
    let unrelated = Completion::new();
    let consumer = Completion::new();
    join_into(&consumer, vec![producer.clone(), consumer_native.clone()]);
    consumer_native.publish(Ok(()));
    assert!(consumer.result().is_none());
    producer.publish(Err(Error::DeviceLost));
    assert!(matches!(consumer.result(), Some(Err(Error::DeviceLost))));
    assert!(unrelated.result().is_none());
}

#[test]
fn completion_handles_are_thread_safe_for_executor_wakers() {
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_send<T: Send>() {}

    assert_send_sync::<Completion>();
    assert_send::<CompletionFuture>();
}

#[test]
fn command_plan_accepts_ordered_same_lane_dependencies() {
    let device = api::DeviceId::allocate().unwrap();
    let id = api::PlanId { device, ordinal: 1 };
    let lane = LaneId { device, ordinal: 0 };
    let access = api::AccessScope {
        stages: api::StageMask(0),
        accesses: api::AccessMask(0),
    };
    // SAFETY: this fixture supplies an ordered, device-local, realizable
    // two-step plan and exercises only the backend's shape predicate.
    let plan = unsafe {
        SubmissionPlan::<Metal>::from_compiled(
            id,
            Arc::from([
                api::StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: Vec::new(),
                },
                api::StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![api::StepWait {
                        previous: api::StepId { plan: id, index: 0 },
                        destination: access,
                    }],
                },
            ]),
        )
    };
    assert!(supports_command_plan(&plan, 0));
}

#[test]
fn command_plan_accepts_commands_followed_by_one_present() {
    let device = api::DeviceId::allocate().unwrap();
    let id = api::PlanId { device, ordinal: 3 };
    let lane = LaneId { device, ordinal: 0 };
    // SAFETY: this fixture is a valid, ordered graphics plan; it tests
    // plan-shape acceptance only and never creates or submits native work.
    let plan = unsafe {
        SubmissionPlan::<Metal>::from_compiled(
            id,
            Arc::from([
                api::StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: Vec::new(),
                },
                api::StepDesc {
                    lane,
                    kind: StepKind::Present,
                    waits: vec![api::StepWait {
                        previous: api::StepId { plan: id, index: 0 },
                        destination: api::AccessScope {
                            stages: api::StageMask(0),
                            accesses: api::AccessMask(0),
                        },
                    }],
                },
            ]),
        )
    };
    assert!(supports_command_plan(&plan, 0));
}

#[test]
fn command_plan_rejects_other_step_kinds_and_lanes() {
    let device = api::DeviceId::allocate().unwrap();
    let id = api::PlanId { device, ordinal: 2 };
    let lane = LaneId { device, ordinal: 0 };
    // SAFETY: these deliberately unsupported descriptors are inspected by
    // the shape predicate only and are never submitted to a native queue.
    let sparse = unsafe {
        SubmissionPlan::<Metal>::from_compiled(
            id,
            Arc::from([api::StepDesc {
                lane,
                kind: StepKind::SparseBind,
                waits: Vec::new(),
            }]),
        )
    };
    assert!(!supports_command_plan(&sparse, 0));

    // SAFETY: same as above; the invalid lane is a rejection fixture only.
    let other_lane = unsafe {
        SubmissionPlan::<Metal>::from_compiled(
            id,
            Arc::from([api::StepDesc {
                lane: LaneId { device, ordinal: 1 },
                kind: StepKind::Commands,
                waits: Vec::new(),
            }]),
        )
    };
    assert!(!supports_command_plan(&other_lane, 0));
}
