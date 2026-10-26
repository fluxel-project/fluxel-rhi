//! Unit tests for the parent module.

use super::*;
use crate::api::tests::type_contract_tests::NativeShared;
use std::sync::Arc;

#[test]
fn issue_failure_reports_effects_without_claiming_complete_steps() {
    let device = DeviceId::allocate().unwrap();
    let step = StepId {
        plan: PlanId { device, ordinal: 1 },
        index: 0,
    };
    let native = Arc::new(());
    let checkpoint = Checkpoint {
        // SAFETY: this mock completion belongs to the allocated test device.
        point: unsafe { CompletionPoint::from_native(device, native.clone()) },
        covers: CompletionCoverage {
            lane_frontiers: vec![],
        },
    };
    let outcome = ImmediateRunOutcome::<NativeShared>::Failed {
        error: Error::InvalidState,
        prefix: IssuedPrefix {
            steps: IssuedSteps::from_test_steps(Arc::from([step])),
            checkpoints: vec![checkpoint],
        },
    };
    let count = Arc::strong_count(&native);
    let facts = outcome.facts();
    assert_eq!(facts.status, ExecutionStatus::Failed);
    assert!(facts.accepted.is_none());
    assert_eq!(facts.affected.unwrap(), &[step]);
    assert_eq!(facts.checkpoints.unwrap().len(), 1);
    assert_eq!(Arc::strong_count(&native), count);
    assert!(matches!(outcome, ImmediateRunOutcome::Failed { .. }));
}

#[test]
fn receipt_alone_does_not_claim_full_submission_acceptance() {
    let device = DeviceId::allocate().unwrap();
    let step = StepId {
        plan: PlanId { device, ordinal: 2 },
        index: 0,
    };
    let receipt = SubmissionReceipt::<NativeShared> {
        device,
        index: None,
        accepted: vec![step],
        checkpoints: vec![],
        presents: vec![],
    };
    assert_eq!(receipt.facts().status, ExecutionStatus::Unknown);
    let outcome = SubmitOutcome::Accepted(receipt);
    let facts = outcome.facts();
    assert_eq!(facts.status, ExecutionStatus::Accepted);
    assert_eq!(facts.accepted.unwrap(), &[step]);
    assert!(facts.affected.is_none());
    assert_eq!(facts.checkpoints.unwrap().len(), 0);
}
