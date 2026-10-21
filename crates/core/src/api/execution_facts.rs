//! Borrowed observations shared by deferred submission and immediate issue.
//!
//! This view does not own recovery payloads, infer completion, or equate a
//! partially issued immediate step with a completely accepted step. Callers
//! retain the original outcome to recover its error, tail, or uncertainty.
use super::*;

/// The kind of observation from which execution facts were obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionStatus {
    /// All work represented by the operation was accepted or issued.
    Accepted,
    /// The operation was rejected without a native effect.
    Rejected,
    /// A deferred submission accepted a known part of its work.
    PartiallyAccepted,
    /// An immediate operation failed; its known issue prefix is reported.
    Failed,
    /// Native acceptance or the issue boundary cannot be established.
    Indeterminate,
    /// A previous operation already returned the facts.
    AlreadyReported,
    /// A prefix was inspected without its enclosing operation's outcome.
    Unknown,
}

/// The native effect of an unsuccessful execution step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepEffect {
    /// The step is known to have had no native effect.
    None,
    /// Some native work was issued, without completing the step.
    Partial,
    /// The native effect cannot be determined.
    Unknown,
}

/// A failed step and the extent of its known native effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FailedStep {
    /// The execution slot in the compiled plan.
    pub step: StepId,
    /// Effect of the unsuccessful slot, not GPU completion.
    pub effect: StepEffect,
}

/// A zero-copy view of facts reported by an execution operation.
///
/// `None` means the source does not establish that fact; `Some(&[])` means
/// that it explicitly reports an empty set. The slices borrow their original
/// owner and do not transfer or clone completion evidence.
pub struct ExecutionFacts<'a, B: Backend> {
    /// Classification of the source operation, not GPU completion status.
    pub status: ExecutionStatus,
    /// Failed slot when the source can identify one.
    pub failed_step: Option<FailedStep>,
    /// Steps known to have been completely accepted or successfully issued.
    /// This is not necessarily the complete execution history.
    pub accepted: Option<&'a [StepId]>,
    /// Steps in a reported native issue prefix. On failure, the last step may
    /// have only partially issued; the source does not identify its boundary.
    pub affected: Option<&'a [StepId]>,
    /// Reported completion evidence. Coverage concerns native work and must
    /// not be interpreted as proof that a partially issued step succeeded.
    pub checkpoints: Option<&'a [Checkpoint<B>]>,
    /// Presentation handoff facts when the source provides them.
    pub presents: Option<&'a [PresentReceipt]>,
}

impl<B: Backend> SubmissionReceipt<B> {
    /// Borrows the accepted steps and completion/presentation evidence.
    pub fn facts(&self) -> ExecutionFacts<'_, B> {
        ExecutionFacts {
            status: ExecutionStatus::Unknown,
            failed_step: None,
            accepted: Some(&self.accepted),
            affected: None,
            checkpoints: Some(&self.checkpoints),
            presents: Some(&self.presents),
        }
    }
}

impl<B: DeferredBackend> SubmitOutcome<B> {
    /// Borrows known facts without consuming the recovery payload.
    pub fn facts(&self) -> ExecutionFacts<'_, B> {
        match self {
            Self::Accepted(receipt) => ExecutionFacts {
                status: ExecutionStatus::Accepted,
                ..receipt.facts()
            },
            Self::PartiallyAccepted { accepted, .. } => ExecutionFacts {
                status: ExecutionStatus::PartiallyAccepted,
                ..accepted.facts()
            },
            Self::Rejected { .. } => ExecutionFacts {
                status: ExecutionStatus::Rejected,
                failed_step: None,
                accepted: Some(&[]),
                affected: Some(&[]),
                checkpoints: None,
                presents: None,
            },
            Self::Indeterminate { .. } => ExecutionFacts {
                status: ExecutionStatus::Indeterminate,
                failed_step: None,
                accepted: None,
                affected: None,
                checkpoints: None,
                presents: None,
            },
        }
    }
}

impl<B: Backend> IssuedPrefix<B> {
    /// Borrows an issue prefix without assuming every step completed issuing.
    pub fn facts(&self) -> ExecutionFacts<'_, B> {
        ExecutionFacts {
            status: ExecutionStatus::Unknown,
            failed_step: None,
            accepted: None,
            affected: Some(&self.steps),
            checkpoints: Some(&self.checkpoints),
            presents: None,
        }
    }
}

impl<B: Backend> ImmediateRunOutcome<B> {
    /// Borrows the facts reported by this immediate operation.
    pub fn facts(&self) -> ExecutionFacts<'_, B> {
        match self {
            Self::Issued(step) => ExecutionFacts {
                status: ExecutionStatus::Accepted,
                failed_step: None,
                accepted: Some(std::slice::from_ref(step)),
                affected: Some(std::slice::from_ref(step)),
                checkpoints: None,
                presents: None,
            },
            Self::Failed { prefix, .. } => ExecutionFacts {
                status: ExecutionStatus::Failed,
                ..prefix.facts()
            },
            Self::Indeterminate { prefix, .. } => ExecutionFacts {
                status: ExecutionStatus::Indeterminate,
                failed_step: None,
                accepted: None,
                affected: Some(prefix),
                checkpoints: None,
                presents: None,
            },
        }
    }
}

impl<B: Backend> ImmediateFinishOutcome<B> {
    /// Borrows the facts available when the immediate session ended.
    pub fn facts(&self) -> ExecutionFacts<'_, B> {
        match self {
            Self::Issued(prefix) => ExecutionFacts {
                status: ExecutionStatus::Accepted,
                failed_step: None,
                accepted: Some(&prefix.steps),
                ..prefix.facts()
            },
            Self::Failed(outcome) => outcome.facts(),
            Self::AlreadyReported => ExecutionFacts {
                status: ExecutionStatus::AlreadyReported,
                failed_step: None,
                accepted: None,
                affected: None,
                checkpoints: None,
                presents: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
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
                steps: vec![step],
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
}
