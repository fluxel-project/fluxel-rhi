//! Ordered native handoff for one-time initialization of private ray inputs.

use fluxel_rhi_core::api::{Error, Result};
use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Clone)]
pub(crate) struct PreparedInputId(Arc<AtomicBool>);

impl PreparedInputId {
    pub(super) fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }
}

impl PartialEq for PreparedInputId {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for PreparedInputId {}

pub(crate) enum PreparedInputOp {
    Initialize { id: PreparedInputId, segment: usize },
    Read(PreparedInputId),
}

#[derive(Default)]
pub(super) struct PreparedInputAdmission {
    initializations: Option<HashSet<*const AtomicBool>>,
}

impl PreparedInputAdmission {
    pub(super) fn validate(&mut self, ops: &[PreparedInputOp]) -> Result<()> {
        for op in ops {
            match op {
                PreparedInputOp::Initialize { id, .. } => {
                    if id.0.load(Ordering::Relaxed)
                        || !self
                            .initializations
                            .get_or_insert_with(HashSet::new)
                            .insert(Arc::as_ptr(&id.0))
                    {
                        return Err(Error::InvalidState);
                    }
                }
                PreparedInputOp::Read(id) => {
                    if !id.0.load(Ordering::Relaxed)
                        && !self
                            .initializations
                            .as_ref()
                            .is_some_and(|initializations| {
                                initializations.contains(&Arc::as_ptr(&id.0))
                            })
                    {
                        return Err(Error::InvalidState);
                    }
                }
            }
        }
        Ok(())
    }
}

pub(super) fn publish(ops: &[PreparedInputOp], segment: usize) {
    for op in ops {
        if let PreparedInputOp::Initialize {
            id,
            segment: initialization_segment,
        } = op
        {
            if *initialization_segment == segment {
                // All admission and publication run under the submission gate.
                id.0.store(true, Ordering::Relaxed);
            }
        }
    }
}
