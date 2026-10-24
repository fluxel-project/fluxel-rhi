//! Immutable submission topology and its reusable validation/prefix metadata.
//!
//! A compiled plan is permanently bound to one immutable Device identity and
//! lane set.  Dynamic packet bindings, completion requests, and resource
//! ownership are deliberately not cached here; they remain per-submission
//! validation obligations.

use super::*;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    marker::PhantomData,
    ops::Deref,
    sync::{Arc, OnceLock},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    Commands,
    SparseBind,
    Present,
}

#[derive(Clone, Copy, Debug)]
pub struct StepWait {
    pub previous: StepId,
    pub destination: AccessScope,
}

pub struct StepDesc {
    pub lane: LaneId,
    pub kind: StepKind,
    pub waits: Vec<StepWait>,
}

struct PlanMetadata {
    step_ids: Arc<[StepId]>,
    has_sparse: bool,
    // PlanId binds topology to one device, and Device capabilities/lanes are
    // immutable after `Device::from_native`. Consequently both successes and
    // topology errors are stable across every clone of this compiled plan.
    validation: OnceLock<Result<()>>,
    #[cfg(test)]
    validation_runs: AtomicUsize,
}

/// A shared contiguous prefix of a compiled plan's cached step identities.
///
/// It is cheap to clone and never allocates a per-execution `Vec`. Borrowed
/// iteration yields `&StepId`; consuming `into_iter()` yields copied `StepId`
/// values while retaining the same shared backing allocation.
#[derive(Clone)]
pub struct IssuedSteps {
    all: Arc<[StepId]>,
    len: usize,
}

impl std::fmt::Debug for IssuedSteps {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(formatter)
    }
}

impl PartialEq for IssuedSteps {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for IssuedSteps {}

impl IssuedSteps {
    pub fn as_slice(&self) -> &[StepId] {
        &self.all[..self.len]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> std::slice::Iter<'_, StepId> {
        self.as_slice().iter()
    }

    #[cfg(test)]
    pub(crate) fn from_test_steps(all: Arc<[StepId]>) -> Self {
        let len = all.len();
        Self { all, len }
    }
}

impl Deref for IssuedSteps {
    type Target = [StepId];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl AsRef<[StepId]> for IssuedSteps {
    fn as_ref(&self) -> &[StepId] {
        self.as_slice()
    }
}

impl<'a> IntoIterator for &'a IssuedSteps {
    type Item = &'a StepId;
    type IntoIter = std::slice::Iter<'a, StepId>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct IssuedStepsIntoIter {
    all: Arc<[StepId]>,
    index: usize,
    len: usize,
}

impl Iterator for IssuedStepsIntoIter {
    type Item = StepId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.len {
            return None;
        }
        let step = self.all.get(self.index).copied()?;
        self.index += 1;
        Some(step)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len.saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for IssuedStepsIntoIter {}
impl std::iter::FusedIterator for IssuedStepsIntoIter {}

impl IntoIterator for IssuedSteps {
    type Item = StepId;
    type IntoIter = IssuedStepsIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        IssuedStepsIntoIter {
            all: self.all,
            index: 0,
            len: self.len,
        }
    }
}

pub struct SubmissionPlan<B: Backend> {
    pub(crate) id: PlanId,
    pub(crate) steps: Arc<[StepDesc]>,
    metadata: Arc<PlanMetadata>,
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> Clone for SubmissionPlan<B> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            steps: self.steps.clone(),
            metadata: self.metadata.clone(),
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> SubmissionPlan<B> {
    /// # Safety
    /// Renderer supplies an ordered, device-local, realizable execution layout.
    /// RHI does not sort or validate a second DAG at submission time.
    pub unsafe fn from_compiled(id: PlanId, steps: Arc<[StepDesc]>) -> Self {
        // Keep malformed oversized topology cheap until the usual execution
        // validation reports InvalidRange; never allocate a second enormous
        // step-id array merely to describe work that cannot execute.
        let step_ids = if steps.len() <= u32::MAX as usize {
            (0..steps.len())
                .map(|index| StepId {
                    plan: id,
                    index: index as u32,
                })
                .collect()
        } else {
            Arc::from([])
        };
        let has_sparse = steps.iter().any(|step| step.kind == StepKind::SparseBind);
        Self {
            id,
            steps,
            metadata: Arc::new(PlanMetadata {
                step_ids,
                has_sparse,
                validation: OnceLock::new(),
                #[cfg(test)]
                validation_runs: AtomicUsize::new(0),
            }),
            _backend: PhantomData,
        }
    }

    pub fn id(&self) -> PlanId {
        self.id
    }

    pub fn steps(&self) -> &[StepDesc] {
        &self.steps
    }

    pub fn step(&self, index: usize) -> StepId {
        StepId {
            plan: self.id,
            // `validate_for_execution` limits a usable plan to `u32::MAX`
            // entries, so this sentinel can never identify a valid slot. Do
            // not let an oversized public `usize` wrap onto an earlier step.
            index: u32::try_from(index).unwrap_or(u32::MAX),
        }
    }

    pub(crate) fn issued_steps(&self, len: usize) -> IssuedSteps {
        debug_assert!(len <= self.steps.len());
        IssuedSteps {
            all: self.metadata.step_ids.clone(),
            len,
        }
    }

    pub(crate) fn has_sparse(&self) -> bool {
        self.metadata.has_sparse
    }

    fn validate_topology(&self, device: &Device<B>) -> Result<()> {
        #[cfg(test)]
        self.metadata
            .validation_runs
            .fetch_add(1, Ordering::Relaxed);
        if self.steps.len() > u32::MAX as usize {
            return Err(Error::InvalidRange);
        }
        for (index, step) in self.steps.iter().enumerate() {
            if step.lane.device != device.id {
                return Err(Error::WrongDevice);
            }
            if !device.lanes().iter().any(|lane| lane.id == step.lane) {
                return Err(Error::WrongLane);
            }
            for wait in &step.waits {
                if wait.previous.plan != self.id || wait.previous.index as usize >= index {
                    return Err(Error::InvalidState);
                }
            }
        }
        Ok(())
    }

    /// Validate the immutable device-local ordered topology before executing.
    ///
    /// The cache is shared by plan clones only after the dynamic device-id
    /// check. Packet bindings and completion requests remain uncached.
    pub fn validate_for_execution(&self, device: &Device<B>) -> Result<()> {
        if self.id.device != device.id {
            return Err(Error::WrongDevice);
        }
        self.metadata
            .validation
            .get_or_init(|| self.validate_topology(device))
            .clone()
    }

    #[cfg(test)]
    pub(crate) fn topology_validation_cached(&self) -> bool {
        self.metadata.validation.get().is_some()
    }

    #[cfg(test)]
    pub(crate) fn topology_validation_runs(&self) -> usize {
        self.metadata.validation_runs.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::tests::type_contract_tests::NativeShared;

    fn device() -> Device<NativeShared> {
        let id = DeviceId::allocate().unwrap();
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
        // SAFETY: the mock native device and lane use this allocated ID.
        unsafe { Device::from_native(id, Arc::new(()), caps) }
    }

    fn commands(device: &Device<NativeShared>, count: usize) -> Arc<[StepDesc]> {
        Arc::from(
            (0..count)
                .map(|_| StepDesc {
                    lane: device.lanes()[0].id,
                    kind: StepKind::Commands,
                    waits: vec![],
                })
                .collect::<Vec<_>>(),
        )
    }

    fn plan(device: &Device<NativeShared>, steps: Arc<[StepDesc]>) -> SubmissionPlan<NativeShared> {
        // SAFETY: fixtures deliberately control topology validity.
        unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: device.id(),
                    ordinal: 71,
                },
                steps,
            )
        }
    }

    #[test]
    fn topology_cache_and_compact_prefix_are_shared_by_clones() {
        let device = device();
        let plan = plan(&device, commands(&device, 3));
        assert!(!plan.topology_validation_cached());
        for _ in 0..128 {
            plan.clone().validate_for_execution(&device).unwrap();
        }
        assert!(plan.topology_validation_cached());
        // Typical execution reuses the same compiled plan. This diagnostic
        // proves those begins perform no second topology/lane scan.
        assert_eq!(plan.topology_validation_runs(), 1);
        let first = plan.issued_steps(2);
        let second = plan.clone().issued_steps(2);
        assert_eq!(first.as_slice(), &[plan.step(0), plan.step(1)]);
        assert_eq!(first.as_slice().as_ptr(), second.as_slice().as_ptr());
        assert!(plan.issued_steps(0).into_iter().next().is_none());
        assert_eq!(plan.issued_steps(3).into_iter().count(), 3);
        assert_eq!(plan.step(9).index, 9);
        assert_eq!(
            first.iter().copied().collect::<Vec<_>>(),
            first.clone().into_iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn foreign_device_is_not_cached_and_invalid_topology_is() {
        let device = device();
        let foreign = self::device();
        let compiled = plan(&device, commands(&device, 1));
        assert_eq!(
            compiled.validate_for_execution(&foreign),
            Err(Error::WrongDevice)
        );
        assert!(!compiled.topology_validation_cached());
        compiled.validate_for_execution(&device).unwrap();
        assert_eq!(
            compiled.validate_for_execution(&foreign),
            Err(Error::WrongDevice)
        );
        assert_eq!(compiled.topology_validation_runs(), 1);
        let invalid = self::plan(
            &device,
            Arc::from([StepDesc {
                lane: foreign.lanes()[0].id,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        );
        assert_eq!(
            invalid.validate_for_execution(&device),
            Err(Error::WrongDevice)
        );
        assert!(invalid.topology_validation_cached());
        assert_eq!(
            invalid.clone().validate_for_execution(&device),
            Err(Error::WrongDevice)
        );
    }

    #[test]
    fn packet_binding_validation_survives_a_topology_cache_hit() {
        let device = device();
        let plan = plan(&device, commands(&device, 1));
        plan.validate_for_execution(&device).unwrap();
        let packet = SubmitPacket::<NativeShared> {
            plan,
            steps: vec![],
            completion: CompletionRequest::None,
        };
        assert_eq!(packet.validate(device.id()), Err(Error::InvalidState));
    }
}
