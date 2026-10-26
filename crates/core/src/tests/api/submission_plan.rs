//! Unit tests for the parent module.

use super::*;
use crate::api::tests::type_contract_tests::NativeShared;

impl IssuedSteps {
    pub(crate) fn from_test_steps(all: Arc<[StepId]>) -> Self {
        let len = all.len();
        Self { all, len }
    }
}

impl<B: Backend> SubmissionPlan<B> {
    pub(crate) fn topology_validation_cached(&self) -> bool {
        self.metadata.validation.get().is_some()
    }

    pub(crate) fn topology_validation_runs(&self) -> usize {
        self.metadata.validation_runs.load(Ordering::Relaxed)
    }
}

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
