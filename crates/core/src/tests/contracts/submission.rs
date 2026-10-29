//! Recording, provenance, and immediate-submission contracts.
use super::super::support::*;
use crate::{api::*, backend::*};
use std::{
    marker::PhantomData,
    ops::Range,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[test]
fn native_recording_can_be_borrowed_without_consumption() {
    let device = DeviceId::allocate().unwrap();
    let work = RecordedWork::<NativeShared> {
        device,
        lane: LaneId { device, ordinal: 0 },
        native: (),
    };
    // SAFETY: this mock backend only inspects its own work and retains nothing.
    assert_eq!(unsafe { work.native_ref() }, &());
    assert_eq!(work.device(), device);
    // SAFETY: the mock backend consumes its recording exactly once.
    assert_eq!(unsafe { work.into_native() }, ());
}
#[test]
fn deferred_binding_rejects_wrong_lane_and_foreign_tail() {
    let device_id = DeviceId::allocate().unwrap();
    let other_id = DeviceId::allocate().unwrap();
    // SAFETY: the fixture pairs this native device with its newly allocated ID.
    let device = unsafe {
        Device::<NativeShared>::from_native(device_id, Arc::new(()), DeviceCapabilities::default())
    };
    let lane = LaneId {
        device: device_id,
        ordinal: 0,
    };
    let wrong_lane = LaneId {
        device: device_id,
        ordinal: 1,
    };
    // SAFETY: this test provides an ordered, device-local compiled plan.
    let plan = unsafe {
        SubmissionPlan::<NativeShared>::from_compiled(
            PlanId {
                device: device_id,
                ordinal: 1,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: Vec::new(),
            }]),
        )
    };
    let packet = SubmitPacket {
        steps: vec![BoundStep {
            step: plan.step(0),
            payload: StepPayload::Commands(vec![RecordedWork {
                device: device_id,
                lane: wrong_lane,
                native: (),
            }]),
        }],
        plan: plan.clone(),
        completion: CompletionRequest::End,
    };
    // SAFETY: the test intentionally invokes submit to assert Core rejects the invalid lane.
    match unsafe { device.queue().submit(packet) } {
        SubmitOutcome::Rejected {
            error: Error::WrongLane,
            ..
        } => {}
        _ => panic!("wrong-lane work reached the backend"),
    }
    // SAFETY: this test constructs a device-local plan for the other fixture device.
    let foreign_plan = unsafe {
        SubmissionPlan::<NativeShared>::from_compiled(
            PlanId {
                device: other_id,
                ordinal: 1,
            },
            Arc::from([]),
        )
    };
    let tail = DeferredTail {
        plan: foreign_plan,
        next_step: 0,
        pending: Vec::new(),
        accepted_frontier: Vec::new(),
        completion: CompletionRequest::None,
    };
    // SAFETY: the test intentionally invokes resume to assert Core rejects the foreign tail.
    match unsafe { device.queue().resume(tail) } {
        SubmitOutcome::Rejected {
            error: Error::WrongDevice,
            ..
        } => {}
        _ => panic!("foreign tail reached the backend"),
    }
}
#[test]
fn stale_surface_lease_cannot_enter_plan_present() {
    let device_id = DeviceId::allocate().unwrap();
    // SAFETY: the fixture pairs this native device with its newly allocated ID.
    let device = unsafe {
        Device::<NativeShared>::from_native(device_id, Arc::new(()), DeviceCapabilities::default())
    };
    let lane = LaneId {
        device: device_id,
        ordinal: 0,
    };
    // SAFETY: this test provides an ordered, device-local compiled plan.
    let plan = unsafe {
        SubmissionPlan::<NativeShared>::from_compiled(
            PlanId {
                device: device_id,
                ordinal: 2,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Present,
                waits: Vec::new(),
            }]),
        )
    };
    let current = Arc::new(AtomicU64::new(1));
    let image = SurfaceImage {
        device: device_id,
        surface: SurfaceId::allocate().unwrap(),
        generation: 1,
        stats: Arc::new(RhiStats::new()),
        native: TestLease {
            current: current.clone(),
            handed_off: false,
        },
    };
    current.store(2, Ordering::Release);
    let packet = SubmitPacket {
        steps: vec![BoundStep {
            step: plan.step(0),
            payload: StepPayload::Present(PresentBinding::new(image)),
        }],
        plan,
        completion: CompletionRequest::None,
    };
    // SAFETY: the test invokes submit only to exercise stale-lease validation.
    match unsafe { device.queue().submit(packet) } {
        SubmitOutcome::Rejected {
            error: Error::SurfaceOutdated,
            ..
        } => {}
        _ => panic!("stale surface lease reached the backend"),
    }
}
#[test]
fn immediate_validates_waits_and_lanes_before_execution() {
    let id = DeviceId::allocate().unwrap();
    let lane = LaneId {
        device: id,
        ordinal: 0,
    };
    let info = LaneInfo {
        id: lane,
        class: LaneClass::Graphics,
        routes: Vec::new(),
    };
    assert_eq!(info.route_to(lane), WaitRoute::SameQueue);
    assert_eq!(
        info.route_to(LaneId {
            device: id,
            ordinal: 1
        }),
        WaitRoute::Unsupported
    );
    let caps = DeviceCapabilities {
        lanes: Arc::from([info]),
        ..DeviceCapabilities::default()
    };
    // SAFETY: the fixture owns the native witness and its matching ID.
    let device = unsafe { Device::<GlLocal>::from_native(id, Rc::new(()), caps) };
    let plan_id = PlanId {
        device: id,
        ordinal: 19,
    };
    let scope = AccessScope {
        stages: StageMask::ALL_COMMANDS,
        accesses: AccessMask::ALL,
    };
    let make_plan = |second_lane, previous| {
        // SAFETY: malformed fixtures intentionally exercise Core validation only.
        unsafe {
            SubmissionPlan::from_compiled(
                plan_id,
                Arc::from([
                    StepDesc {
                        lane,
                        kind: StepKind::Commands,
                        waits: vec![],
                    },
                    StepDesc {
                        lane: second_lane,
                        kind: StepKind::Commands,
                        waits: vec![StepWait {
                            previous,
                            destination: scope,
                        }],
                    },
                ]),
            )
        }
    };
    let valid = make_plan(
        lane,
        StepId {
            plan: plan_id,
            index: 0,
        },
    );
    // SAFETY: mock session has no GPU resources and both steps are closed.
    let mut session = unsafe { device.immediate(valid, CompletionRequest::None) }.unwrap();
    for _ in 0..2 {
        let encoder = session.begin_commands().ok().unwrap();
        assert!(matches!(encoder.close(), ImmediateRunOutcome::Issued(_)));
    }
    assert!(matches!(
        session.finish(),
        ImmediateFinishOutcome::Issued(_)
    ));
    let forward = make_plan(
        lane,
        StepId {
            plan: plan_id,
            index: 1,
        },
    );
    // SAFETY: rejected before issuing native work.
    assert!(matches!(
        unsafe { device.immediate(forward, CompletionRequest::None) },
        Err(Error::InvalidState)
    ));
    let other_lane = make_plan(
        LaneId {
            device: id,
            ordinal: 1,
        },
        StepId {
            plan: plan_id,
            index: 0,
        },
    );
    // SAFETY: rejected before issuing native work.
    assert!(matches!(
        unsafe { device.immediate(other_lane, CompletionRequest::None) },
        Err(Error::WrongLane)
    ));
}
#[test]
fn dropped_immediate_session_reports_issued_prefix_to_backend() {
    ABANDONED_STEPS.with(|count| count.set(0));
    let device_id = DeviceId::allocate().unwrap();
    let lane = LaneId {
        device: device_id,
        ordinal: 0,
    };
    let caps = DeviceCapabilities {
        lanes: Arc::from([LaneInfo {
            id: lane,
            class: LaneClass::Graphics,
            routes: Vec::new(),
        }]),
        ..DeviceCapabilities::default()
    };
    // SAFETY: the fixture pairs this native device with its newly allocated ID.
    let device = unsafe { Device::<GlLocal>::from_native(device_id, Rc::new(()), caps) };
    // SAFETY: this test provides an ordered, device-local compiled plan.
    let plan = unsafe {
        SubmissionPlan::<GlLocal>::from_compiled(
            PlanId {
                device: device_id,
                ordinal: 3,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: Vec::new(),
            }]),
        )
    };
    // SAFETY: the fixture plan belongs to this immediate-only device.
    let mut session = unsafe { device.immediate(plan, CompletionRequest::None).unwrap() };
    let encoder = match session.begin_commands() {
        Ok(encoder) => encoder,
        Err(_) => panic!("mock command slot could not begin"),
    };
    match encoder.close() {
        ImmediateRunOutcome::Issued(_) => {}
        _ => panic!("mock command slot did not issue"),
    }
    drop(session);
    ABANDONED_STEPS.with(|count| assert_eq!(count.get(), 1));
}
