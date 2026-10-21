//! Multi-queue submissions must enforce the same packet integrity as a queue.
use super::{type_contract_tests::NativeShared, *};
use std::cell::Cell;

thread_local! {
    static SUBMIT_CALLS: Cell<usize> = const { Cell::new(0) };
}

impl MultiQueueBackend for NativeShared {
    type NativeGpuSignal = Arc<()>;

    unsafe fn submit_multi(
        _: &Arc<()>,
        packet: MultiQueuePacket<Self>,
    ) -> MultiQueueSubmitOutcome<Self> {
        SUBMIT_CALLS.with(|calls| calls.set(calls.get() + 1));
        MultiQueueSubmitOutcome::Rejected {
            error: Error::Unsupported("mock"),
            packet,
        }
    }
}

fn multi_fixture() -> (
    Device<NativeShared>,
    SubmissionPlan<NativeShared>,
    LaneId,
    LaneId,
) {
    let id = DeviceId::allocate().unwrap();
    let producer_lane = LaneId {
        device: id,
        ordinal: 0,
    };
    let consumer_lane = LaneId {
        device: id,
        ordinal: 1,
    };
    let caps = DeviceCapabilities {
        multi_queue: true,
        lanes: Arc::from([
            LaneInfo {
                id: producer_lane,
                class: LaneClass::Graphics,
                routes: vec![(consumer_lane, WaitRoute::NativeGpuWait)],
            },
            LaneInfo {
                id: consumer_lane,
                class: LaneClass::Graphics,
                routes: vec![],
            },
        ]),
        ..Default::default()
    };
    // SAFETY: identity and native mock fixture belong together.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    // SAFETY: the mock accepts this ordered device-local layout.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: id,
                ordinal: 31,
            },
            Arc::from([
                StepDesc {
                    lane: producer_lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane: consumer_lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
            ]),
        )
    };
    (device, plan, producer_lane, consumer_lane)
}

fn work_for(plan: &SubmissionPlan<NativeShared>) -> SubmitPacket<NativeShared> {
    let steps = plan
        .steps()
        .iter()
        .enumerate()
        .map(|(index, desc)| BoundStep {
            step: plan.step(index),
            payload: StepPayload::Commands(vec![RecordedWork {
                device: plan.id().device,
                lane: desc.lane,
                native: (),
            }]),
        })
        .collect();
    SubmitPacket {
        plan: plan.clone(),
        steps,
        completion: CompletionRequest::None,
    }
}

fn scope() -> AccessScope {
    AccessScope {
        stages: StageMask::ALL_COMMANDS,
        accesses: AccessMask::ALL,
    }
}

fn token(
    device: DeviceId,
    producer: StepId,
    producer_lane: LaneId,
    consumer: StepId,
    consumer_lane: LaneId,
    native: Arc<()>,
) -> GpuSignalPoint<NativeShared> {
    // SAFETY: the fixture has no native queue effects; metadata is intentionally
    // constructed for Core's static validation tests.
    unsafe {
        GpuSignalPoint::from_native(
            device,
            producer,
            producer_lane,
            consumer,
            consumer_lane,
            native,
        )
    }
}

#[test]
fn signal_and_wait_metadata_are_checked_before_backend() {
    let (device, plan, producer_lane, consumer_lane) = multi_fixture();
    let id = device.id();
    let foreign_id = DeviceId::allocate().unwrap();
    let foreign_plan = PlanId {
        device: foreign_id,
        ordinal: 32,
    };
    let other_plan = PlanId {
        device: id,
        ordinal: 32,
    };
    let unsupported_lane = LaneId {
        device: id,
        ordinal: 2,
    };
    let foreign_lane = LaneId {
        device: foreign_id,
        ordinal: 1,
    };
    let other_step = StepId {
        plan: other_plan,
        index: 0,
    };
    let foreign_step = StepId {
        plan: foreign_plan,
        index: 0,
    };
    SUBMIT_CALLS.with(|calls| calls.set(0));
    let cases = [
        (
            GpuSignalRequest {
                producer: foreign_step,
                consumer: plan.step(1),
                consumer_lane,
            },
            Error::WrongDevice,
        ),
        (
            GpuSignalRequest {
                producer: other_step,
                consumer: plan.step(1),
                consumer_lane,
            },
            Error::InvalidState,
        ),
        (
            GpuSignalRequest {
                producer: plan.step(2),
                consumer: plan.step(1),
                consumer_lane,
            },
            Error::InvalidState,
        ),
        (
            GpuSignalRequest {
                producer: plan.step(0),
                consumer: foreign_step,
                consumer_lane,
            },
            Error::WrongDevice,
        ),
        (
            GpuSignalRequest {
                producer: plan.step(0),
                consumer: plan.step(1),
                consumer_lane: foreign_lane,
            },
            Error::WrongDevice,
        ),
        (
            GpuSignalRequest {
                producer: plan.step(0),
                consumer: plan.step(1),
                consumer_lane: unsupported_lane,
            },
            Error::WrongLane,
        ),
    ];
    for (signal, expected) in cases {
        let packet = MultiQueuePacket {
            work: work_for(&plan),
            signals: vec![signal],
            waits: vec![],
        };
        // SAFETY: the mock never issues native work.
        match unsafe { device.multi_queue().unwrap().submit(packet) } {
            MultiQueueSubmitOutcome::Rejected { error, packet } => {
                assert_eq!(
                    std::mem::discriminant(&error),
                    std::mem::discriminant(&expected)
                );
                assert_eq!(packet.signals.len(), 1);
                assert_eq!(packet.work.steps.len(), 2);
            }
            _ => panic!("invalid signal reached backend"),
        }
    }
    let wait_cases = [
        (
            token(
                foreign_id,
                plan.step(0),
                producer_lane,
                plan.step(1),
                consumer_lane,
                Arc::new(()),
            ),
            Error::WrongDevice,
        ),
        (
            token(
                id,
                foreign_step,
                producer_lane,
                plan.step(1),
                consumer_lane,
                Arc::new(()),
            ),
            Error::WrongDevice,
        ),
        (
            token(
                id,
                plan.step(0),
                foreign_lane,
                plan.step(1),
                consumer_lane,
                Arc::new(()),
            ),
            Error::WrongDevice,
        ),
        (
            token(
                id,
                plan.step(0),
                producer_lane,
                foreign_step,
                consumer_lane,
                Arc::new(()),
            ),
            Error::WrongDevice,
        ),
        (
            token(
                id,
                plan.step(0),
                producer_lane,
                other_step,
                consumer_lane,
                Arc::new(()),
            ),
            Error::InvalidState,
        ),
        (
            token(
                id,
                plan.step(0),
                producer_lane,
                plan.step(2),
                consumer_lane,
                Arc::new(()),
            ),
            Error::InvalidState,
        ),
        (
            token(
                id,
                plan.step(0),
                producer_lane,
                plan.step(1),
                producer_lane,
                Arc::new(()),
            ),
            Error::WrongLane,
        ),
        (
            token(
                id,
                plan.step(0),
                unsupported_lane,
                plan.step(1),
                consumer_lane,
                Arc::new(()),
            ),
            Error::WrongLane,
        ),
    ];
    for (signal, expected) in wait_cases {
        let packet = MultiQueuePacket {
            work: work_for(&plan),
            signals: vec![],
            waits: vec![GpuWait {
                signal,
                destination: scope(),
            }],
        };
        // SAFETY: the mock never issues native work.
        match unsafe { device.multi_queue().unwrap().submit(packet) } {
            MultiQueueSubmitOutcome::Rejected { error, packet } => {
                assert_eq!(
                    std::mem::discriminant(&error),
                    std::mem::discriminant(&expected)
                );
                assert_eq!(packet.waits.len(), 1);
                assert_eq!(packet.work.steps.len(), 2);
            }
            _ => panic!("invalid wait reached backend"),
        }
    }
    SUBMIT_CALLS.with(|calls| assert_eq!(calls.get(), 0));
}

#[test]
fn reused_plan_routes_distinct_owned_tokens_to_backend() {
    let (device, plan, producer_lane, consumer_lane) = multi_fixture();
    let first_native = Arc::new(());
    let second_native = Arc::new(());
    let first_weak = Arc::downgrade(&first_native);
    let second_weak = Arc::downgrade(&second_native);
    assert!(!Arc::ptr_eq(&first_native, &second_native));
    let make_packet = |native| MultiQueuePacket {
        work: work_for(&plan),
        signals: vec![],
        waits: vec![GpuWait {
            signal: token(
                device.id(),
                plan.step(0),
                producer_lane,
                plan.step(1),
                consumer_lane,
                native,
            ),
            destination: scope(),
        }],
    };
    SUBMIT_CALLS.with(|calls| calls.set(0));
    let signal_packet = MultiQueuePacket {
        work: work_for(&plan),
        signals: vec![GpuSignalRequest {
            producer: plan.step(0),
            consumer: plan.step(1),
            consumer_lane,
        }],
        waits: vec![],
    };
    // SAFETY: the mock only checks static identities and cleanly rejects.
    assert!(matches!(
        unsafe { device.multi_queue().unwrap().submit(signal_packet) },
        MultiQueueSubmitOutcome::Rejected {
            error: Error::Unsupported("mock"),
            ..
        }
    ));
    let mut packets = Vec::new();
    for native in [first_native, second_native] {
        // SAFETY: distinct owned fixture tokens are paired by this caller with
        // their intended mock consumer executions; mock has no native queue.
        match unsafe { device.multi_queue().unwrap().submit(make_packet(native)) } {
            MultiQueueSubmitOutcome::Rejected {
                error: Error::Unsupported("mock"),
                packet,
            } => packets.push(packet),
            _ => panic!("valid metadata did not reach the mock backend"),
        }
    }
    SUBMIT_CALLS.with(|calls| assert_eq!(calls.get(), 3));
    assert_eq!(packets[0].waits[0].signal.producer(), plan.step(0));
    assert_eq!(packets[0].waits[0].signal.consumer(), plan.step(1));
    drop(packets.remove(0));
    assert!(first_weak.upgrade().is_none());
    assert!(second_weak.upgrade().is_some());
    drop(packets);
    assert!(second_weak.upgrade().is_none());
}

#[test]
fn invalid_work_is_rejected_before_multi_queue_backend() {
    let id = DeviceId::allocate().unwrap();
    let caps = DeviceCapabilities {
        multi_queue: true,
        ..Default::default()
    };
    // SAFETY: the fixture pairs the native device with its allocated identity.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    let lane = LaneId {
        device: id,
        ordinal: 0,
    };
    // SAFETY: the plan is a realizable one-step, device-local layout.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: id,
                ordinal: 1,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: Vec::new(),
            }]),
        )
    };
    let step = || BoundStep {
        step: plan.step(0),
        payload: StepPayload::Commands(vec![RecordedWork {
            device: id,
            lane,
            native: (),
        }]),
    };
    let invalid_packets = [
        (Vec::new(), CompletionRequest::None),
        (
            vec![BoundStep {
                step: plan.step(1),
                ..step()
            }],
            CompletionRequest::None,
        ),
        (
            vec![BoundStep {
                payload: StepPayload::SparseBind(Vec::new()),
                ..step()
            }],
            CompletionRequest::None,
        ),
        (
            vec![step()],
            CompletionRequest::Steps(Arc::from([plan.step(1)])),
        ),
    ];
    SUBMIT_CALLS.with(|calls| calls.set(0));
    for (steps, completion) in invalid_packets {
        let packet = MultiQueuePacket {
            work: SubmitPacket {
                plan: plan.clone(),
                steps,
                completion,
            },
            signals: Vec::new(),
            waits: Vec::new(),
        };
        // SAFETY: invalid binding is deliberately supplied to exercise Core's clean rejection.
        match unsafe { device.multi_queue().unwrap().submit(packet) } {
            MultiQueueSubmitOutcome::Rejected {
                error: Error::InvalidState,
                packet,
            } => {
                assert_eq!(packet.work.plan.id(), plan.id());
            }
            _ => panic!("invalid work was not rejected by Core"),
        }
    }
    let wrong_lane = LaneId {
        device: id,
        ordinal: 1,
    };
    let packet = MultiQueuePacket {
        work: SubmitPacket {
            plan: plan.clone(),
            steps: vec![BoundStep {
                step: plan.step(0),
                payload: StepPayload::Commands(vec![RecordedWork {
                    device: id,
                    lane: wrong_lane,
                    native: (),
                }]),
            }],
            completion: CompletionRequest::None,
        },
        signals: Vec::new(),
        waits: Vec::new(),
    };
    // SAFETY: the mock never issues work; the invalid lane exercises the Core guard.
    assert!(matches!(
        unsafe { device.multi_queue().unwrap().submit(packet) },
        MultiQueueSubmitOutcome::Rejected {
            error: Error::WrongLane,
            ..
        }
    ));
    SUBMIT_CALLS.with(|calls| assert_eq!(calls.get(), 0));

    let foreign_id = DeviceId::allocate().unwrap();
    let foreign_work = || SubmitPacket {
        plan: plan.clone(),
        steps: vec![BoundStep {
            step: plan.step(0),
            payload: StepPayload::Commands(vec![RecordedWork {
                device: foreign_id,
                lane: LaneId {
                    device: foreign_id,
                    ordinal: 0,
                },
                native: (),
            }]),
        }],
        completion: CompletionRequest::None,
    };
    // SAFETY: both entry points must reject foreign work before the mock SPI.
    assert!(matches!(
        unsafe { device.queue().submit(foreign_work()) },
        SubmitOutcome::Rejected {
            error: Error::WrongDevice,
            ..
        }
    ));
    // SAFETY: the foreign packet exercises Core validation, not native execution.
    assert!(matches!(
        unsafe {
            device.multi_queue().unwrap().submit(MultiQueuePacket {
                work: foreign_work(),
                signals: Vec::new(),
                waits: Vec::new(),
            })
        },
        MultiQueueSubmitOutcome::Rejected {
            error: Error::WrongDevice,
            ..
        }
    ));
    SUBMIT_CALLS.with(|calls| assert_eq!(calls.get(), 0));

    let packet = MultiQueuePacket {
        work: SubmitPacket {
            plan: plan.clone(),
            steps: vec![step()],
            completion: CompletionRequest::None,
        },
        signals: Vec::new(),
        waits: Vec::new(),
    };
    // SAFETY: the mock rejects without issuing native work.
    assert!(matches!(
        unsafe { device.multi_queue().unwrap().submit(packet) },
        MultiQueueSubmitOutcome::Rejected {
            error: Error::Unsupported("mock"),
            ..
        }
    ));
    SUBMIT_CALLS.with(|calls| assert_eq!(calls.get(), 1));
}
