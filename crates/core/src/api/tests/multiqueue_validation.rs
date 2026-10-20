//! Multi-queue submissions must enforce the same packet integrity as a queue.
use super::{type_contract_tests::NativeShared, *};
use std::cell::Cell;

thread_local! {
    static SUBMIT_CALLS: Cell<usize> = const { Cell::new(0) };
}

impl MultiQueueBackend for NativeShared {
    type NativeGpuSignal = ();

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
