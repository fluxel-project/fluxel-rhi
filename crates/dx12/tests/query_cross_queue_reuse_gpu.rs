//! Completed query slots can move between direct and compute queues.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;
#[test]
fn timestamp_slot_moves_only_after_completion() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let direct = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let compute = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Compute)
        .unwrap()
        .id;
    let q = d.queries(QueryKind::Timestamp).unwrap();
    let set = q
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 1,
        })
        .unwrap();
    for (i, lane) in [direct, compute, direct].into_iter().enumerate() {
        let dst = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 8,
                usage: BufferUsage::QUERY_RESOLVE | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .unwrap();
        let mut e = d.encoder(lane).unwrap();
        q.reset(&mut e, &set, 0..1).unwrap();
        q.timestamp(&mut e, &set, 0).unwrap();
        q.resolve(&mut e, &set, 0..1, &dst, 0).unwrap();
        let ticket = e.readback_buffer(&dst, 0..8).unwrap();
        // SAFETY: this single-use plan uses the encoder's device and lane.
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: d.id(),
                    ordinal: 88000 + i as u64,
                },
                Arc::from([StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                }]),
            )
        };
        let step = plan.step(0);
        // SAFETY: recorded work is submitted once on its matching lane.
        let outcome = unsafe {
            d.queue().submit(SubmitPacket {
                plan,
                steps: vec![BoundStep {
                    step,
                    payload: StepPayload::Commands(vec![e.finish().unwrap()]),
                }],
                completion: CompletionRequest::End,
            })
        };
        let point = match outcome {
            SubmitOutcome::Accepted(receipt) => {
                receipt.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("query submission rejected at iteration {i}"),
        };
        block(d.wait_completion(&point)).unwrap();
        // SAFETY: completion covers the readback ticket.
        let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
        assert_ne!(u64::from_ne_bytes(bytes[..8].try_into().unwrap()), 0);
    }
    // A completed older direct point cannot authorize a new direct operation
    // followed by an unordered compute operation in the same packet.
    let mut first = d.encoder(direct).unwrap();
    let mut second = d.encoder(compute).unwrap();
    q.timestamp(&mut first, &set, 0).unwrap();
    q.timestamp(&mut second, &set, 0).unwrap();
    // SAFETY: the two command lists have the lanes declared in this plan.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 88005,
            },
            Arc::from([
                StepDesc {
                    lane: direct,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane: compute,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
            ]),
        )
    };
    let a = plan.step(0);
    let b = plan.step(1);
    // SAFETY: ownership of both recordings is transferred exactly once.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![
                BoundStep {
                    step: a,
                    payload: StepPayload::Commands(vec![first.finish().unwrap()]),
                },
                BoundStep {
                    step: b,
                    payload: StepPayload::Commands(vec![second.finish().unwrap()]),
                },
            ],
            completion: CompletionRequest::End,
        })
    };
    assert!(matches!(
        outcome,
        SubmitOutcome::Rejected {
            error: Error::Unsupported(_),
            ..
        }
    ));
}

#[test]
fn timestamp_slot_moves_within_packet_after_gpu_wait() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let direct = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let compute = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Compute)
        .unwrap()
        .id;
    let q = d.queries(QueryKind::Timestamp).unwrap();
    let set = q
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 1,
        })
        .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 8,
            usage: BufferUsage::QUERY_RESOLVE | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut first = d.encoder(direct).unwrap();
    q.reset(&mut first, &set, 0..1).unwrap();
    q.timestamp(&mut first, &set, 0).unwrap();
    let mut second = d.encoder(compute).unwrap();
    q.reset(&mut second, &set, 0..1).unwrap();
    q.timestamp(&mut second, &set, 0).unwrap();
    q.resolve(&mut second, &set, 0..1, &dst, 0).unwrap();
    let ticket = second.readback_buffer(&dst, 0..8).unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 88010,
    };
    let producer = StepId { plan: id, index: 0 };
    // SAFETY: the declared GPU wait orders the query slot across both lanes.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([
                StepDesc {
                    lane: direct,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane: compute,
                    kind: StepKind::Commands,
                    waits: vec![StepWait {
                        previous: producer,
                        destination: AccessScope {
                            stages: StageMask::COPY,
                            accesses: AccessMask::COPY_WRITE,
                        },
                    }],
                },
                StepDesc {
                    lane: compute,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
            ]),
        )
    };
    let a = plan.step(0);
    let b = plan.step(1);
    let c = plan.step(2);
    // SAFETY: both recordings are transferred once to their declared lanes.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![
                BoundStep {
                    step: a,
                    payload: StepPayload::Commands(vec![first.finish().unwrap()]),
                },
                BoundStep {
                    step: b,
                    payload: StepPayload::Commands(vec![]),
                },
                BoundStep {
                    step: c,
                    payload: StepPayload::Commands(vec![second.finish().unwrap()]),
                },
            ],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("ordered query submission rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    // SAFETY: the end checkpoint covers the readback ticket.
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    assert_ne!(u64::from_ne_bytes(bytes[..8].try_into().unwrap()), 0);
}
