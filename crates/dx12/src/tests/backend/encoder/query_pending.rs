//! Cross-packet waits while the producer queue is deliberately blocked.
use super::*;
use crate::backend::Dx12Instance;
use fluxel_rhi_core::api::*;
use std::{future::Future, sync::Arc};

fn block<F: Future>(f: F) -> F::Output {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut f = std::pin::pin!(f);
    loop {
        if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

struct Release(ID3D12Fence);
impl Drop for Release {
    fn drop(&mut self) {
        // SAFETY: release the private test fence even if an assertion fails.
        unsafe {
            let _ = self.0.Signal(1);
        }
    }
}

#[test]
fn pending_query_slot_moves_after_native_wait() {
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
    let make_plan = |ordinal, lane| unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let producer = make_plan(88021, direct);
    let consumer = make_plan(88022, compute);
    let ps = producer.step(0);
    let cs = consumer.step(0);
    // SAFETY: this backend-owned queue is deliberately blocked by a private
    // fence; the guard always releases it before waiting for completion.
    let native = unsafe { d.native_ref() };
    let gate = unsafe {
        native
            .0
            .native
            .CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE)
    }
    .unwrap();
    let other = unsafe {
        native
            .0
            .native
            .CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE)
    }
    .unwrap();
    assert!(!fence_wait_covers(&gate, 2, &[(gate.clone(), 1)]));
    assert!(!fence_wait_covers(&gate, 1, &[(other, 2)]));
    assert!(fence_wait_covers(&gate, 1, &[(gate.clone(), 2)]));
    let guard = Release(gate.clone());
    unsafe { native.0.queue.Wait(&gate, 1) }.unwrap();
    let domain = d.multi_queue().unwrap();
    let sent = unsafe {
        domain.submit(MultiQueuePacket {
            work: SubmitPacket {
                plan: producer,
                steps: vec![BoundStep {
                    step: ps,
                    payload: StepPayload::Commands(vec![first.finish().unwrap()]),
                }],
                completion: CompletionRequest::None,
            },
            signals: vec![GpuSignalRequest {
                producer: ps,
                consumer: cs,
                consumer_lane: compute,
            }],
            waits: vec![],
        })
    };
    let signal = match sent {
        MultiQueueSubmitOutcome::Accepted(r) => r.signals.into_iter().next().unwrap(),
        _ => panic!("producer rejected"),
    };
    assert_eq!(unsafe { gate.GetCompletedValue() }, 0);
    let mut unordered = d.encoder(compute).unwrap();
    q.timestamp(&mut unordered, &set, 0).unwrap();
    // SAFETY: this intentionally lacks the producer dependency; rejection
    // happens during preflight before native submission.
    let rejected = unsafe {
        domain.submit(MultiQueuePacket {
            work: SubmitPacket {
                plan: consumer.clone(),
                steps: vec![BoundStep {
                    step: cs,
                    payload: StepPayload::Commands(vec![unordered.finish().unwrap()]),
                }],
                completion: CompletionRequest::None,
            },
            signals: vec![],
            waits: vec![],
        })
    };
    assert!(matches!(
        rejected,
        MultiQueueSubmitOutcome::Rejected {
            error: Error::Unsupported(_),
            ..
        }
    ));
    let received = unsafe {
        domain.submit(MultiQueuePacket {
            work: SubmitPacket {
                plan: consumer,
                steps: vec![BoundStep {
                    step: cs,
                    payload: StepPayload::Commands(vec![second.finish().unwrap()]),
                }],
                completion: CompletionRequest::End,
            },
            signals: vec![],
            waits: vec![GpuWait {
                signal,
                destination: AccessScope {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
            }],
        })
    };
    let point = match received {
        MultiQueueSubmitOutcome::Accepted(r) => {
            r.work.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("consumer rejected"),
    };
    drop(guard);
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    assert_ne!(u64::from_ne_bytes(bytes[..8].try_into().unwrap()), 0);
}
