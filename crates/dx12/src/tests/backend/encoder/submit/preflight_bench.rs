//! Hardware-backed CPU diagnostic for inherited DX12 native wait frontiers.
//!
//! This is deliberately ignored: it measures only `preflight::packet`, after
//! a real producer has issued one GPU signal.  The consumer packet is never
//! handed to a queue, so its result excludes command execution, queue locking,
//! and driver submission time.

use super::*;
use fluxel_rhi_core::api::{
    self, AccessMask, AccessScope, BoundStep, CompletionRequest, GpuSignalRequest, GpuWait,
    LaneClass, LaneId, MultiQueueDomain, MultiQueuePacket, MultiQueueSubmitOutcome, PlanId,
    StepDesc, StepId, StepKind, StepPayload, SubmissionPlan, SubmitPacket,
};
use std::{future::Future, hint::black_box, sync::Arc, time::Instant};

fn block<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

fn producer_signal(
    device: &api::Device<crate::Dx12>,
    domain: &MultiQueueDomain<'_, crate::Dx12>,
    producer_lane: LaneId,
    consumer: StepId,
    consumer_lane: LaneId,
    ordinal: u64,
) -> api::GpuSignalPoint<crate::Dx12> {
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal,
            },
            Arc::from([StepDesc {
                lane: producer_lane,
                kind: StepKind::Commands,
                waits: Vec::new(),
            }]),
        )
    };
    let producer = plan.step(0);
    let encoder = device.encoder(producer_lane).unwrap();
    match unsafe {
        domain.submit(MultiQueuePacket {
            work: SubmitPacket {
                plan,
                steps: vec![BoundStep {
                    step: producer,
                    payload: StepPayload::Commands(vec![encoder.finish().unwrap()]),
                }],
                completion: CompletionRequest::None,
            },
            signals: vec![GpuSignalRequest {
                producer,
                consumer,
                consumer_lane,
            }],
            waits: Vec::new(),
        })
    } {
        MultiQueueSubmitOutcome::Accepted(receipt) => receipt.signals.into_iter().next().unwrap(),
        _ => panic!("diagnostic producer was rejected"),
    }
}

fn consumer_packet(
    plan: SubmissionPlan<crate::Dx12>,
    waits: Vec<GpuWait<crate::Dx12>>,
) -> MultiQueuePacket<crate::Dx12> {
    let plan_id = plan.id();
    let step_count = plan.steps().len();
    MultiQueuePacket {
        work: SubmitPacket {
            plan,
            steps: (0..step_count)
                .map(|index| BoundStep {
                    step: StepId {
                        plan: plan_id,
                        index: index as u32,
                    },
                    payload: StepPayload::Commands(Vec::new()),
                })
                .collect(),
            completion: CompletionRequest::None,
        },
        signals: Vec::new(),
        waits,
    }
}

fn measure(
    native: &crate::backend::Dx12Device,
    label: &str,
    candidate: &MultiQueuePacket<crate::Dx12>,
    count: usize,
    iterations: u32,
) {
    for _ in 0..10 {
        black_box(packet(native, candidate).unwrap());
    }
    let start = Instant::now();
    let mut frontier_fences = 0_usize;
    for _ in 0..iterations {
        let prepared = packet(native, candidate).unwrap();
        frontier_fences += prepared.external_waits.last().unwrap().len();
        black_box(prepared.coverage.len());
    }
    let elapsed = start.elapsed();
    println!(
        "dx12-preflight,case={label},steps={count},wait_fences={},iterations={iterations},elapsed_ns={},ns_per_packet={:.1}",
        frontier_fences / iterations as usize,
        elapsed.as_nanos(),
        elapsed.as_nanos() as f64 / f64::from(iterations),
    );
}

/// Measures the inherited one-fence case at the sizes named in review3.
///
/// The backend owns exactly direct and compute queue fences.  Therefore a
/// valid same-device packet's native wait frontier has at most two distinct
/// fences; this diagnostic measures the non-empty inherited frontier rather
/// than synthesizing invalid arbitrary fence identities.
#[test]
#[ignore = "CPU diagnostic; run only when investigating DX12 preflight cost"]
fn inherited_wait_frontier_cpu_cost() {
    let device = block(crate::Dx12Instance::new().unwrap().open_default()).unwrap();
    let direct = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let compute = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Compute)
        .unwrap()
        .id;
    let domain = device.multi_queue().unwrap();
    let native = unsafe { device.native_ref() };

    for (ordinal, count, iterations) in [(700_001, 1_000_usize, 1_000_u32), (700_002, 10_000, 100)]
    {
        let consumer_plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: device.id(),
                    ordinal,
                },
                (0..count)
                    .map(|_| StepDesc {
                        lane: direct,
                        kind: StepKind::Commands,
                        waits: Vec::new(),
                    })
                    .collect::<Vec<_>>()
                    .into(),
            )
        };
        let consumer = consumer_plan.step(0);
        let scope = AccessScope {
            stages: api::StageMask::ALL_COMMANDS,
            accesses: AccessMask::ALL,
        };
        let one_fence = consumer_packet(
            consumer_plan,
            vec![GpuWait {
                signal: producer_signal(
                    &device,
                    &domain,
                    compute,
                    consumer,
                    direct,
                    ordinal + 100_000,
                ),
                destination: scope,
            }],
        );
        measure(native, "one-fence", &one_fence, count, iterations);

        let two_fence_plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: device.id(),
                    ordinal: ordinal + 10_000,
                },
                (0..count)
                    .map(|_| StepDesc {
                        lane: direct,
                        kind: StepKind::Commands,
                        waits: Vec::new(),
                    })
                    .collect::<Vec<_>>()
                    .into(),
            )
        };
        let two_fence_consumer = two_fence_plan.step(0);
        let two_fence = consumer_packet(
            two_fence_plan,
            vec![
                GpuWait {
                    signal: producer_signal(
                        &device,
                        &domain,
                        compute,
                        two_fence_consumer,
                        direct,
                        ordinal + 110_000,
                    ),
                    destination: scope,
                },
                // This same-queue token is valid under the published
                // `WaitRoute::SameQueue` contract and exercises the direct
                // fence alongside the compute fence without fabricating a
                // native signal identity.
                GpuWait {
                    signal: producer_signal(
                        &device,
                        &domain,
                        direct,
                        two_fence_consumer,
                        direct,
                        ordinal + 120_000,
                    ),
                    destination: scope,
                },
            ],
        );
        measure(native, "two-fence-max", &two_fence, count, iterations);
    }
    block(device.wait_idle()).unwrap();
}
