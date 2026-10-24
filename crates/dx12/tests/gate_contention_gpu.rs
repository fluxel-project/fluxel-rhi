//! Real-DX12 submission-gate contention diagnostics.
//!
//! This is a measurement fixture, not a scheduling policy test. Each worker
//! records its own command lists and plan before the start barrier, then only
//! contends at native submission. It reports aggregate gate wait/hold time and
//! wall time for one and four submitters at two packet sizes.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::{diagnostics, Dx12, Dx12Instance};
use std::{
    future::Future,
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

fn block<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => thread::sleep(Duration::from_millis(1)),
        }
    }
}

#[derive(Clone, Copy)]
struct Measurement {
    wall: Duration,
    wait_ns: u64,
    hold_ns: u64,
}

fn run(
    device: Device<Dx12>,
    lane: LaneId,
    workers: usize,
    records_per_submit: usize,
    ordinal_base: u64,
) -> Measurement {
    let ready = Arc::new(Barrier::new(workers + 1));
    let start = Arc::new(Barrier::new(workers + 1));
    let before = diagnostics::snapshot();
    let mut threads = Vec::with_capacity(workers);
    for worker in 0..workers {
        let device = device.clone();
        let ready = ready.clone();
        let start = start.clone();
        threads.push(thread::spawn(move || {
            // Each worker constructs its own plan and recordings before the
            // measured barrier; no encoder or plan is shared by submitters.
            let plan = unsafe {
                SubmissionPlan::from_compiled(
                    PlanId {
                        device: device.id(),
                        ordinal: ordinal_base + worker as u64,
                    },
                    Arc::from([StepDesc {
                        lane,
                        kind: StepKind::Commands,
                        waits: vec![],
                    }]),
                )
            };
            let mut work = Vec::with_capacity(records_per_submit);
            for _ in 0..records_per_submit {
                work.push(device.encoder(lane).unwrap().finish().unwrap());
            }
            ready.wait();
            start.wait();
            let point = match unsafe {
                device.queue().submit(SubmitPacket {
                    plan: plan.clone(),
                    steps: vec![BoundStep {
                        step: plan.step(0),
                        payload: StepPayload::Commands(work),
                    }],
                    completion: CompletionRequest::End,
                })
            } {
                SubmitOutcome::Accepted(receipt) => {
                    receipt
                        .checkpoints
                        .into_iter()
                        .find(|checkpoint| {
                            checkpoint.covers.lane_frontiers == vec![(lane, plan.step(0))]
                        })
                        .expect("single-lane completion checkpoint")
                        .point
                }
                _ => panic!("contention submission rejected"),
            };
            block(device.wait_completion(&point)).unwrap();
        }));
    }
    ready.wait();
    let wall_started = Instant::now();
    start.wait();
    for thread in threads {
        thread.join().unwrap();
    }
    block(device.wait_idle()).unwrap();
    let after = diagnostics::snapshot();
    Measurement {
        wall: wall_started.elapsed(),
        wait_ns: after.submission_gate_wait_ns - before.submission_gate_wait_ns,
        hold_ns: after.submission_gate_hold_ns - before.submission_gate_hold_ns,
    }
}

#[test]
fn reports_query_submission_gate_contention_without_queries() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .expect("DX12 direct lane")
        .id;
    for records_per_submit in [1, 64] {
        for workers in [1, 4] {
            let measurement = run(
                device.clone(),
                lane,
                workers,
                records_per_submit,
                100_000 + (records_per_submit * 10 + workers) as u64,
            );
            println!(
                "gate-contention workers={workers} records={records_per_submit} submitwall_ns={} gatewait_ns={} gatehold_ns={} completed_and_drained=true",
                measurement.wall.as_nanos(), measurement.wait_ns, measurement.hold_ns,
            );
        }
    }
}
