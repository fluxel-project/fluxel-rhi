use crate::backend::Dx12Instance;
use fluxel_rhi_core::api::*;
use std::{
    future::Future,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
use windows::Win32::Graphics::Direct3D12::{D3D12_FENCE_FLAG_NONE, ID3D12Fence};

/// Releases the private native gate even while an assertion unwinds, so a
/// failed retention assertion cannot strand either queue in this process.
struct GateRelease {
    fence: ID3D12Fence,
    released: bool,
}

impl GateRelease {
    fn new(fence: ID3D12Fence) -> Self {
        Self {
            fence,
            released: false,
        }
    }

    fn release(&mut self) {
        if !std::mem::replace(&mut self.released, true) {
            // SAFETY: this fixture owns the sole CPU producer signal.
            unsafe { self.fence.Signal(1) }.expect("release retirement test gate");
        }
    }
}

impl Drop for GateRelease {
    fn drop(&mut self) {
        if !std::mem::replace(&mut self.released, true) {
            // SAFETY: best-effort unwinding cleanup for the same signal.
            let _ = unsafe { self.fence.Signal(1) };
        }
    }
}

fn hold_native_queues(device: &Device<crate::backend::Dx12>, compute: bool) -> GateRelease {
    // SAFETY: the test owns no concurrent raw queue access; the device
    // keeps these native interfaces valid through the two Wait calls.
    let native = unsafe { device.native_ref() };
    let fence = unsafe {
        native
            .0
            .native
            .CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE)
    }
    .expect("create private retirement test fence");
    unsafe { native.0.queue.Wait(&fence, 1) }.expect("hold direct queue");
    if compute {
        unsafe { native.0.multi_queue.queue.Wait(&fence, 1) }.expect("hold compute queue");
    }
    GateRelease::new(fence)
}

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

#[test]
fn none_submission_retires_direct_and_compute_after_external_device_drop() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let mut gate = hold_native_queues(&device, true);
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
    let direct_buffer = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 4,
            usage: BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let compute_buffer = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 4,
            usage: BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let direct_weak = Arc::downgrade(&unsafe { direct_buffer.native_ref() }.mapped);
    let compute_weak = Arc::downgrade(&unsafe { compute_buffer.native_ref() }.mapped);
    let state_weak = Arc::downgrade(&unsafe { device.native_ref() }.0);
    let mut direct_encoder = device.encoder(direct).unwrap();
    direct_encoder
        .write_buffer(&direct_buffer, 0, &[1, 2, 3, 4])
        .unwrap();
    let mut compute_encoder = device.encoder(compute).unwrap();
    compute_encoder
        .write_buffer(&compute_buffer, 0, &[5, 6, 7, 8])
        .unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 99_941,
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
    let first = plan.step(0);
    let second = plan.step(1);
    // SAFETY: independent one-shot lists use their declared lanes only.
    let receipt = unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![
                BoundStep {
                    step: first,
                    payload: StepPayload::Commands(vec![direct_encoder.finish().unwrap()]),
                },
                BoundStep {
                    step: second,
                    payload: StepPayload::Commands(vec![compute_encoder.finish().unwrap()]),
                },
            ],
            completion: CompletionRequest::None,
        })
    };
    assert!(matches!(receipt, SubmitOutcome::Accepted(_)));
    drop(receipt);
    drop(direct_buffer);
    drop(compute_buffer);
    drop(device);

    // The queues cannot pass their terminal signals yet, so this proves
    // retention comes from the autonomous callback/in-flight stores, not
    // from a fast synchronous `GetCompletedValue` path.
    assert!(state_weak.upgrade().is_some());
    assert!(direct_weak.upgrade().is_some());
    assert!(compute_weak.upgrade().is_some());
    gate.release();
    drop(gate);

    let deadline = Instant::now() + Duration::from_secs(5);
    while state_weak.upgrade().is_some()
        || direct_weak.upgrade().is_some()
        || compute_weak.upgrade().is_some()
    {
        assert!(
            Instant::now() < deadline,
            "native fence retirement did not break retention"
        );
        thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn submitted_default_pool_lease_waits_for_direct_fence_retirement() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let mut gate = hold_native_queues(&device, false);
    let direct = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let desc = BufferDesc {
        label: None,
        size: 4,
        usage: BufferUsage::COPY_DST,
        memory: MemoryClass::Device,
    };
    let buffer = device.create_buffer(&desc).unwrap();
    let (first_gpu_address, first_weak) = {
        // SAFETY: `buffer` is live for this whole scope and the fixture
        // only reads its owned placed-resource allocation facts.
        let native = unsafe { buffer.native_ref() };
        assert!(
            native.default_allocation.is_some(),
            "ordinary buffer uses the default pool"
        );
        (
            unsafe { native.resource.GetGPUVirtualAddress() },
            Arc::downgrade(&native.mapped),
        )
    };
    assert_ne!(first_gpu_address, 0, "placed buffer has a GPU address");
    let mut encoder = device.encoder(direct).unwrap();
    encoder.write_buffer(&buffer, 0, &[1, 2, 3, 4]).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 99_944,
            },
            Arc::from([StepDesc {
                lane: direct,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    // SAFETY: the finished list is recorded only for its declared lane.
    let receipt = unsafe {
        device.queue().submit(SubmitPacket {
            plan: plan.clone(),
            steps: vec![BoundStep {
                step: plan.step(0),
                payload: StepPayload::Commands(vec![encoder.finish().unwrap()]),
            }],
            completion: CompletionRequest::None,
        })
    };
    assert!(matches!(receipt, SubmitOutcome::Accepted(_)));
    drop(receipt);
    drop(buffer);

    // The private queue wait keeps this submission in the direct in-flight
    // store. The only public owner is gone, so a live buffer here can only
    // come from submitted work retaining its pool lease and range.
    assert!(
        first_weak.upgrade().is_some(),
        "submitted work retains the default-pool lease"
    );
    let competing = device.create_buffer(&desc).unwrap();
    let competing_gpu_address = {
        // SAFETY: `competing` remains live while its native address is read.
        let native = unsafe { competing.native_ref() };
        assert!(native.default_allocation.is_some());
        unsafe { native.resource.GetGPUVirtualAddress() }
    };
    assert_ne!(competing_gpu_address, first_gpu_address);
    drop(competing);

    gate.release();
    drop(gate);

    let deadline = Instant::now() + Duration::from_secs(5);
    while first_weak.upgrade().is_some() {
        assert!(
            Instant::now() < deadline,
            "direct fence retirement did not release the default-pool lease"
        );
        thread::sleep(Duration::from_millis(5));
    }
    let reused = device.create_buffer(&desc).unwrap();
    let reused_gpu_address = {
        // SAFETY: `reused` remains live while its native address is read.
        let native = unsafe { reused.native_ref() };
        assert!(native.default_allocation.is_some());
        unsafe { native.resource.GetGPUVirtualAddress() }
    };
    assert_eq!(reused_gpu_address, first_gpu_address);
}

#[test]
fn every_failed_setup_stage_quarantines_until_explicit_idle_recovery() {
    for (offset, stage) in [
        super::FailureStage::Event,
        super::FailureStage::Wait,
        super::FailureStage::SetEvent,
    ]
    .into_iter()
    .enumerate()
    {
        failed_stage_quarantines_until_idle(stage, offset as u64);
    }
}

fn failed_stage_quarantines_until_idle(stage: super::FailureStage, offset: u64) {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let mut gate = hold_native_queues(&device, false);
    let direct = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let buffer = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 4,
            usage: BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let buffer_weak = Arc::downgrade(&unsafe { buffer.native_ref() }.mapped);
    let state_weak = Arc::downgrade(&unsafe { device.native_ref() }.0);
    let mut encoder = device.encoder(direct).unwrap();
    encoder.write_buffer(&buffer, 0, &[9, 8, 7, 6]).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 99_942 + offset,
            },
            Arc::from([StepDesc {
                lane: direct,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    super::inject_failure(stage);
    // SAFETY: the completed command list belongs to this plan's direct lane.
    let receipt = unsafe {
        device.queue().submit(SubmitPacket {
            plan: plan.clone(),
            steps: vec![BoundStep {
                step: plan.step(0),
                payload: StepPayload::Commands(vec![encoder.finish().unwrap()]),
            }],
            completion: CompletionRequest::None,
        })
    };
    assert!(matches!(receipt, SubmitOutcome::Indeterminate { .. }));
    drop(receipt);
    assert_eq!(
        unsafe { device.native_ref() }
            .0
            .abandoned_fence_events
            .lock()
            .unwrap()
            .len(),
        1,
        "failed unarmed setup must retain the accepted submission"
    );

    // The queue is still behind the private fence. This prevents a fast
    // GPU from bypassing all injected setup stages through the synchronous
    // completion fast path.
    assert!(state_weak.upgrade().is_some());
    assert!(buffer_weak.upgrade().is_some());
    gate.release();
    drop(gate);

    // This is the explicit recovery operation available after an
    // indeterminate handoff. It observes completion and releases both the
    // event and the error-only self-retention Arc.
    block(device.wait_idle()).unwrap();
    assert!(
        unsafe { device.native_ref() }
            .0
            .abandoned_fence_events
            .lock()
            .unwrap()
            .is_empty()
    );
    drop(buffer);
    drop(device);

    let deadline = Instant::now() + Duration::from_secs(5);
    while state_weak.upgrade().is_some() || buffer_weak.upgrade().is_some() {
        assert!(
            Instant::now() < deadline,
            "explicit recovery left a failed-setup device retention cycle"
        );
        thread::sleep(Duration::from_millis(5));
    }
}
