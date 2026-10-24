//! Real Vulkan proof that per-lane terminal fences do not add a queue join.

use super::*;
use crate::backend::VulkanInstance;
use ash::vk::Handle;
use fluxel_rhi_core::api::*;
use std::{future::Future, sync::Arc, thread, time::Duration};

struct Gate {
    state: Arc<DeviceState>,
    semaphore: vk::Semaphore,
    released: bool,
}

impl Gate {
    fn release(&mut self) {
        if !std::mem::replace(&mut self.released, true) {
            let signal = vk::SemaphoreSignalInfo::default()
                .semaphore(self.semaphore)
                .value(1);
            // SAFETY: this fixture owns the only host signal for value 1.
            unsafe { self.state.device.signal_semaphore(&signal) }
                .expect("release held Vulkan compute gate");
        }
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        if !std::mem::replace(&mut self.released, true) {
            let signal = vk::SemaphoreSignalInfo::default()
                .semaphore(self.semaphore)
                .value(1);
            // SAFETY: panic cleanup must release the queue but must not panic
            // again if a driver/device fault prevented it.
            let _ = unsafe { self.state.device.signal_semaphore(&signal) };
        }
    }
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

fn hold_compute_queue(device: &Device<Vulkan>) -> Option<Gate> {
    // SAFETY: this fixture keeps the typed device live and performs one private
    // queue wait before RHI submits its independent packet.
    let state = unsafe { device.native_ref() }.0.clone();
    let queue = state.compute_queue?;
    if !state.timeline_semaphore {
        return None;
    }
    let mut kind = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
    // SAFETY: timeline feature is enabled on this logical device.
    let semaphore = unsafe { state.device.create_semaphore(&info, None) }.ok()?;
    let waits = [semaphore];
    let values = [1];
    let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
    let submit = vk::SubmitInfo::default()
        .wait_semaphores(&waits)
        .wait_dst_stage_mask(&stages)
        .push_next(&mut timeline);
    // SAFETY: no other raw queue caller exists in this fixture; the RHI packet
    // is submitted only after this ordering wait is enqueued.
    if unsafe {
        state
            .device
            .queue_submit(queue, &[submit], vk::Fence::null())
    }
    .is_err()
    {
        unsafe { state.device.destroy_semaphore(semaphore, None) };
        return None;
    }
    Some(Gate {
        state,
        semaphore,
        released: false,
    })
}

#[test]
fn graphics_completion_does_not_join_held_compute_queue() {
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let Some(mut gate) = hold_compute_queue(&device) else {
        eprintln!("Vulkan independent timeline compute queue unavailable; skipping");
        return;
    };
    let graphics = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let compute = match device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Compute)
    {
        Some(lane) => lane.id,
        None => {
            eprintln!("Vulkan exposes no independent compute lane; skipping");
            return;
        }
    };
    eprintln!(
        "vulkan_queue_gate executing graphics_family={} compute_family={} graphics_queue=0x{:x} compute_queue=0x{:x}",
        gate.state.graphics_family,
        gate.state.compute_family,
        gate.state.graphics_queue.as_raw(),
        gate.state
            .compute_queue
            .expect("gate has compute queue")
            .as_raw(),
    );
    let direct_work = device.encoder(graphics).unwrap().finish().unwrap();
    let compute_work = device.encoder(compute).unwrap().finish().unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 74_102,
            },
            Arc::from([
                StepDesc {
                    lane: graphics,
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
    let direct_step = plan.step(0);
    let compute_step = plan.step(1);
    let receipt = unsafe {
        device.multi_queue().unwrap().submit(MultiQueuePacket {
            work: SubmitPacket {
                plan,
                steps: vec![
                    BoundStep {
                        step: direct_step,
                        payload: StepPayload::Commands(vec![direct_work]),
                    },
                    BoundStep {
                        step: compute_step,
                        payload: StepPayload::Commands(vec![compute_work]),
                    },
                ],
                completion: CompletionRequest::End,
            },
            signals: vec![],
            waits: vec![],
        })
    };
    let receipt = match receipt {
        MultiQueueSubmitOutcome::Accepted(receipt) => receipt,
        _ => panic!("independent Vulkan packet was not accepted"),
    };
    let direct_point = receipt
        .work
        .checkpoints
        .iter()
        .find(|checkpoint| checkpoint.covers.lane_frontiers == [(graphics, direct_step)])
        .map(|checkpoint| &checkpoint.point)
        .expect("per-graphics completion point");
    let compute_point = receipt
        .work
        .checkpoints
        .iter()
        .find(|checkpoint| checkpoint.covers.lane_frontiers == [(compute, compute_step)])
        .map(|checkpoint| &checkpoint.point)
        .expect("per-compute completion point");
    block(device.wait_completion(direct_point)).unwrap();
    assert!(matches!(
        device.completion_status(compute_point).unwrap(),
        CompletionState::Pending
    ));
    // The aggregate idle marker must include held compute, while the graphics
    // checkpoint above remains independently observable.
    let mut idle = std::pin::pin!(device.wait_idle());
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    assert!(matches!(
        idle.as_mut().poll(&mut context),
        std::task::Poll::Pending
    ));
    gate.release();
    block(idle).unwrap();
    block(device.wait_completion(compute_point)).unwrap();
    // SAFETY: both queue markers completed, so nothing still references this semaphore.
    unsafe { gate.state.device.destroy_semaphore(gate.semaphore, None) };
    gate.semaphore = vk::Semaphore::null();
}

#[test]
fn terminal_marker_failure_quarantines_drained_lane_work() {
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let Some(mut gate) = hold_compute_queue(&device) else {
        eprintln!("Vulkan independent timeline compute queue unavailable; skipping");
        return;
    };
    let compute = match device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Compute)
    {
        Some(lane) => lane.id,
        None => {
            eprintln!("Vulkan exposes no independent compute lane; skipping");
            return;
        }
    };
    let buffer = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 4,
            usage: BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let weak = Arc::downgrade(&unsafe { buffer.native_ref() }.0);
    let mut encoder = device.encoder(compute).unwrap();
    encoder.write_buffer(&buffer, 0, &[1, 2, 3, 4]).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 74_103,
            },
            Arc::from([StepDesc {
                lane: compute,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    super::inject_terminal_submit_failure();
    // SAFETY: the held compute queue has accepted this one matching work item.
    let outcome = unsafe {
        device.multi_queue().unwrap().submit(MultiQueuePacket {
            work: SubmitPacket {
                plan: plan.clone(),
                steps: vec![BoundStep {
                    step: plan.step(0),
                    payload: StepPayload::Commands(vec![encoder.finish().unwrap()]),
                }],
                completion: CompletionRequest::None,
            },
            signals: vec![],
            waits: vec![],
        })
    };
    assert!(matches!(
        outcome,
        MultiQueueSubmitOutcome::Indeterminate { .. }
    ));
    drop(outcome);
    drop(buffer);
    // The terminal marker error happens after GPU handoff. The queue remains
    // gated, but the recorded work/resource is intentionally quarantined
    // rather than naturally dropped from the drained lane vector.
    assert!(weak.upgrade().is_some());
    gate.release();
}
