//! Native timeline synchronization and per-lane retirement for exposed queues.

use super::{
    completion::{Completion, fence_waiter, retain_with_sync},
    *,
};

pub(in crate::backend) use super::Timeline;

#[cfg(test)]
thread_local! {
    static TEST_MULTI_SUBMITS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static TEST_FAIL_TERMINAL_SUBMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(super) fn reset_test_submit_count() {
    TEST_MULTI_SUBMITS.with(|count| count.set(0));
}

#[cfg(test)]
pub(super) fn test_submit_count() -> u64 {
    TEST_MULTI_SUBMITS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(super) fn inject_terminal_submit_failure() {
    TEST_FAIL_TERMINAL_SUBMIT.with(|fail| fail.set(true));
}

#[cfg(test)]
fn take_terminal_submit_failure() -> bool {
    TEST_FAIL_TERMINAL_SUBMIT.with(|fail| fail.replace(false))
}

pub struct VulkanGpuSignal {
    timeline: Arc<Timeline>,
    value: u64,
}

impl api::MultiQueueBackend for Vulkan {
    type NativeGpuSignal = VulkanGpuSignal;
    unsafe fn submit_multi(
        device: &VulkanDevice,
        packet: api::MultiQueuePacket<Self>,
    ) -> api::MultiQueueSubmitOutcome<Self> {
        use api::MultiQueueSubmitOutcome as Outcome;
        #[cfg(test)]
        TEST_MULTI_SUBMITS.with(|count| count.set(count.get() + 1));
        let _gate = device
            .0
            .submit_gate
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let validate = || -> Result<()> {
            if !device.0.timeline_semaphore {
                return Err(Error::Unsupported("Vulkan independent queues"));
            }
            for step in &packet.work.steps {
                let desc = &packet.work.plan.steps()[step.step.index as usize];
                if desc.lane.ordinal > 1
                    || (desc.lane.ordinal == 1 && device.0.compute_queue.is_none())
                {
                    return Err(Error::WrongLane);
                }
                match &step.payload {
                    StepPayload::Commands(records) => {
                        for record in records {
                            // SAFETY: Core validates packet device and backend identity.
                            let native = unsafe { record.native_ref() };
                            if native.encoder.lane != desc.lane {
                                return Err(Error::WrongLane);
                            }
                        }
                    }
                    // Presentation itself uses the graphics WSI queue; the
                    // binary signal can originate on either exposed lane.
                    StepPayload::Present(_) => {}
                    StepPayload::SparseBind(binds)
                        if device.0.sparse_image
                            && (if desc.lane.ordinal == 0 {
                                device.0.graphics_sparse_binding
                            } else {
                                device.0.compute_sparse_binding
                            })
                            && !binds.is_empty() => {}
                    _ => {
                        return Err(Error::Unsupported(
                            "Vulkan multi-queue sparse binding or present position",
                        ));
                    }
                }
            }
            super::super::query::validate_ops(packet.work.steps.iter().flat_map(|step| {
                match &step.payload {
                    StepPayload::Commands(work) => work.as_slice(),
                    _ => &[],
                }
                .iter()
                .flat_map(|record| {
                    let encoder = &unsafe { record.native_ref() }.encoder;
                    encoder
                        .query_ops
                        .iter()
                        .map(move |op| (encoder.lane.ordinal, op))
                })
            }))?;
            Ok(())
        };
        if let Err(error) = validate() {
            return Outcome::Rejected { error, packet };
        }
        let first = packet
            .work
            .steps
            .first()
            .map_or(packet.work.plan.steps().len(), |step| {
                step.step.index as usize
            });
        let mut prepared: Vec<Vec<super::super::sparse::PreparedSparseBind>> =
            (0..first).map(|_| Vec::new()).collect();
        for step in &packet.work.steps {
            let mut items = Vec::new();
            if let StepPayload::SparseBind(binds) = &step.payload {
                for bind in binds {
                    match super::super::sparse::prepare_bind(bind) {
                        Ok(bind) => items.push(bind),
                        Err(error) => return Outcome::Rejected { error, packet },
                    }
                }
            }
            prepared.push(items);
        }
        let transaction: Vec<_> = prepared.iter().flat_map(|step| step.iter()).collect();
        if let Err(error) = super::super::sparse::validate_transaction(&transaction) {
            return Outcome::Rejected { error, packet };
        }
        let timelines = match Timeline::acquire(device) {
            Ok(timelines) => timelines,
            Err(error) => return Outcome::Rejected { error, packet },
        };
        let make_terminal = || unsafe {
            device
                .0
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map(|fence| Completion::new(device.0.clone(), fence))
                .map_err(vk_error)
        };
        let mut terminal_fences = match (make_terminal(), make_terminal()) {
            (Ok(graphics), Ok(compute)) => [Some(graphics), Some(compute)],
            (Err(error), _) | (_, Err(error)) => return Outcome::Rejected { error, packet },
        };
        let mut present_signals = (0..first).map(|_| None).collect::<Vec<_>>();
        for step in &packet.work.steps {
            if matches!(step.payload, StepPayload::Present(_)) {
                // SAFETY: each fresh semaphore is owned until its image accepts it.
                let semaphore = match unsafe {
                    device
                        .0
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                } {
                    Ok(semaphore) => semaphore,
                    Err(error) => {
                        return Outcome::Rejected {
                            error: vk_error(error),
                            packet,
                        };
                    }
                };
                present_signals.push(Some(PendingPresentSignal {
                    device: device.0.clone(),
                    semaphore: Some(semaphore),
                }));
            } else {
                present_signals.push(None);
            }
        }
        let requested = match &packet.work.completion {
            api::CompletionRequest::Steps(ids) => ids.to_vec(),
            _ => Vec::new(),
        };
        let mut step_fences = Vec::new();
        for id in &requested {
            if step_fences.iter().any(|(existing, _)| existing == id) {
                continue;
            }
            // SAFETY: an unpublished fence is owned by Completion before any native handoff.
            let native = match unsafe {
                device
                    .0
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            } {
                Ok(native) => native,
                Err(error) => {
                    return Outcome::Rejected {
                        error: vk_error(error),
                        packet,
                    };
                }
            };
            step_fences.push((*id, Completion::new(device.0.clone(), native)));
        }
        if let Err(error) = fence_waiter() {
            return Outcome::Rejected { error, packet };
        }
        let api::MultiQueuePacket {
            work:
                api::SubmitPacket {
                    plan,
                    steps,
                    completion,
                },
            signals,
            waits,
        } = packet;
        let mut waits: Vec<_> = waits.into_iter().map(Some).collect();
        let mut values = [0u64; 2];
        let mut step_values: Vec<(usize, u64)> = vec![(usize::MAX, 0); first];
        let mut issued = Vec::new();
        let mut retained = timelines.to_vec();
        let mut sparse_issued = Vec::new();
        let mut lane_work: [Vec<RecordedWork>; 2] = std::array::from_fn(|_| Vec::new());
        let mut lane_sparse: [Vec<super::super::sparse::PreparedSparseBind>; 2] =
            std::array::from_fn(|_| Vec::new());
        let mut accepted = Vec::new();
        let mut frontiers = Vec::new();
        let mut tokens = Vec::new();
        let mut presents = Vec::new();
        let mut checkpoints = Vec::new();
        macro_rules! uncertain {
            ($error:expr) => {{
                // No completion proof after a failed queue call: keep all native owners alive.
                for record in &issued {
                    super::super::query::invalidate_ops(
                        &record.encoder.query_ops,
                        record.encoder.lane.ordinal,
                    );
                }
                for record in lane_work.iter().flatten() {
                    super::super::query::invalidate_ops(
                        &record.encoder.query_ops,
                        record.encoder.lane.ordinal,
                    );
                }
                std::mem::forget(issued);
                std::mem::forget(retained);
                std::mem::forget(sparse_issued);
                std::mem::forget(lane_work);
                std::mem::forget(lane_sparse);
                std::mem::forget(terminal_fences);
                std::mem::forget(step_fences);
                return Outcome::Indeterminate {
                    error: $error,
                    uncertain: unsafe { api::NativeUncertainty::from_native(plan.id().device, ()) },
                };
            }};
        }
        for step in steps {
            let id = step.step;
            let desc = &plan.steps()[id.index as usize];
            let lane = desc.lane.ordinal as usize;
            let mut wait_semaphores = Vec::new();
            let mut wait_values = Vec::new();
            for dependency in &desc.waits {
                if (dependency.previous.index as usize) < first {
                    // The unsafe resume caller already observed the accepted frontier complete.
                    continue;
                }
                let (producer, value) = step_values[dependency.previous.index as usize];
                if producer != lane {
                    wait_semaphores.push(timelines[producer].semaphore);
                    wait_values.push(value);
                }
            }
            // Sparse bindings and commands need an execution dependency even
            // when both are issued on the same native queue.
            if values[lane] != 0 {
                wait_semaphores.push(timelines[lane].semaphore);
                wait_values.push(values[lane]);
            }
            for entry in &mut waits {
                if entry.as_ref().is_some_and(|w| w.signal.consumer() == id) {
                    let wait = entry.take().expect("matching wait");
                    // SAFETY: Core guarantees this token belongs to this consumer execution.
                    let signal = unsafe { wait.signal.into_native() };
                    wait_semaphores.push(signal.timeline.semaphore);
                    wait_values.push(signal.value);
                    retained.push(signal.timeline);
                }
            }
            // Vulkan permits each semaphore at most once in a submit; the
            // largest timeline value subsumes earlier waits on the same object.
            let mut index = 0;
            while index < wait_semaphores.len() {
                let mut other = index + 1;
                while other < wait_semaphores.len() {
                    if wait_semaphores[index] == wait_semaphores[other] {
                        wait_values[index] = wait_values[index].max(wait_values[other]);
                        wait_semaphores.remove(other);
                        wait_values.remove(other);
                    } else {
                        other += 1;
                    }
                }
                index += 1;
            }
            let start = issued.len();
            let mut present_binding = None;
            let mut sparse_step = false;
            match step.payload {
                StepPayload::Commands(work) => {
                    issued.extend(work.into_iter().map(|w| unsafe { w.into_native() }))
                }
                StepPayload::Present(binding) => {
                    let pending = present_signals[id.index as usize]
                        .as_mut()
                        .expect("preallocated present semaphore");
                    let semaphore = pending.semaphore.expect("present semaphore owned");
                    // SAFETY: Core validated the exclusive image lease and its device.
                    if let Err(error) =
                        unsafe { binding.image().native_ref() }.set_present_wait(semaphore)
                    {
                        uncertain!(error);
                    }
                    pending.semaphore = None;
                    present_binding = Some((binding, semaphore));
                }
                StepPayload::SparseBind(_) => sparse_step = true,
            }
            super::super::ray::retention::pin_batch(&mut issued[start..]);
            let has_external = issued[start..]
                .iter()
                .any(|record| !record.encoder.external_segments.is_empty());
            let commands: Vec<_> = issued[start..].iter().map(RecordedWork::command).collect();
            values[lane] = match timelines[lane].next_value() {
                Ok(value) => value,
                Err(error) => uncertain!(error),
            };
            let mut signal_values = vec![values[lane]];
            let mut signal_semaphores = vec![timelines[lane].semaphore];
            if let Some((_, semaphore)) = &present_binding {
                signal_semaphores.push(*semaphore);
                signal_values.push(0);
            }
            let stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; wait_semaphores.len()];
            let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values)
                .signal_semaphore_values(&signal_values);
            let info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&stages)
                .command_buffers(&commands)
                .signal_semaphores(&signal_semaphores)
                .push_next(&mut timeline_info);
            let queue = if lane == 0 {
                device.0.graphics_queue
            } else {
                device.0.compute_queue.unwrap()
            };
            let result = if sparse_step {
                let mut binds = std::mem::take(&mut prepared[id.index as usize]);
                let initial_waits: Vec<_> = wait_semaphores
                    .iter()
                    .copied()
                    .zip(wait_values.iter().copied())
                    .collect();
                let mut previous = None;
                let mut error = None;
                for mut bind in binds.drain(..) {
                    let dependencies = previous.map_or_else(
                        || initial_waits.clone(),
                        |value| vec![(timelines[lane].semaphore, value)],
                    );
                    let result = {
                        let _queue = device
                            .0
                            .queue_gate
                            .lock()
                            .unwrap_or_else(|p| p.into_inner());
                        bind.submit(device, queue, &dependencies, &timelines[lane], values[lane])
                    };
                    sparse_issued.push((lane, bind));
                    if let Err(e) = result {
                        error = Some(e);
                        break;
                    }
                    previous = Some(values[lane]);
                    values[lane] = match timelines[lane].next_value() {
                        Ok(value) => value,
                        Err(error) => uncertain!(error),
                    };
                }
                if error.is_none() {
                    values[lane] =
                        previous.expect("non-empty sparse binding issued a timeline value");
                }
                if let Some(error) = error {
                    Err(error)
                } else {
                    Ok(())
                }
            } else if has_external {
                let waits: Vec<_> = wait_semaphores
                    .iter()
                    .copied()
                    .zip(wait_values.iter().copied())
                    .collect();
                let signals: Vec<_> = signal_semaphores
                    .iter()
                    .copied()
                    .zip(signal_values.iter().copied())
                    .collect();
                super::submit_external_records(
                    device,
                    queue,
                    &issued[start..],
                    &waits,
                    &signals,
                    true,
                )
            } else {
                let _queue = device
                    .0
                    .queue_gate
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                // SAFETY: queue is externally synchronized; command and semaphore owners are retained.
                unsafe {
                    device
                        .0
                        .device
                        .queue_submit(queue, &[info], vk::Fence::null())
                }
                .map_err(vk_error)
            };
            if let Err(error) = result {
                // The queue may have observed the signal semaphore and image lease.
                std::mem::forget(present_binding);
                uncertain!(error);
            }
            for record in &issued[start..] {
                if !has_external {
                    super::super::ray::retention::publish_segment(
                        record,
                        record.encoder.external_segments.len(),
                    );
                }
                super::super::query::commit_ops(&record.encoder.query_ops, lane as u32);
            }
            if let Some((mut binding, _)) = present_binding {
                // SAFETY: the queue signal was accepted before handing off the image.
                if let Err(error) = unsafe { binding.try_handoff() } {
                    std::mem::forget(binding);
                    uncertain!(error);
                }
                presents.push(api::PresentReceipt {
                    step: Some(id),
                    accepted: true,
                });
            }
            step_values.push((lane, values[lane]));
            accepted.push(id);
            frontiers.retain(|(existing, _)| *existing != desc.lane);
            frontiers.push((desc.lane, id));
            for request in signals.iter().filter(|s| s.producer == id) {
                let native = VulkanGpuSignal {
                    timeline: timelines[lane].clone(),
                    value: values[lane],
                };
                // SAFETY: signal was enqueued immediately after this producer's commands.
                tokens.push(unsafe {
                    api::GpuSignalPoint::from_native(
                        plan.id().device,
                        id,
                        desc.lane,
                        request.consumer,
                        request.consumer_lane,
                        native,
                    )
                });
            }
            if let Some(index) = step_fences.iter().position(|(step, _)| *step == id) {
                let marker = step_fences.remove(index).1;
                let result = {
                    let _queue = device
                        .0
                        .queue_gate
                        .lock()
                        .unwrap_or_else(|p| p.into_inner());
                    // SAFETY: same-queue order proves this requested step and
                    // its explicit Vulkan waits; no unrelated lane is joined.
                    unsafe { device.0.device.queue_submit(queue, &[], marker.fence()) }
                };
                if let Err(error) = result {
                    std::mem::forget(marker);
                    uncertain!(vk_error(error));
                }
                checkpoints.push(api::Checkpoint {
                    point: unsafe {
                        api::CompletionPoint::from_native(plan.id().device, marker.clone())
                    },
                    covers: api::CompletionCoverage {
                        lane_frontiers: vec![(desc.lane, id)],
                    },
                });
                retain_with_sync(
                    marker,
                    Vec::new(),
                    vec![timelines[lane].clone()],
                    Vec::new(),
                );
            }
        }
        for work in issued.drain(..) {
            lane_work[work.encoder.lane.ordinal as usize].push(work);
        }
        for (lane, bind) in sparse_issued.drain(..) {
            lane_sparse[lane].push(bind);
        }
        for lane in 0..2 {
            if values[lane] == 0 {
                continue;
            }
            let marker = terminal_fences[lane]
                .take()
                .expect("active queue has a terminal fence");
            let queue = if lane == 0 {
                device.0.graphics_queue
            } else {
                device.0.compute_queue.expect("compute lane was validated")
            };
            let result = {
                #[cfg(test)]
                if take_terminal_submit_failure() {
                    std::mem::forget(marker);
                    uncertain!(Error::Backend(
                        "injected Vulkan terminal marker failure".into()
                    ));
                }
                let _queue = device
                    .0
                    .queue_gate
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                // SAFETY: this marker follows only its own queue's accepted
                // submissions, so it supplies per-lane retirement evidence.
                unsafe { device.0.device.queue_submit(queue, &[], marker.fence()) }
            };
            if let Err(error) = result {
                std::mem::forget(marker);
                uncertain!(vk_error(error));
            }
            retain_with_sync(
                marker.clone(),
                std::mem::take(&mut lane_work[lane]),
                retained.clone(),
                std::mem::take(&mut lane_sparse[lane]),
            );
            if matches!(completion, api::CompletionRequest::End) {
                if let Some((lane_id, step)) =
                    frontiers.iter().find(|(id, _)| id.ordinal as usize == lane)
                {
                    checkpoints.push(api::Checkpoint {
                        point: unsafe {
                            api::CompletionPoint::from_native(plan.id().device, marker)
                        },
                        covers: api::CompletionCoverage {
                            lane_frontiers: vec![(*lane_id, *step)],
                        },
                    });
                }
            }
        }
        Outcome::Accepted(api::MultiQueueReceipt {
            work: api::SubmissionReceipt {
                device: plan.id().device,
                index: None,
                accepted,
                checkpoints,
                presents,
            },
            signals: tokens,
        })
    }
}

#[cfg(all(test, feature = "hardware-tests"))]
#[path = "multi_queue/tests.rs"]
mod hardware_tests;
