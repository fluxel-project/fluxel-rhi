//! Native timeline synchronization for independent queues in one family.

use super::{
    completion::{Completion, fence_waiter, retain_with_sync},
    *,
};

pub(in crate::backend) struct Timeline {
    device: Arc<DeviceState>,
    pub(in crate::backend) semaphore: vk::Semaphore,
}
impl Drop for Timeline {
    fn drop(&mut self) {
        // SAFETY: submitted waits and signals retain this owner until retirement.
        unsafe { self.device.device.destroy_semaphore(self.semaphore, None) };
    }
}
impl Timeline {
    fn new(device: &VulkanDevice) -> Result<Arc<Self>> {
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        // SAFETY: feature was enabled at device creation; info lives for this call.
        let semaphore =
            unsafe { device.0.device.create_semaphore(&info, None) }.map_err(vk_error)?;
        Ok(Arc::new(Self {
            device: device.0.clone(),
            semaphore,
        }))
    }
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
        let timelines = match (Timeline::new(device), Timeline::new(device)) {
            (Ok(a), Ok(b)) => [a, b],
            (Err(error), _) | (_, Err(error)) => return Outcome::Rejected { error, packet },
        };
        let fence = match unsafe {
            device
                .0
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => Completion::new(device.0.clone(), fence),
            Err(error) => {
                return Outcome::Rejected {
                    error: vk_error(error),
                    packet,
                };
            }
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
        let mut prior = {
            let frontier = device
                .0
                .lane_frontier
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            frontier.each_ref().map(|entry| {
                entry
                    .as_ref()
                    .and_then(|(timeline, value)| timeline.upgrade().map(|t| (t, *value)))
            })
        };
        let mut step_values: Vec<(usize, u64)> = vec![(usize::MAX, 0); first];
        let mut issued = Vec::new();
        let mut retained = timelines.to_vec();
        let mut sparse_issued = Vec::new();
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
                std::mem::forget(issued);
                std::mem::forget(retained);
                std::mem::forget(sparse_issued);
                std::mem::forget(fence);
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
            } else if let Some((timeline, value)) = prior[lane].take() {
                wait_semaphores.push(timeline.semaphore);
                wait_values.push(value);
                retained.push(timeline);
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
            values[lane] += 1;
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
                    sparse_issued.push(bind);
                    if let Err(e) = result {
                        error = Some(e);
                        break;
                    }
                    previous = Some(values[lane]);
                    values[lane] += 1;
                }
                if error.is_none() {
                    values[lane] -= 1;
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
            device
                .0
                .lane_frontier
                .lock()
                .unwrap_or_else(|p| p.into_inner())[lane] =
                Some((Arc::downgrade(&timelines[lane]), values[lane]));
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
                let semaphore = [timelines[0].semaphore, timelines[1].semaphore];
                let marker_values = values;
                let stages = [vk::PipelineStageFlags::ALL_COMMANDS; 2];
                let mut ti = vk::TimelineSemaphoreSubmitInfo::default()
                    .wait_semaphore_values(&marker_values);
                let info = vk::SubmitInfo::default()
                    .wait_semaphores(&semaphore)
                    .wait_dst_stage_mask(&stages)
                    .push_next(&mut ti);
                let result = {
                    let _queue = device
                        .0
                        .queue_gate
                        .lock()
                        .unwrap_or_else(|p| p.into_inner());
                    // SAFETY: marker waits for both lane frontiers before its fence signals.
                    unsafe {
                        device.0.device.queue_submit(
                            device.0.graphics_queue,
                            &[info],
                            marker.fence(),
                        )
                    }
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
                        lane_frontiers: frontiers.clone(),
                    },
                });
                retain_with_sync(marker, Vec::new(), timelines.to_vec(), Vec::new());
            }
        }
        let semaphores = [timelines[0].semaphore, timelines[1].semaphore];
        let wait_values = values;
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS; 2];
        let mut ti = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wait_values);
        let info = vk::SubmitInfo::default()
            .wait_semaphores(&semaphores)
            .wait_dst_stage_mask(&stages)
            .push_next(&mut ti);
        let result = {
            let _queue = device
                .0
                .queue_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            // SAFETY: the marker waits for both lane frontiers and owns its fence.
            unsafe {
                device
                    .0
                    .device
                    .queue_submit(device.0.graphics_queue, &[info], fence.fence())
            }
        };
        if let Err(error) = result {
            uncertain!(vk_error(error));
        }
        retain_with_sync(fence.clone(), issued, retained, sparse_issued);
        if matches!(completion, api::CompletionRequest::End) {
            checkpoints.push(api::Checkpoint {
                point: unsafe { api::CompletionPoint::from_native(plan.id().device, fence) },
                covers: api::CompletionCoverage {
                    lane_frontiers: frontiers,
                },
            });
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
