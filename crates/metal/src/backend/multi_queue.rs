//! Independent Metal queue submission using per-step native GPU events.
//!
//! Fresh per-execution events synchronize command and present steps, including
//! cross-packet waits. Apple GPU and presentation acceptance is still pending.
use super::{Metal, MetalDevice};
use fluxel_rhi_core::api::{self, MultiQueueBackend, MultiQueueSubmitOutcome as Outcome};

pub struct MetalGpuSignal {
    #[cfg(target_vendor = "apple")]
    event: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>>,
    #[cfg(target_vendor = "apple")]
    producer: super::submit::Completion,
}

#[cfg(target_vendor = "apple")]
struct ReleaseOnFailure {
    completion: super::submit::Completion,
    event: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>>,
}
#[cfg(target_vendor = "apple")]
impl ReleaseOnFailure {
    fn observe(self: &std::sync::Arc<Self>) {
        use objc2_metal::MTLSharedEvent as _;
        let waker = std::task::Waker::from(self.clone());
        if matches!(self.completion.watch(&waker), Some(Err(_))) {
            // A failed producer still releases native waits. Their host
            // completion also joins the producer failure, so no result is
            // reported as successful. GPU signals use the larger value 2.
            self.event.setSignaledValue(1);
        }
    }
}
#[cfg(target_vendor = "apple")]
impl std::task::Wake for ReleaseOnFailure {
    fn wake(self: std::sync::Arc<Self>) {
        self.observe();
    }
    fn wake_by_ref(self: &std::sync::Arc<Self>) {
        self.observe();
    }
}

#[cfg(target_vendor = "apple")]
struct SparseDispatch {
    ready: super::submit::Completion,
    done: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>>,
    queue: std::sync::Arc<
        std::sync::Mutex<
            objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4CommandQueue>>,
        >,
    >,
    binds: std::sync::Mutex<Option<Vec<super::sparse::PreparedSparseBind>>>,
}
// SAFETY: this one-shot callback owns retained native resources. Access to the
// mapping queue is serialized by its shared mutex, each bind's mutable mapping
// bookkeeping is separately locked, and the event supports cross-thread signals.
// No Objective-C encoder is shared; GPU dependencies are explicit.
#[cfg(target_vendor = "apple")]
unsafe impl Send for SparseDispatch {}
#[cfg(target_vendor = "apple")]
unsafe impl Sync for SparseDispatch {}
#[cfg(target_vendor = "apple")]
impl SparseDispatch {
    fn observe(self: &std::sync::Arc<Self>) {
        use objc2_metal::{MTL4CommandQueue as _, MTLSharedEvent as _};
        let waker = std::task::Waker::from(self.clone());
        let Some(result) = self.ready.watch(&waker) else {
            return;
        };
        let Some(binds) = self.binds.lock().unwrap_or_else(|e| e.into_inner()).take() else {
            return;
        };
        if result.is_err() {
            self.done.setSignaledValue(1);
            return;
        }
        let queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        for bind in &binds {
            bind.enqueue(&queue);
        }
        let retained_queue = queue.clone();
        let event = self.done.clone();
        let retained_event = event.clone();
        let callback = block2::RcBlock::new(
            move |_: std::ptr::NonNull<
                objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>,
            >,
                  _: u64| {
                let _ = (&binds, &retained_queue, &retained_event);
            },
        );
        let listener = objc2_metal::MTLSharedEventListener::new();
        // SAFETY: Metal copies the callback; it retains mappings and the queue until signal completion.
        unsafe {
            event.notifyListener_atValue_block(
                &listener,
                1,
                block2::RcBlock::as_ptr(&callback).cast(),
            )
        };
        let native = objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(&*event);
        queue.signalEvent_value(native, 1);
    }
}
#[cfg(target_vendor = "apple")]
impl std::task::Wake for SparseDispatch {
    fn wake(self: std::sync::Arc<Self>) {
        self.observe();
    }
    fn wake_by_ref(self: &std::sync::Arc<Self>) {
        self.observe();
    }
}

impl MultiQueueBackend for Metal {
    type NativeGpuSignal = MetalGpuSignal;

    unsafe fn submit_multi(
        device: &MetalDevice,
        packet: api::MultiQueuePacket<Self>,
    ) -> Outcome<Self> {
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = device;
            Outcome::Rejected {
                error: api::Error::Unsupported("Metal requires an Apple target"),
                packet,
            }
        }
        #[cfg(target_vendor = "apple")]
        {
            use api::{CompletionCoverage, CompletionPoint, CompletionRequest, StepPayload};
            use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _, MTLDevice as _};
            let _gate = device
                ._state
                .queue_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let plan = packet.work.plan.clone();
            let start = packet
                .work
                .steps
                .first()
                .map_or(plan.steps().len(), |step| step.step.index as usize);
            let end = start + packet.work.steps.len();
            let plan_steps = &plan.steps()[start..end];
            if plan_steps.is_empty() {
                let checkpoints = if matches!(packet.work.completion, CompletionRequest::End) {
                    let done = super::submit::Completion::new();
                    done.publish(Ok(()));
                    vec![api::Checkpoint {
                        // SAFETY: an empty packet has no GPU work to complete.
                        point: unsafe { CompletionPoint::from_native(plan.id().device, done) },
                        covers: CompletionCoverage {
                            lane_frontiers: Vec::new(),
                        },
                    }]
                } else {
                    Vec::new()
                };
                return Outcome::Accepted(api::MultiQueueReceipt {
                    work: api::SubmissionReceipt {
                        device: plan.id().device,
                        index: None,
                        accepted: Vec::new(),
                        checkpoints,
                        presents: Vec::new(),
                    },
                    signals: Vec::new(),
                });
            }
            for (i, bound) in packet.work.steps.iter().enumerate() {
                let desc = &plan_steps[i];
                if desc.lane.ordinal > 1 {
                    return Outcome::Rejected {
                        error: api::Error::WrongLane,
                        packet,
                    };
                }
                let records = match &bound.payload {
                    StepPayload::Commands(records) => records.as_slice(),
                    StepPayload::Present(_) if desc.lane.ordinal == 0 => &[],
                    StepPayload::SparseBind(_) if super::sparse::placement_supported(device) => &[],
                    _ => {
                        return Outcome::Rejected {
                            error: api::Error::Unsupported("Metal multi-queue step"),
                            packet,
                        };
                    }
                };
                for record in records {
                    // SAFETY: Core verified backend and device provenance; this only borrows the recording.
                    let work = unsafe { record.native_ref() };
                    if work.lane != desc.lane.ordinal {
                        return Outcome::Rejected {
                            error: api::Error::WrongLane,
                            packet,
                        };
                    }
                }
            }
            let ops = packet
                .work
                .steps
                .iter()
                .flat_map(|step| match &step.payload {
                    StepPayload::Commands(records) => records.as_slice(),
                    _ => &[],
                })
                .flat_map(|record| {
                    // SAFETY: Core validates backend identity of each recording.
                    let work = unsafe { record.native_ref() };
                    work.query_ops.iter().map(move |op| (work.lane, op))
                });
            if let Err(error) = super::query::validate_ops(ops) {
                return Outcome::Rejected { error, packet };
            }
            let mut sparse = Vec::with_capacity(packet.work.steps.len());
            for bound in &packet.work.steps {
                let mut items = Vec::new();
                if let StepPayload::SparseBind(binds) = &bound.payload {
                    for bind in binds {
                        match super::sparse::prepare_bind(bind) {
                            Ok(item) => items.push(item),
                            Err(error) => return Outcome::Rejected { error, packet },
                        }
                    }
                }
                sparse.push(items);
            }
            let has_sparse = plan_steps
                .iter()
                .any(|step| step.kind == api::StepKind::SparseBind);
            let sparse_queue = if has_sparse {
                match device._state._device.newMTL4CommandQueue() {
                    Some(queue) => Some(std::sync::Arc::new(std::sync::Mutex::new(queue))),
                    None => {
                        return Outcome::Rejected {
                            error: api::Error::OutOfMemory,
                            packet,
                        };
                    }
                }
            } else {
                None
            };
            let mut present_commands = Vec::new();
            for bound in &packet.work.steps {
                if let StepPayload::Present(binding) = &bound.payload {
                    let Some(command) = device._state._queue.commandBuffer() else {
                        return Outcome::Rejected {
                            error: api::Error::OutOfMemory,
                            packet,
                        };
                    };
                    // SAFETY: Core validated this exclusive drawable lease.
                    let image = unsafe { binding.image().native_ref() };
                    let drawable =
                        objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLDrawable>::from_ref(
                            &*image.drawable,
                        );
                    let _ = drawable;
                    present_commands.push((bound.step, command));
                }
            }
            let mut events = Vec::with_capacity(plan_steps.len());
            let mut markers = Vec::with_capacity(plan_steps.len());
            let mut waits = Vec::with_capacity(plan_steps.len());
            let mut sparse_bridges = Vec::with_capacity(plan_steps.len());
            for desc in plan_steps {
                let queue = if desc.lane.ordinal == 0 {
                    &device._state._queue
                } else {
                    &device._state.compute_queue
                };
                let (Some(event), Some(wait), Some(marker)) = (
                    device._state._device.newSharedEvent(),
                    queue.commandBuffer(),
                    queue.commandBuffer(),
                ) else {
                    return Outcome::Rejected {
                        error: api::Error::OutOfMemory,
                        packet,
                    };
                };
                events.push(event);
                waits.push(wait);
                markers.push(marker);
                if desc.kind == api::StepKind::SparseBind {
                    let (Some(ready), Some(done), Some(bridge)) = (
                        device._state._device.newSharedEvent(),
                        device._state._device.newSharedEvent(),
                        queue.commandBuffer(),
                    ) else {
                        return Outcome::Rejected {
                            error: api::Error::OutOfMemory,
                            packet,
                        };
                    };
                    sparse_bridges.push(Some((ready, done, bridge)));
                } else {
                    sparse_bridges.push(None);
                }
            }
            for (i, desc) in plan_steps.iter().enumerate() {
                for edge in desc
                    .waits
                    .iter()
                    .filter(|edge| edge.previous.index as usize >= start)
                {
                    let event =
                        objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(
                            &*events[edge.previous.index as usize - start],
                        );
                    waits[i].encodeWaitForEvent_value(event, 1);
                }
                let event = objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(
                    &*events[i],
                );
                markers[i].encodeSignalEvent_value(event, 2);
                if let StepPayload::Present(binding) = &packet.work.steps[i].payload {
                    let command = &present_commands
                        .iter()
                        .find(|(step, _)| *step == packet.work.steps[i].step)
                        .expect("present preallocated")
                        .1;
                    for edge in desc
                        .waits
                        .iter()
                        .filter(|edge| edge.previous.index as usize >= start)
                    {
                        let event =
                            objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(
                                &*events[edge.previous.index as usize - start],
                            );
                        command.encodeWaitForEvent_value(event, 1);
                    }
                    // SAFETY: Core validated the drawable lease.
                    let image = unsafe { binding.image().native_ref() };
                    let drawable =
                        objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLDrawable>::from_ref(
                            &*image.drawable,
                        );
                    command.presentDrawable(drawable);
                }
                if let Some((ready, done, bridge)) = &sparse_bridges[i] {
                    let ready =
                        objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(
                            &**ready,
                        );
                    let done =
                        objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(
                            &**done,
                        );
                    waits[i].encodeSignalEvent_value(ready, 1);
                    bridge.encodeWaitForEvent_value(done, 1);
                }
            }
            let step_completions: Vec<_> = plan_steps
                .iter()
                .map(|_| super::submit::Completion::new())
                .collect();
            let mut last = [None, None];
            for (i, desc) in plan_steps.iter().enumerate() {
                last[desc.lane.ordinal as usize] = Some(i);
            }
            let completions: Vec<_> = (0..plan_steps.len())
                .filter(|&i| match &packet.work.completion {
                    CompletionRequest::None => false,
                    CompletionRequest::End => last.contains(&Some(i)),
                    CompletionRequest::Steps(steps) => steps.contains(&plan.step(start + i)),
                })
                .map(|i| (i, step_completions[i].clone()))
                .collect();
            let mut signal_points = Vec::with_capacity(packet.signals.len());
            for request in &packet.signals {
                let producer_index = request.producer.index as usize - start;
                let lane = plan_steps[producer_index].lane;
                // SAFETY: marker on the exact producer execution signals this fresh event.
                signal_points.push(unsafe {
                    api::GpuSignalPoint::from_native(
                        plan.id().device,
                        request.producer,
                        lane,
                        request.consumer,
                        request.consumer_lane,
                        MetalGpuSignal {
                            event: events[producer_index].clone(),
                            producer: step_completions[producer_index].clone(),
                        },
                    )
                });
            }
            let mut external: Vec<Vec<super::submit::Completion>> =
                (0..plan_steps.len()).map(|_| Vec::new()).collect();
            for wait in &packet.waits {
                let i = wait.signal.consumer().index as usize - start;
                // SAFETY: Core validates token provenance; borrowing does not consume it.
                let signal = unsafe { wait.signal.native_ref() };
                let event = objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(
                    &*signal.event,
                );
                waits[i].encodeWaitForEvent_value(event, 1);
                external[i].push(signal.producer.clone());
            }
            let api::MultiQueuePacket {
                work,
                signals: signal_requests,
                waits: mut remaining_waits,
            } = packet;
            let api::SubmitPacket {
                plan,
                steps,
                completion,
            } = work;
            let mut accepted: Vec<api::StepId> = Vec::new();
            let mut checkpoints = Vec::new();
            let mut lane_frontier = device
                ._state
                .lane_frontier
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone();
            let mut pending = steps.into_iter().enumerate();
            while let Some((i, mut bound)) = pending.next() {
                if let StepPayload::Present(binding) = &mut bound.payload {
                    // SAFETY: allocations were preflighted; handoff is performed only at this issue boundary.
                    if let Err(error) = unsafe { binding.try_handoff() } {
                        let mut tail = vec![bound];
                        tail.extend(pending.map(|(_, step)| step));
                        if accepted.is_empty() {
                            return Outcome::Rejected {
                                error,
                                packet: api::MultiQueuePacket {
                                    work: api::SubmitPacket {
                                        plan,
                                        steps: tail,
                                        completion,
                                    },
                                    signals: signal_requests,
                                    waits: remaining_waits,
                                },
                            };
                        }
                        let mut accepted_signals = Vec::new();
                        let mut remaining_signals = Vec::new();
                        for (request, point) in signal_requests.into_iter().zip(signal_points) {
                            if (request.producer.index as usize) < start + i {
                                accepted_signals.push(point);
                            } else {
                                remaining_signals.push(request);
                            }
                        }
                        for lane in 0..2 {
                            if let Some(last_step) = accepted
                                .iter()
                                .rev()
                                .find(|step| plan.steps()[step.index as usize].lane.ordinal == lane)
                            {
                                let index = last_step.index as usize - start;
                                checkpoints.push(api::Checkpoint {
                                    // SAFETY: the completion covers the accepted lane prefix.
                                    point: unsafe {
                                        CompletionPoint::from_native(
                                            plan.id().device,
                                            step_completions[index].clone(),
                                        )
                                    },
                                    covers: CompletionCoverage {
                                        lane_frontiers: vec![(plan_steps[index].lane, *last_step)],
                                    },
                                });
                            }
                        }
                        let presents = present_commands
                            .iter()
                            .filter(|(step, _)| accepted.contains(step))
                            .map(|(step, _)| api::PresentReceipt {
                                step: Some(*step),
                                accepted: true,
                            })
                            .collect();
                        return Outcome::PartiallyAccepted {
                            error,
                            accepted: api::MultiQueueReceipt {
                                work: api::SubmissionReceipt {
                                    device: plan.id().device,
                                    index: None,
                                    accepted,
                                    checkpoints,
                                    presents,
                                },
                                signals: accepted_signals,
                            },
                            unaccepted: api::MultiQueueTail {
                                plan,
                                pending: tail,
                                remaining_signals,
                                remaining_waits,
                                completion,
                            },
                        };
                    }
                }
                remaining_waits.retain(|wait| wait.signal.consumer() != bound.step);
                std::sync::Arc::new(ReleaseOnFailure {
                    completion: step_completions[i].clone(),
                    event: events[i].clone(),
                })
                .observe();
                let desc = &plan_steps[i];
                let mut covered = Vec::new();
                covered.extend(external[i].iter().cloned());
                if let Some(previous) = &lane_frontier[desc.lane.ordinal as usize] {
                    covered.push(previous.clone());
                }
                for edge in desc
                    .waits
                    .iter()
                    .filter(|edge| edge.previous.index as usize >= start)
                {
                    covered.push(step_completions[edge.previous.index as usize - start].clone());
                }
                covered.push(super::submit::track_command(device, &waits[i], None));
                waits[i].commit();
                let records = match bound.payload {
                    StepPayload::Commands(records) => records,
                    StepPayload::Present(_) => Vec::new(),
                    StepPayload::SparseBind(_) => Vec::new(),
                };
                for record in records {
                    // SAFETY: the validated one-shot packet transfers its recording once.
                    let work = unsafe { record.into_native() };
                    let query_commit =
                        std::sync::Arc::new(super::query::commit_ops(&work.query_ops, work.lane));
                    work.retain_indirect_arguments_until_completed();
                    work.submit_ray_builds();
                    covered.push(super::submit::track_command(
                        device,
                        &work.command,
                        Some((query_commit, work.lane)),
                    ));
                    work.command.commit();
                    work.guard_ray_completion(&mut covered);
                }
                for (step, command) in &present_commands {
                    if *step == bound.step {
                        covered.push(super::submit::track_command(device, command, None));
                        command.commit();
                    }
                }
                if let Some((_ready, done, bridge)) = &sparse_bridges[i] {
                    let ready = super::submit::Completion::new();
                    super::submit::join_into(&ready, covered.clone());
                    std::sync::Arc::new(SparseDispatch {
                        ready,
                        done: done.clone(),
                        queue: sparse_queue
                            .as_ref()
                            .expect("sparse queue prepared")
                            .clone(),
                        binds: std::sync::Mutex::new(Some(std::mem::take(&mut sparse[i]))),
                    })
                    .observe();
                    covered.push(super::submit::track_command(device, bridge, None));
                    bridge.commit();
                }
                covered.push(super::submit::track_command(device, &markers[i], None));
                super::submit::join_into(&step_completions[i], covered);
                lane_frontier[desc.lane.ordinal as usize] = Some(step_completions[i].clone());
                device
                    ._state
                    .lane_frontier
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())[desc.lane.ordinal as usize] =
                    Some(step_completions[i].clone());
                markers[i].commit();
                accepted.push(bound.step);
                if let Some((_, point)) = completions.iter().find(|(index, _)| *index == i) {
                    // A marker proves only its own queue prefix, not unordered other lanes.
                    let lane = plan_steps[i].lane;
                    checkpoints.push(api::Checkpoint {
                        point: unsafe {
                            CompletionPoint::from_native(plan.id().device, point.clone())
                        },
                        covers: CompletionCoverage {
                            lane_frontiers: vec![(lane, bound.step)],
                        },
                    });
                }
            }
            Outcome::Accepted(api::MultiQueueReceipt {
                work: api::SubmissionReceipt {
                    device: plan.id().device,
                    index: None,
                    accepted,
                    checkpoints,
                    presents: present_commands
                        .iter()
                        .map(|(step, _)| api::PresentReceipt {
                            step: Some(*step),
                            accepted: true,
                        })
                        .collect(),
                },
                signals: signal_points,
            })
        }
    }
}
