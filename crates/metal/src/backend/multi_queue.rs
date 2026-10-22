//! Independent Metal queue submission using per-step native GPU events.
//!
//! The route accepts same-packet command and present steps with at most four
//! waits per recording. A fresh event belongs to one producer execution, so
//! overlapping uses of the same Core plan cannot alias. Cross-packet GPU
//! tokens are rejected. Apple GPU and presentation acceptance is still pending.
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
            use objc2_metal::{
                MTL4CommandQueue as _, MTLCommandBuffer as _, MTLCommandQueue as _, MTLDevice as _,
                MTLSharedEvent as _,
            };
            let _gate = device
                ._state
                .queue_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if device
                ._state
                .sparse_pending
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_ref()
                .is_some_and(|p| p.result().is_none())
            {
                return Outcome::Rejected {
                    error: api::Error::InvalidState,
                    packet,
                };
            }
            let plan = packet.work.plan.clone();
            if packet
                .work
                .steps
                .iter()
                .any(|s| matches!(s.payload, StepPayload::SparseBind(_)))
            {
                if packet.work.steps.len() != 1
                    || !packet.waits.is_empty()
                    || !packet.signals.is_empty()
                    || plan.steps()[0].lane.ordinal != 0
                    || !plan.steps()[0].waits.is_empty()
                    || !super::sparse::placement_supported(device)
                {
                    return Outcome::Rejected {
                        error: api::Error::Unsupported(
                            "Metal sparse requires an isolated graphics step",
                        ),
                        packet,
                    };
                }
                let frontiers = device
                    ._state
                    .lane_frontier
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                if frontiers
                    .iter()
                    .flatten()
                    .any(|p| !matches!(p.result(), Some(Ok(()))))
                {
                    return Outcome::Rejected {
                        error: api::Error::InvalidState,
                        packet,
                    };
                }
                drop(frontiers);
                let StepPayload::SparseBind(binds) = &packet.work.steps[0].payload else {
                    unreachable!()
                };
                if binds.len() != 1 {
                    return Outcome::Rejected {
                        error: api::Error::Unsupported("Metal sparse requires one mapping bind"),
                        packet,
                    };
                }
                let mut prepared = Vec::new();
                for bind in binds {
                    match super::sparse::prepare_bind(bind) {
                        Ok(item) => prepared.push(item),
                        Err(error) => return Outcome::Rejected { error, packet },
                    }
                }
                let (Some(queue), Some(event)) = (
                    device._state._device.newMTL4CommandQueue(),
                    device._state._device.newSharedEvent(),
                ) else {
                    return Outcome::Rejected {
                        error: api::Error::OutOfMemory,
                        packet,
                    };
                };
                let done = super::submit::Completion::new();
                let done_callback = done.clone();
                let retained_queue = queue.clone();
                let retained_event = event.clone();
                let prepared = std::sync::Arc::new(prepared);
                let retained = prepared.clone();
                let callback = block2::RcBlock::new(
                    move |_: std::ptr::NonNull<
                        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>,
                    >,
                          _: u64| {
                        let _ = (&retained, &retained_queue, &retained_event);
                        done_callback.publish(Ok(()));
                    },
                );
                let listener = objc2_metal::MTLSharedEventListener::new();
                // SAFETY: Metal retains the block. It retains every mapping and
                // its old/new heaps until the actual Metal 4 signal is observed.
                unsafe {
                    event.notifyListener_atValue_block(
                        &listener,
                        1,
                        block2::RcBlock::as_ptr(&callback).cast(),
                    )
                };
                *device
                    ._state
                    .sparse_pending
                    .lock()
                    .unwrap_or_else(|p| p.into_inner()) = Some(done.clone());
                for bind in prepared.iter() {
                    bind.enqueue(&queue);
                }
                let native_event =
                    objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(&*event);
                queue.signalEvent_value(native_event, 1);
                let step = packet.work.steps[0].step;
                let requested = match &packet.work.completion {
                    CompletionRequest::None => false,
                    CompletionRequest::End => true,
                    CompletionRequest::Steps(steps) => steps.contains(&step),
                };
                let checkpoints = if requested {
                    vec![api::Checkpoint {
                        point: unsafe {
                            CompletionPoint::from_native(plan.id().device, done.clone())
                        },
                        covers: CompletionCoverage {
                            lane_frontiers: vec![(plan.steps()[0].lane, step)],
                        },
                    }]
                } else {
                    Vec::new()
                };
                device
                    ._state
                    .lane_frontier
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())[0] = Some(done);
                return Outcome::Accepted(api::MultiQueueReceipt {
                    work: api::SubmissionReceipt {
                        device: plan.id().device,
                        index: None,
                        accepted: vec![step],
                        checkpoints,
                        presents: Vec::new(),
                    },
                    signals: Vec::new(),
                });
            }
            if plan.steps().is_empty() && !matches!(packet.work.completion, CompletionRequest::None)
            {
                return Outcome::Rejected {
                    error: api::Error::Unsupported("Metal empty multi-queue completion"),
                    packet,
                };
            }
            if !packet.waits.is_empty() || !packet.signals.is_empty() {
                return Outcome::Rejected {
                    error: api::Error::Unsupported("Metal cross-packet GPU signals"),
                    packet,
                };
            }
            let mut present_count = 0;
            for (i, bound) in packet.work.steps.iter().enumerate() {
                let desc = &plan.steps()[i];
                if desc.lane.ordinal > 1 {
                    return Outcome::Rejected {
                        error: api::Error::WrongLane,
                        packet,
                    };
                }
                let records = match &bound.payload {
                    StepPayload::Commands(records) => records.as_slice(),
                    StepPayload::Present(_) if desc.lane.ordinal == 0 => {
                        present_count += 1;
                        if present_count > 1 {
                            return Outcome::Rejected {
                                error: api::Error::Unsupported(
                                    "Metal multi-queue multiple presents",
                                ),
                                packet,
                            };
                        }
                        &[]
                    }
                    // Metal 4 event waits can time out and let mappings run
                    // before their producer finishes. Reject before handoff until
                    // mapping submission is gated by host-observed success.
                    StepPayload::SparseBind(_) => {
                        return Outcome::Rejected {
                            error: api::Error::Unsupported("Metal sparse mapping synchronization"),
                            packet,
                        };
                    }
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
                    if desc.waits.len() > work.wait_slots.len() {
                        return Outcome::Rejected {
                            error: api::Error::Unsupported("Metal command wait slot limit"),
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
            let has_sparse = sparse.iter().any(|items| !items.is_empty());
            let sparse_queue = if has_sparse {
                match device._state._device.newMTL4CommandQueue() {
                    Some(queue) => Some(queue),
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
            let mut events = Vec::with_capacity(plan.steps().len());
            let mut markers = Vec::with_capacity(plan.steps().len());
            let mut waits = Vec::with_capacity(plan.steps().len());
            let mut sparse_bridges = Vec::with_capacity(plan.steps().len());
            for desc in plan.steps() {
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
            for (i, desc) in plan.steps().iter().enumerate() {
                for edge in &desc.waits {
                    let event =
                        objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(
                            &*events[edge.previous.index as usize],
                        );
                    waits[i].encodeWaitForEvent_value(event, 1);
                }
                let event = objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(
                    &*events[i],
                );
                markers[i].encodeSignalEvent_value(event, 2);
                if let StepPayload::Commands(records) = &packet.work.steps[i].payload {
                    for record in records {
                        // SAFETY: packet provenance was checked during preflight.
                        let work = unsafe { record.native_ref() };
                        for (slot, edge) in work.wait_slots.iter().zip(&desc.waits) {
                            let event = objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(&**slot);
                            markers[edge.previous.index as usize].encodeSignalEvent_value(event, 2);
                        }
                    }
                }
                if let StepPayload::Present(binding) = &packet.work.steps[i].payload {
                    let command = &present_commands
                        .iter()
                        .find(|(step, _)| *step == packet.work.steps[i].step)
                        .expect("present preallocated")
                        .1;
                    for edge in &desc.waits {
                        let event =
                            objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(
                                &*events[edge.previous.index as usize],
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
            let step_completions: Vec<_> = plan
                .steps()
                .iter()
                .map(|_| super::submit::Completion::new())
                .collect();
            let mut last = [None, None];
            for (i, desc) in plan.steps().iter().enumerate() {
                last[desc.lane.ordinal as usize] = Some(i);
            }
            let completions: Vec<_> = (0..plan.steps().len())
                .filter(|&i| match &packet.work.completion {
                    CompletionRequest::None => false,
                    CompletionRequest::End => last.contains(&Some(i)),
                    CompletionRequest::Steps(steps) => steps.contains(&plan.step(i)),
                })
                .map(|i| (i, step_completions[i].clone()))
                .collect();
            let mut signal_points = Vec::with_capacity(packet.signals.len());
            for request in &packet.signals {
                let producer_index = request.producer.index as usize;
                let lane = plan.steps()[producer_index].lane;
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
                (0..plan.steps().len()).map(|_| Vec::new()).collect();
            let mut packet = packet;
            for bound in &mut packet.work.steps {
                if let StepPayload::Present(binding) = &mut bound.payload {
                    // SAFETY: all native allocations succeeded and only one handoff is allowed.
                    if let Err(error) = unsafe { binding.try_handoff() } {
                        return Outcome::Rejected { error, packet };
                    }
                }
            }
            for wait in packet.waits {
                let i = wait.signal.consumer().index as usize;
                // SAFETY: Core validated this single-use token for the consumer execution.
                let signal = unsafe { wait.signal.into_native() };
                let event = objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(
                    &*signal.event,
                );
                waits[i].encodeWaitForEvent_value(event, 1);
                external[i].push(signal.producer);
            }
            for (completion, event) in step_completions.iter().zip(events.iter()) {
                std::sync::Arc::new(ReleaseOnFailure {
                    completion: completion.clone(),
                    event: event.clone(),
                })
                .observe();
            }
            let api::SubmitPacket {
                plan,
                steps,
                completion: _,
            } = packet.work;
            let mut accepted = Vec::new();
            let mut checkpoints = Vec::new();
            let mut lane_frontier = device
                ._state
                .lane_frontier
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone();
            for (i, bound) in steps.into_iter().enumerate() {
                let desc = &plan.steps()[i];
                let mut covered = Vec::new();
                covered.extend(external[i].iter().cloned());
                if let Some(previous) = &lane_frontier[desc.lane.ordinal as usize] {
                    covered.push(previous.clone());
                }
                for edge in &desc.waits {
                    covered.push(step_completions[edge.previous.index as usize].clone());
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
                    for (j, slot) in work.wait_slots.iter().enumerate() {
                        if let Some(edge) = desc.waits.get(j) {
                            std::sync::Arc::new(ReleaseOnFailure {
                                completion: step_completions[edge.previous.index as usize].clone(),
                                event: slot.clone(),
                            })
                            .observe();
                        } else {
                            slot.setSignaledValue(1);
                        }
                    }
                    let query_commit = super::query::commit_ops(&work.query_ops, work.lane);
                    work.retain_indirect_arguments_until_completed();
                    covered.push(super::submit::track_command(
                        device,
                        &work.command,
                        Some((query_commit, work.lane)),
                    ));
                    work.command.commit();
                }
                for (step, command) in &present_commands {
                    if *step == bound.step {
                        covered.push(super::submit::track_command(device, command, None));
                        command.commit();
                    }
                }
                if let Some((ready, done, bridge)) = &sparse_bridges[i] {
                    let queue = sparse_queue.as_ref().expect("sparse queue prepared");
                    let ready =
                        objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(
                            &**ready,
                        );
                    let done_event =
                        objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(
                            &**done,
                        );
                    queue.waitForEvent_value(ready, 1);
                    for bind in &sparse[i] {
                        bind.enqueue(queue);
                    }
                    let retained = std::mem::take(&mut sparse[i]);
                    let retained_queue = queue.clone();
                    let retained_event = done.clone();
                    let callback = block2::RcBlock::new(
                        move |_: std::ptr::NonNull<
                            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>,
                        >,
                              _: u64| {
                            let _ = (&retained, &retained_queue, &retained_event);
                        },
                    );
                    let listener = objc2_metal::MTLSharedEventListener::new();
                    // SAFETY: Metal copies the block and holds the native objects
                    // until the mapping queue actually signals completion. On
                    // device failure an unsignaled event intentionally retains them.
                    unsafe {
                        done.notifyListener_atValue_block(
                            &listener,
                            1,
                            block2::RcBlock::as_ptr(&callback).cast(),
                        )
                    };
                    queue.signalEvent_value(done_event, 1);
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
                    let lane = plan.steps()[i].lane;
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
