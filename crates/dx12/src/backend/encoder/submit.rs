use super::*;
impl api::CompletionBackend for Dx12 {
    type WaitFuture<'a>
        = super::fence_wait::Dx12FenceFuture
    where
        Self: 'a;
    fn completion_status(device: &Dx12Device, point: &Dx12Completion) -> CompletionState {
        // GetCompletedValue is native queue evidence; no host-side counter is
        // treated as completion.
        match super::fence_wait::fence_progress(
            unsafe { point.fence.GetCompletedValue() },
            point.value,
        ) {
            Err(error) => CompletionState::Failed(error),
            Ok(true) => {
                retire(device);
                CompletionState::Complete
            }
            Ok(false) => CompletionState::Pending,
        }
    }
    fn wait_completion<'a>(
        device: &'a Dx12Device,
        point: &'a Dx12Completion,
    ) -> Self::WaitFuture<'a> {
        super::fence_wait::Dx12FenceFuture::new(device.clone(), point.fence.clone(), point.value)
    }
    fn wait_idle(device: &Dx12Device) -> Self::WaitFuture<'_> {
        let value = match signal_queue(device) {
            Ok(value) => value,
            Err(error) => {
                return super::fence_wait::Dx12FenceFuture::immediate(
                    device.clone(),
                    device.0.fence.clone(),
                    0,
                    Err(error),
                );
            }
        };
        super::fence_wait::Dx12FenceFuture::new(device.clone(), device.0.fence.clone(), value)
    }
}

pub(super) fn retire(device: &Dx12Device) {
    let completed = unsafe { device.0.fence.GetCompletedValue() };
    device
        .0
        .abandoned_fence_events
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .retain(|event| completed != u64::MAX && event.value > completed);
    if completed == u64::MAX {
        return;
    }
    device
        .0
        .in_flight
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .retain(|(value, _)| retain_after(completed, *value));
}

fn retain_after(completed: u64, value: u64) -> bool {
    completed == u64::MAX || value == u64::MAX || value > completed
}

// The queue gate serializes each fence allocation with its native Signal.
// u64::MAX is reserved for device removal and never used as ordinary
// completion evidence.
fn signal_queue(device: &Dx12Device) -> Result<u64, Error> {
    let _queue = device
        .0
        .queue_lock
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let value = device
        .0
        .next_fence
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |next| next.checked_add(1),
        )
        .map_err(|_| Error::InvalidRange)?;
    unsafe { device.0.queue.Signal(&device.0.fence, value) }.map_err(native_error)?;
    Ok(value)
}

fn checkpoint(
    device: &Dx12Device,
    device_id: api::DeviceId,
    value: u64,
    lane_frontiers: &[(LaneId, api::StepId)],
) -> api::Checkpoint<Dx12> {
    api::Checkpoint {
        point: unsafe {
            CompletionPoint::from_native(
                device_id,
                Dx12Completion {
                    fence: device.0.fence.clone(),
                    value,
                },
            )
        },
        covers: CompletionCoverage {
            lane_frontiers: lane_frontiers.to_vec(),
        },
    }
}

fn retain_indeterminate(device: &Dx12Device, issued: Vec<Dx12RecordedWork>) {
    if !issued.is_empty() {
        device
            .0
            .in_flight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((u64::MAX, issued));
    }
}

impl DeferredBackend for Dx12 {
    type NativeRecordedWork = Dx12RecordedWork;
    fn finish_encoder(mut encoder: Dx12Encoder, lane: LaneId) -> Result<Dx12RecordedWork, Error> {
        unsafe { encoder.list.Close() }.map_err(native_error)?;
        encoder.closed = true;
        Ok(Dx12RecordedWork { lane, encoder })
    }
    unsafe fn submit(device: &Dx12Device, packet: SubmitPacket<Self>) -> SubmitOutcome<Self> {
        // Completion records from a prior submission can be released without
        // entering the native queue gate.
        retire(device);
        let SubmitPacket {
            plan,
            steps,
            completion,
        } = packet;
        let mut accepted = Vec::new();
        let mut issued = Vec::new();
        let mut presents = Vec::new();
        let mut lane_frontiers = Vec::new();
        let requested = match &completion {
            api::CompletionRequest::Steps(steps) => steps.as_ref(),
            _ => &[],
        };
        let mut checkpoints = Vec::new();
        let mut last_signal = None;
        for step in steps {
            let id = step.step;
            match step.payload {
                StepPayload::Commands(work) => {
                    let mut lists = Vec::with_capacity(work.len());
                    let mut lane = None;
                    for record in work {
                        let native = unsafe { record.into_native() };
                        if !native.encoder.closed {
                            device
                                .0
                                .in_flight
                                .lock()
                                .unwrap_or_else(|p| p.into_inner())
                                .push((u64::MAX, issued));
                            return SubmitOutcome::Indeterminate {
                                error: Error::InvalidState,
                                uncertain: unsafe {
                                    api::NativeUncertainty::from_native(plan.id().device, ())
                                },
                            };
                        }
                        lane = Some(native.lane);
                        lists.push(
                            native
                                .encoder
                                .list
                                .cast::<ID3D12CommandList>()
                                .expect("graphics command list is command list"),
                        );
                        issued.push(native);
                    }
                    if !lists.is_empty() {
                        // Serialize only the native queue operation. Do not
                        // carry this lock into Present: DXGI Present may wait
                        // for vblank, while Execute/Signal still need a
                        // single total order across submitters.
                        let _queue = device
                            .0
                            .queue_lock
                            .lock()
                            .unwrap_or_else(|p| p.into_inner());
                        unsafe {
                            device.0.queue.ExecuteCommandLists(
                                &lists.into_iter().map(Some).collect::<Vec<_>>(),
                            );
                        }
                    }
                    accepted.push(id);
                    let lane = lane.unwrap_or(plan.steps()[id.index as usize].lane);
                    lane_frontiers.retain(|(existing, _)| *existing != lane);
                    lane_frontiers.push((lane, id));
                }
                StepPayload::Present(mut binding) => {
                    if accepted.is_empty() {
                        return SubmitOutcome::Indeterminate {
                            error: Error::InvalidState,
                            uncertain: unsafe {
                                api::NativeUncertainty::from_native(plan.id().device, ())
                            },
                        };
                    }
                    if let Err(error) = unsafe { binding.try_handoff() } {
                        device
                            .0
                            .in_flight
                            .lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .push((u64::MAX, issued));
                        return SubmitOutcome::Indeterminate {
                            error,
                            uncertain: unsafe {
                                api::NativeUncertainty::from_native(plan.id().device, ())
                            },
                        };
                    }
                    accepted.push(id);
                    presents.push(PresentReceipt {
                        step: Some(id),
                        accepted: true,
                    });
                    let lane = plan.steps()[id.index as usize].lane;
                    lane_frontiers.retain(|(existing, _)| *existing != lane);
                    lane_frontiers.push((lane, id));
                }
                StepPayload::SparseBind(_) => {
                    if !issued.is_empty() {
                        device
                            .0
                            .in_flight
                            .lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .push((u64::MAX, issued));
                    }
                    return SubmitOutcome::Indeterminate {
                        error: Error::Unsupported("DX12 sparse bind"),
                        uncertain: unsafe {
                            api::NativeUncertainty::from_native(plan.id().device, ())
                        },
                    };
                }
            }
            if requested.contains(&id) {
                let value = match signal_queue(device) {
                    Ok(value) => value,
                    Err(error) => {
                        retain_indeterminate(device, issued);
                        return SubmitOutcome::Indeterminate {
                            error,
                            uncertain: unsafe {
                                api::NativeUncertainty::from_native(plan.id().device, ())
                            },
                        };
                    }
                };
                checkpoints.push(checkpoint(device, plan.id().device, value, &lane_frontiers));
                last_signal = Some((id, value));
            }
        }
        // Each submission has a final fence value for allocator/upload
        // retention, even when the caller requested no completion evidence.
        // Reuse a signal made at the final accepted step when possible.
        let final_step = accepted.last().copied();
        let retention_value =
            if final_step.is_some() && last_signal.map(|(step, _)| step) == final_step {
                last_signal.map(|(_, value)| value)
            } else if final_step.is_some() {
                let value = match signal_queue(device) {
                    Ok(value) => value,
                    Err(error) => {
                        retain_indeterminate(device, issued);
                        return SubmitOutcome::Indeterminate {
                            error,
                            uncertain: unsafe {
                                api::NativeUncertainty::from_native(plan.id().device, ())
                            },
                        };
                    }
                };
                Some(value)
            } else {
                None
            };
        if let Some(value) = retention_value {
            device
                .0
                .in_flight
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push((value, issued));
            retire(device);
        }
        match completion {
            api::CompletionRequest::None => checkpoints.clear(),
            api::CompletionRequest::End => {
                checkpoints.clear();
                if let Some(value) = retention_value {
                    checkpoints.push(checkpoint(device, plan.id().device, value, &lane_frontiers));
                }
            }
            api::CompletionRequest::Steps(_) => {}
        }
        SubmitOutcome::Accepted(SubmissionReceipt {
            device: plan.id().device,
            index: None,
            accepted,
            checkpoints,
            presents,
        })
    }
    unsafe fn resume(_: &Dx12Device, tail: api::DeferredTail<Self>) -> SubmitOutcome<Self> {
        SubmitOutcome::Rejected {
            error: Error::Unsupported("DX12 submit resume"),
            work: api::RetryableSubmit::Tail(tail),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::retain_after;

    #[test]
    fn device_removed_keeps_all_submitted_native_owners() {
        assert!(retain_after(u64::MAX, 1));
        assert!(retain_after(u64::MAX, u64::MAX));
        assert!(retain_after(10, u64::MAX));
        assert!(retain_after(10, 11));
        assert!(!retain_after(10, 10));
    }
}
