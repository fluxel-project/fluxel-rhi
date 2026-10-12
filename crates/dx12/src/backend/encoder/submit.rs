use super::*;
impl api::CompletionBackend for Dx12 {
    type WaitFuture<'a>
        = Ready<Result<(), Error>>
    where
        Self: 'a;
    fn completion_status(device: &Dx12Device, point: &Dx12Completion) -> CompletionState {
        // GetCompletedValue is native queue evidence; no host-side counter is
        // treated as completion.
        if unsafe { point.fence.GetCompletedValue() } >= point.value {
            retire(device);
            CompletionState::Complete
        } else {
            CompletionState::Pending
        }
    }
    fn wait_completion<'a>(
        device: &'a Dx12Device,
        point: &'a Dx12Completion,
    ) -> Self::WaitFuture<'a> {
        let result = wait_fence(device, point);
        if result.is_ok() {
            retire(device);
        }
        ready(result)
    }
    fn wait_idle(device: &Dx12Device) -> Self::WaitFuture<'_> {
        let value = device
            .0
            .next_fence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let result = unsafe { device.0.queue.Signal(&device.0.fence, value) }
            .map_err(native_error)
            .and_then(|_| {
                wait_fence(
                    device,
                    &Dx12Completion {
                        fence: device.0.fence.clone(),
                        value,
                    },
                )
            });
        if result.is_ok() {
            retire(device);
        }
        ready(result)
    }
}

fn retire(device: &Dx12Device) {
    let completed = unsafe { device.0.fence.GetCompletedValue() };
    device
        .0
        .in_flight
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .retain(|(value, _)| *value == u64::MAX || *value > completed);
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

fn wait_fence(_: &Dx12Device, point: &Dx12Completion) -> Result<(), Error> {
    if unsafe { point.fence.GetCompletedValue() } >= point.value {
        return Ok(());
    }
    let event = unsafe { CreateEventW(None, false, false, None) }.map_err(native_error)?;
    let result = unsafe { point.fence.SetEventOnCompletion(point.value, event) }
        .map_err(native_error)
        .and_then(|_| {
            if unsafe { WaitForSingleObject(event, u32::MAX) } == WAIT_OBJECT_0 {
                Ok(())
            } else {
                Err(Error::DeviceLost)
            }
        });
    unsafe {
        let _ = CloseHandle(event);
    }
    result
}

impl DeferredBackend for Dx12 {
    type NativeRecordedWork = Dx12RecordedWork;
    fn finish_encoder(mut encoder: Dx12Encoder, lane: LaneId) -> Result<Dx12RecordedWork, Error> {
        unsafe { encoder.list.Close() }.map_err(native_error)?;
        encoder.closed = true;
        Ok(Dx12RecordedWork { lane, encoder })
    }
    unsafe fn submit(device: &Dx12Device, packet: SubmitPacket<Self>) -> SubmitOutcome<Self> {
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
                let value = device
                    .0
                    .next_fence
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if let Err(error) =
                    unsafe { device.0.queue.Signal(&device.0.fence, value) }.map_err(native_error)
                {
                    retain_indeterminate(device, issued);
                    return SubmitOutcome::Indeterminate {
                        error,
                        uncertain: unsafe {
                            api::NativeUncertainty::from_native(plan.id().device, ())
                        },
                    };
                }
                checkpoints.push(checkpoint(device, plan.id().device, value, &lane_frontiers));
                last_signal = Some((id, value));
            }
        }
        // Each submission has a final fence value for allocator/upload
        // retention, even when the caller requested no completion evidence.
        // Reuse a signal made at the final accepted step when possible.
        let final_step = accepted.last().copied();
        let retention_value = if final_step.is_some()
            && last_signal.map(|(step, _)| step) == final_step
        {
            last_signal.map(|(_, value)| value)
        } else if final_step.is_some() {
            let value = device
                .0
                .next_fence
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if let Err(error) =
                unsafe { device.0.queue.Signal(&device.0.fence, value) }.map_err(native_error)
            {
                retain_indeterminate(device, issued);
                return SubmitOutcome::Indeterminate {
                    error,
                    uncertain: unsafe { api::NativeUncertainty::from_native(plan.id().device, ()) },
                };
            }
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
