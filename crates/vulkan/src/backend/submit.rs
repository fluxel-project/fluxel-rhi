//! Native Vulkan submission and fence-owned command lifetimes.

use super::{DeviceState, Vulkan, VulkanDevice, VulkanEncoder, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{
    self, CompletionState, DeferredBackend, Error, Result, RetryableSubmit, StepPayload,
    SubmitOutcome,
};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

pub struct RecordedWork {
    pub(super) encoder: VulkanEncoder,
}
impl RecordedWork {
    fn command(&self) -> vk::CommandBuffer {
        self.encoder.command
    }
}

struct WaitState {
    result: Option<Result<()>>,
    wakers: Vec<Waker>,
}

#[derive(Clone)]
pub struct Completion(Arc<FenceState>);
struct FenceState {
    device: Arc<DeviceState>,
    fence: vk::Fence,
    wait: Mutex<WaitState>,
}
impl Drop for FenceState {
    fn drop(&mut self) {
        unsafe { self.device.device.destroy_fence(self.fence, None) };
    }
}
impl Completion {
    fn new(device: Arc<DeviceState>, fence: vk::Fence) -> Self {
        Self(Arc::new(FenceState {
            device,
            fence,
            wait: Mutex::new(WaitState {
                result: None,
                wakers: Vec::new(),
            }),
        }))
    }

    fn publish(&self, result: Result<()>) {
        let wakers = {
            let mut wait = self.0.wait.lock().unwrap_or_else(|p| p.into_inner());
            wait.result = Some(result);
            std::mem::take(&mut wait.wakers)
        };
        for waker in wakers {
            waker.wake();
        }
    }

    fn result(&self) -> Option<Result<()>> {
        self.0
            .wait
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .result
            .clone()
    }
}

pub struct FenceFuture {
    completion: Option<Completion>,
    immediate: Option<Result<()>>,
}
impl FenceFuture {
    fn pending(completion: Completion) -> Self {
        Self {
            completion: Some(completion),
            immediate: None,
        }
    }
    fn immediate(result: Result<()>) -> Self {
        Self {
            completion: None,
            immediate: Some(result),
        }
    }
}
impl Future for FenceFuture {
    type Output = Result<()>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if let Some(result) = &this.immediate {
            return Poll::Ready(result.clone());
        }
        let completion = this
            .completion
            .as_ref()
            .expect("completion future has a fence");
        let mut wait = completion.0.wait.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(result) = &wait.result {
            Poll::Ready(result.clone())
        } else {
            if !wait.wakers.iter().any(|waker| waker.will_wake(cx.waker())) {
                wait.wakers.push(cx.waker().clone());
            }
            Poll::Pending
        }
    }
}

// The waiter is only a native completion observer and owner of already issued
// command pools/uploads. It never schedules GPU commands or replays CPU work.
fn retain_until_fence(completion: Completion, work: Vec<RecordedWork>) {
    let retained = Arc::new(Mutex::new(Some(work)));
    let worker_retained = retained.clone();
    let worker_completion = completion.clone();
    let spawned = std::thread::Builder::new()
        .name("fluxel-vulkan-fence".into())
        .spawn(move || {
            let result = unsafe {
                worker_completion.0.device.device.wait_for_fences(
                    &[worker_completion.0.fence],
                    true,
                    u64::MAX,
                )
            }
            .map_err(vk_error)
            .and_then(|()| {
                let guard = worker_retained.lock().unwrap_or_else(|p| p.into_inner());
                super::capture::write_completed(guard.as_deref().unwrap_or(&[]))
            });
            worker_retained
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .take();
            worker_completion.publish(result);
        });
    if spawned.is_err() {
        // Thread creation failure is exceptional. Wait here so a failed spawn
        // cannot release command pools or staging while the GPU still uses them.
        let result = unsafe {
            completion
                .0
                .device
                .device
                .wait_for_fences(&[completion.0.fence], true, u64::MAX)
        }
        .map_err(vk_error)
        .and_then(|()| {
            let guard = retained.lock().unwrap_or_else(|p| p.into_inner());
            super::capture::write_completed(guard.as_deref().unwrap_or(&[]))
        });
        retained.lock().unwrap_or_else(|p| p.into_inner()).take();
        completion.publish(result);
    }
}

impl api::CompletionBackend for Vulkan {
    type WaitFuture<'a> = FenceFuture;

    fn completion_status(_: &VulkanDevice, point: &Completion) -> CompletionState {
        match point.result() {
            Some(Ok(())) => CompletionState::Complete,
            Some(Err(error)) => CompletionState::Failed(error),
            None => match unsafe { point.0.device.device.get_fence_status(point.0.fence) } {
                Ok(true) => CompletionState::Complete,
                Ok(false) => CompletionState::Pending,
                Err(error) => CompletionState::Failed(vk_error(error)),
            },
        }
    }

    fn wait_completion<'a>(_: &'a VulkanDevice, point: &'a Completion) -> Self::WaitFuture<'a> {
        FenceFuture::pending(point.clone())
    }

    fn wait_idle(device: &VulkanDevice) -> Self::WaitFuture<'_> {
        let fence = match unsafe {
            device
                .0
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => fence,
            Err(error) => return FenceFuture::immediate(Err(vk_error(error))),
        };
        let completion = Completion::new(device.0.clone(), fence);
        let result = {
            let _queue = device
                .0
                .queue_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            unsafe {
                device
                    .0
                    .device
                    .queue_submit(device.0.graphics_queue, &[], fence)
            }
        };
        if let Err(error) = result {
            return FenceFuture::immediate(Err(vk_error(error)));
        }
        retain_until_fence(completion.clone(), Vec::new());
        FenceFuture::pending(completion)
    }
}

impl DeferredBackend for Vulkan {
    type NativeRecordedWork = RecordedWork;

    fn finish_encoder(encoder: VulkanEncoder, _: api::LaneId) -> Result<RecordedWork> {
        if encoder.raster {
            return Err(Error::InvalidState);
        }
        unsafe { encoder.device.device.end_command_buffer(encoder.command) }.map_err(vk_error)?;
        Ok(RecordedWork { encoder })
    }

    unsafe fn submit(
        device: &VulkanDevice,
        packet: api::SubmitPacket<Self>,
    ) -> SubmitOutcome<Self> {
        let supported = packet.steps.len() == 2
            && matches!(&packet.steps[0].payload, StepPayload::Commands(work) if !work.is_empty())
            && matches!(&packet.steps[1].payload, StepPayload::Present(_))
            && matches!(
                packet.completion,
                api::CompletionRequest::End | api::CompletionRequest::None
            )
            && packet.plan.steps()[0].lane == packet.plan.steps()[1].lane;
        if !supported {
            return SubmitOutcome::Rejected {
                error: Error::Unsupported("Vulkan single-lane commands then present submission"),
                work: RetryableSubmit::Fresh(packet),
            };
        }
        let fence = match unsafe {
            device
                .0
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => fence,
            Err(error) => {
                return SubmitOutcome::Rejected {
                    error: vk_error(error),
                    work: RetryableSubmit::Fresh(packet),
                };
            }
        };
        let render_done = match unsafe {
            device
                .0
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        } {
            Ok(semaphore) => semaphore,
            Err(error) => {
                unsafe { device.0.device.destroy_fence(fence, None) };
                return SubmitOutcome::Rejected {
                    error: vk_error(error),
                    work: RetryableSubmit::Fresh(packet),
                };
            }
        };
        if let StepPayload::Present(binding) = &packet.steps[1].payload {
            if let Err(error) =
                unsafe { binding.image().native_ref() }.set_present_wait(render_done)
            {
                unsafe { device.0.device.destroy_semaphore(render_done, None) };
                return SubmitOutcome::Rejected {
                    error,
                    work: RetryableSubmit::Fresh(packet),
                };
            }
        }
        let completion = Completion::new(device.0.clone(), fence);
        let api::SubmitPacket {
            plan,
            steps,
            completion: request,
        } = packet;
        let mut steps = steps.into_iter();
        let first = steps.next().expect("validated commands step");
        let second = steps.next().expect("validated present step");
        let command_id = first.step;
        let present_id = second.step;
        let StepPayload::Commands(recorded) = first.payload else {
            unreachable!()
        };
        let work: Vec<RecordedWork> = recorded
            .into_iter()
            .map(|item| unsafe { item.into_native() })
            .collect();
        let commands: Vec<vk::CommandBuffer> = work.iter().map(RecordedWork::command).collect();
        let signals = [render_done];
        let info = vk::SubmitInfo::default()
            .command_buffers(&commands)
            .signal_semaphores(&signals);
        let issued = {
            let _queue = device
                .0
                .queue_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            unsafe {
                device
                    .0
                    .device
                    .queue_submit(device.0.graphics_queue, &[info], fence)
            }
        };
        if let Err(error) = issued {
            // Vulkan may have crossed the queue boundary. Only release native
            // objects after the queue is known idle, even on the error route.
            if unsafe { device.0.device.queue_wait_idle(device.0.graphics_queue) }.is_err() {
                // There is no completion evidence. Keep command pools and
                // staging alive rather than releasing GPU-referenced memory.
                std::mem::forget(work);
            }
            return SubmitOutcome::Indeterminate {
                error: vk_error(error),
                uncertain: unsafe { api::NativeUncertainty::from_native(plan.id().device, ()) },
            };
        }
        retain_until_fence(completion.clone(), work);
        let StepPayload::Present(mut binding) = second.payload else {
            unreachable!()
        };
        if let Err(error) = unsafe { binding.try_handoff() } {
            return SubmitOutcome::Indeterminate {
                error,
                uncertain: unsafe { api::NativeUncertainty::from_native(plan.id().device, ()) },
            };
        }
        let checkpoints = if matches!(request, api::CompletionRequest::End) {
            vec![api::Checkpoint {
                point: unsafe { api::CompletionPoint::from_native(plan.id().device, completion) },
                covers: api::CompletionCoverage {
                    // The fence proves render submission completion. The
                    // present receipt separately proves handoff acceptance;
                    // this fence does not prove display timing.
                    lane_frontiers: vec![(plan.steps()[0].lane, command_id)],
                },
            }]
        } else {
            Vec::new()
        };
        SubmitOutcome::Accepted(api::SubmissionReceipt {
            device: plan.id().device,
            index: None,
            accepted: vec![command_id, present_id],
            checkpoints,
            presents: vec![api::PresentReceipt {
                step: Some(present_id),
                accepted: true,
            }],
        })
    }

    unsafe fn resume(_: &VulkanDevice, tail: api::DeferredTail<Self>) -> SubmitOutcome<Self> {
        SubmitOutcome::Rejected {
            error: Error::Unsupported("Vulkan partial submission resume"),
            work: RetryableSubmit::Tail(tail),
        }
    }
}
