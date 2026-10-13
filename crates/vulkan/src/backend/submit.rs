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
    sync::{
        Arc, Mutex, OnceLock,
        mpsc::{self, Sender},
    },
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

/// One bounded observer for submitted Vulkan fences.
///
/// It owns only fences and already-recorded native work.  In particular it
/// never chooses queue order, submits commands, or interprets recorded work.
/// A dropped `FenceFuture` leaves its `Completion` in this queue, so command
/// pools and upload allocations still remain alive until the GPU is finished.
struct FenceWaiter {
    sender: Sender<FenceJob>,
}

struct FenceJob {
    completion: Completion,
    work: Vec<RecordedWork>,
}

struct PendingPresentSignal {
    device: Arc<DeviceState>,
    semaphore: Option<vk::Semaphore>,
}
impl Drop for PendingPresentSignal {
    fn drop(&mut self) {
        if let Some(semaphore) = self.semaphore.take() {
            unsafe { self.device.device.destroy_semaphore(semaphore, None) };
        }
    }
}

static FENCE_WAITER: OnceLock<std::result::Result<FenceWaiter, String>> = OnceLock::new();

fn fence_waiter() -> Result<&'static FenceWaiter> {
    match FENCE_WAITER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<FenceJob>();
        std::thread::Builder::new()
            .name("fluxel-vulkan-fence-waiter".into())
            .spawn(move || {
                for job in receiver {
                    let result = unsafe {
                        job.completion.0.device.device.wait_for_fences(
                            &[job.completion.0.fence],
                            true,
                            u64::MAX,
                        )
                    }
                    .map_err(vk_error);
                    // Completion represents GPU evidence only.  Optional
                    // visual capture is test instrumentation and must never
                    // turn completed GPU work into a failed RHI completion.
                    let completed = result.is_ok();
                    job.completion.publish(result);
                    if completed {
                        let _ = super::capture::write_completed(&job.work);
                    }
                    drop(job.work);
                }
            })
            .map(|_| FenceWaiter { sender })
            .map_err(|error| format!("Vulkan fence waiter: {error}"))
    }) {
        Ok(waiter) => Ok(waiter),
        Err(message) => Err(Error::Backend(message.clone())),
    }
}

// The waiter is only a native completion observer and owner of already issued
// command pools/uploads. It never schedules GPU commands or replays CPU work.
fn retain_until_fence(completion: Completion, work: Vec<RecordedWork>) {
    let job = FenceJob {
        completion: completion.clone(),
        work,
    };
    match fence_waiter() {
        Ok(waiter) => match waiter.sender.send(job) {
            Ok(()) => {}
            Err(error) => {
                // A native submission may already reference this work. Never
                // drop it on an observer failure; leaking this exceptional
                // ownership is safer than a synchronous host wait or an early
                // Vulkan command-pool destruction.
                std::mem::forget(error.0);
                completion.publish(Err(Error::Backend(
                    "Vulkan fence waiter unavailable".into(),
                )));
            }
        },
        Err(error) => {
            std::mem::forget(job);
            completion.publish(Err(error));
        }
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
            let _submit = device
                .0
                .submit_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
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
        // A Vulkan queue is already an ordered executor.  The Core plan has
        // chosen the order and synchronization; this backend only lowers its
        // one graphics-lane prefix and never reconstructs a dependency graph.
        let has_commands = packet
            .steps
            .iter()
            .any(|step| matches!(&step.payload, StepPayload::Commands(work) if !work.is_empty()));
        let plan_shape = has_commands
            && packet
                .steps
                .iter()
                .enumerate()
                .all(|(index, bound)| match &bound.payload {
                    StepPayload::Commands(_) => {
                        packet.plan.steps()[index].lane == packet.plan.steps()[0].lane
                    }
                    StepPayload::Present(_) => {
                        index + 1 == packet.steps.len()
                            && packet.plan.steps()[index].lane == packet.plan.steps()[0].lane
                    }
                    StepPayload::SparseBind(_) => false,
                })
            && packet
                .steps
                .iter()
                .filter(|step| matches!(&step.payload, StepPayload::Present(_)))
                .count()
                <= 1;
        if !plan_shape {
            return SubmitOutcome::Rejected {
                error: Error::Unsupported("Vulkan single graphics-queue submission plan"),
                work: RetryableSubmit::Fresh(packet),
            };
        }
        let requested = match &packet.completion {
            api::CompletionRequest::Steps(steps) => steps.as_ref(),
            _ => &[],
        };
        // Allocate every native synchronization object before issuing any
        // queue operation.  Allocation failure remains a clean retryable
        // rejection, rather than creating a half-lowered packet.
        let mut step_fences = (0..packet.steps.len())
            .map(|_| None)
            .collect::<Vec<Option<Completion>>>();
        for step in requested {
            if step_fences[step.index as usize].is_some() {
                continue;
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
            step_fences[step.index as usize] = Some(Completion::new(device.0.clone(), fence));
        }
        let terminal_fence = match unsafe {
            device
                .0
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => Completion::new(device.0.clone(), fence),
            Err(error) => {
                return SubmitOutcome::Rejected {
                    error: vk_error(error),
                    work: RetryableSubmit::Fresh(packet),
                };
            }
        };
        let present_signal = if packet
            .steps
            .iter()
            .any(|step| matches!(&step.payload, StepPayload::Present(_)))
        {
            match unsafe {
                device
                    .0
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            } {
                Ok(semaphore) => Some(semaphore),
                Err(error) => {
                    return SubmitOutcome::Rejected {
                        error: vk_error(error),
                        work: RetryableSubmit::Fresh(packet),
                    };
                }
            }
        } else {
            None
        };
        let mut present_signal = present_signal.map(|semaphore| PendingPresentSignal {
            device: device.0.clone(),
            semaphore: Some(semaphore),
        });
        if let Err(error) = fence_waiter() {
            return SubmitOutcome::Rejected {
                error,
                work: RetryableSubmit::Fresh(packet),
            };
        }
        let api::SubmitPacket {
            plan,
            steps,
            completion: request,
        } = packet;
        let mut issued = Vec::new();
        let mut accepted = Vec::new();
        let mut presents = Vec::new();
        let mut checkpoints = Vec::new();
        let mut lane_frontiers = Vec::new();
        // Hold the plan-level gate from the first queue issue through its
        // terminal marker.  Each native queue call still takes `queue_gate`,
        // so Surface present and wait-idle retain their narrow external
        // synchronization boundary without allowing another plan to splice
        // commands between this plan's explicit transitions.
        let _submit = device
            .0
            .submit_gate
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let indeterminate = |error, issued: Vec<RecordedWork>| {
            // vkQueueSubmit may have crossed the driver boundary even on an
            // error.  Keep command pools and their uploads alive unless an
            // externally synchronized queue-idle probe proves retirement.
            let idle = {
                let _queue = device
                    .0
                    .queue_gate
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                unsafe { device.0.device.queue_wait_idle(device.0.graphics_queue) }
            };
            if idle.is_err() {
                std::mem::forget(issued);
            }
            SubmitOutcome::Indeterminate {
                error,
                uncertain: unsafe { api::NativeUncertainty::from_native(plan.id().device, ()) },
            }
        };

        for bound in steps {
            let id = bound.step;
            match bound.payload {
                StepPayload::Commands(records) => {
                    let start = issued.len();
                    issued.extend(
                        records
                            .into_iter()
                            .map(|record| unsafe { record.into_native() }),
                    );
                    let commands = issued[start..]
                        .iter()
                        .map(RecordedWork::command)
                        .collect::<Vec<_>>();
                    if !commands.is_empty() {
                        let info = vk::SubmitInfo::default().command_buffers(&commands);
                        let result = {
                            let _queue = device
                                .0
                                .queue_gate
                                .lock()
                                .unwrap_or_else(|p| p.into_inner());
                            unsafe {
                                device.0.device.queue_submit(
                                    device.0.graphics_queue,
                                    &[info],
                                    vk::Fence::null(),
                                )
                            }
                        };
                        if let Err(error) = result {
                            return indeterminate(vk_error(error), issued);
                        }
                    }
                    accepted.push(id);
                }
                StepPayload::Present(mut binding) => {
                    let pending = present_signal
                        .as_mut()
                        .expect("validated one Vulkan present");
                    let signal = pending.semaphore.expect("present signal is still owned");
                    if let Err(error) =
                        unsafe { binding.image().native_ref() }.set_present_wait(signal)
                    {
                        return indeterminate(error, issued);
                    }
                    // The surface image now owns the semaphore even if queue
                    // submission or handoff later fails.
                    pending.semaphore = None;
                    let signals = [signal];
                    let info = vk::SubmitInfo::default().signal_semaphores(&signals);
                    let result = {
                        let _queue = device
                            .0
                            .queue_gate
                            .lock()
                            .unwrap_or_else(|p| p.into_inner());
                        unsafe {
                            device.0.device.queue_submit(
                                device.0.graphics_queue,
                                &[info],
                                vk::Fence::null(),
                            )
                        }
                    };
                    if let Err(error) = result {
                        return indeterminate(vk_error(error), issued);
                    }
                    if let Err(error) = unsafe { binding.try_handoff() } {
                        return indeterminate(error, issued);
                    }
                    accepted.push(id);
                    presents.push(api::PresentReceipt {
                        step: Some(id),
                        accepted: true,
                    });
                }
                StepPayload::SparseBind(_) => unreachable!("submission shape was validated"),
            }

            if matches!(
                plan.steps()[id.index as usize].kind,
                api::StepKind::Commands
            ) {
                let lane = plan.steps()[id.index as usize].lane;
                lane_frontiers.retain(|(existing, _)| *existing != lane);
                lane_frontiers.push((lane, id));
            }
            if let Some(completion) = step_fences[id.index as usize].take() {
                let result = {
                    let _queue = device
                        .0
                        .queue_gate
                        .lock()
                        .unwrap_or_else(|p| p.into_inner());
                    unsafe {
                        device.0.device.queue_submit(
                            device.0.graphics_queue,
                            &[],
                            completion.0.fence,
                        )
                    }
                };
                if let Err(error) = result {
                    return indeterminate(vk_error(error), issued);
                }
                checkpoints.push(api::Checkpoint {
                    point: unsafe {
                        api::CompletionPoint::from_native(plan.id().device, completion.clone())
                    },
                    covers: api::CompletionCoverage {
                        lane_frontiers: lane_frontiers.clone(),
                    },
                });
                retain_until_fence(completion, Vec::new());
            }
        }

        // A terminal fence is needed even for CompletionRequest::None: it is
        // the native proof that command pools, uploads and descriptors may be
        // released.  Its marker follows every accepted queue operation.
        let terminal = {
            let _queue = device
                .0
                .queue_gate
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            unsafe {
                device
                    .0
                    .device
                    .queue_submit(device.0.graphics_queue, &[], terminal_fence.0.fence)
            }
        };
        if let Err(error) = terminal {
            return indeterminate(vk_error(error), issued);
        }
        retain_until_fence(terminal_fence.clone(), issued);
        if matches!(request, api::CompletionRequest::End) {
            checkpoints.push(api::Checkpoint {
                point: unsafe {
                    api::CompletionPoint::from_native(plan.id().device, terminal_fence)
                },
                covers: api::CompletionCoverage { lane_frontiers },
            });
        }
        SubmitOutcome::Accepted(api::SubmissionReceipt {
            device: plan.id().device,
            index: None,
            accepted,
            checkpoints,
            presents,
        })
    }

    unsafe fn resume(_: &VulkanDevice, tail: api::DeferredTail<Self>) -> SubmitOutcome<Self> {
        SubmitOutcome::Rejected {
            error: Error::Unsupported("Vulkan partial submission resume"),
            work: RetryableSubmit::Tail(tail),
        }
    }
}
