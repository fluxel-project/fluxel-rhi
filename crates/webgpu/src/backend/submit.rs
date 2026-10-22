//! One-shot native command buffers and browser queue completion.
//!
//! Finished command buffers are submitted once. Their JS objects stay alive in
//! `in_flight` until `onSubmittedWorkDone` settles; loss and validation failure
//! settle the same completion path, so a caller never observes a fake success.

use std::{
    cell::RefCell,
    future::Future,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll, Waker},
};

use fluxel_rhi_core::api::{self, Error, Result};
use js_sys::{Array, Promise};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::{JsFuture, spawn_local};

use super::{
    command::WebGpuEncoder,
    instance::{WebGpu, WebGpuDevice},
    js,
};

pub struct RecordedWork {
    command_buffer: JsValue,
    keep_alive: Vec<JsValue>,
}

#[derive(Clone)]
pub struct Completion(Rc<RefCell<CompletionInner>>);

struct CompletionInner {
    state: api::CompletionState,
    wakers: Vec<Waker>,
}

pub(super) struct InFlight {
    pub(super) completion: Completion,
    _resources: Vec<JsValue>,
}

impl Completion {
    fn new(promise: Promise, device: WebGpuDevice) -> Self {
        let result = Self(Rc::new(RefCell::new(CompletionInner {
            state: api::CompletionState::Pending,
            wakers: Vec::new(),
        })));
        let pending = result.clone();
        spawn_local(async move {
            let state = match JsFuture::from(promise).await {
                Ok(_) if device.0.lost.get() => api::CompletionState::Failed(Error::DeviceLost),
                Ok(_) => match device.0.validation_error.borrow().clone() {
                    Some(message) => api::CompletionState::Failed(Error::Backend(message)),
                    None => api::CompletionState::Complete,
                },
                Err(error) => api::CompletionState::Failed(Error::Backend(js::message(&error))),
            };
            pending.settled(state);
        });
        result
    }

    pub(super) fn settled(&self, state: api::CompletionState) {
        let wakers = {
            let mut inner = self.0.borrow_mut();
            if matches!(inner.state, api::CompletionState::Failed(_))
                || (matches!(inner.state, api::CompletionState::Complete)
                    && !matches!(state, api::CompletionState::Failed(_)))
            {
                return;
            }
            inner.state = state;
            std::mem::take(&mut inner.wakers)
        };
        for waker in wakers {
            waker.wake();
        }
    }

    fn state(&self) -> api::CompletionState {
        match &self.0.borrow().state {
            api::CompletionState::Pending => api::CompletionState::Pending,
            api::CompletionState::Complete => api::CompletionState::Complete,
            api::CompletionState::Failed(error) => api::CompletionState::Failed(error.clone()),
        }
    }
}

impl Future for Completion {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mut inner = self.0.borrow_mut();
        match &inner.state {
            api::CompletionState::Complete => Poll::Ready(Ok(())),
            api::CompletionState::Failed(error) => Poll::Ready(Err(error.clone())),
            api::CompletionState::Pending => {
                if !inner
                    .wakers
                    .iter()
                    .any(|waker| waker.will_wake(context.waker()))
                {
                    inner.wakers.push(context.waker().clone());
                }
                Poll::Pending
            }
        }
    }
}

fn queue_completion(device: &WebGpuDevice) -> Result<Completion> {
    if device.0.lost.get() {
        return Err(Error::DeviceLost);
    }
    if let Some(message) = device.0.validation_error.borrow().clone() {
        return Err(Error::Backend(message));
    }
    let value = js::call0(&device.0.queue, "onSubmittedWorkDone")
        .map_err(|error| Error::Backend(js::message(&error)))?;
    Ok(Completion::new(Promise::from(value), device.clone()))
}

fn retire(device: &WebGpuDevice) {
    device
        .0
        .in_flight
        .borrow_mut()
        .retain(|flight| matches!(flight.completion.state(), api::CompletionState::Pending));
}

impl api::CompletionBackend for WebGpu {
    type WaitFuture<'a> = Pin<Box<dyn Future<Output = Result<()>> + 'a>>;

    fn completion_status(device: &WebGpuDevice, point: &Completion) -> api::CompletionState {
        if device.0.lost.get() {
            point.settled(api::CompletionState::Failed(Error::DeviceLost));
        } else if let Some(message) = device.0.validation_error.borrow().clone() {
            point.settled(api::CompletionState::Failed(Error::Backend(message)));
        }
        let state = point.state();
        retire(device);
        state
    }

    fn wait_completion<'a>(
        device: &'a WebGpuDevice,
        point: &'a Completion,
    ) -> Self::WaitFuture<'a> {
        Box::pin(async move {
            let result = point.clone().await;
            retire(device);
            if device.0.lost.get() {
                return Err(Error::DeviceLost);
            }
            if let Some(message) = device.0.validation_error.borrow().clone() {
                return Err(Error::Backend(message));
            }
            result
        })
    }

    fn wait_idle<'a>(device: &'a WebGpuDevice) -> Self::WaitFuture<'a> {
        let point = queue_completion(device);
        Box::pin(async move {
            let result = point?.await;
            retire(device);
            if device.0.lost.get() {
                return Err(Error::DeviceLost);
            }
            if let Some(message) = device.0.validation_error.borrow().clone() {
                return Err(Error::Backend(message));
            }
            result
        })
    }
}

impl api::DeferredBackend for WebGpu {
    type NativeRecordedWork = RecordedWork;

    fn finish_encoder(encoder: WebGpuEncoder, lane: api::LaneId) -> Result<RecordedWork> {
        if lane.device != encoder.device.0.id || lane.ordinal != 0 || encoder.pass.is_some() {
            return Err(Error::InvalidState);
        }
        let command_buffer = js::call0(&encoder.raw, "finish")
            .map_err(|error| Error::Backend(js::message(&error)))?;
        Ok(RecordedWork {
            command_buffer,
            keep_alive: encoder.keep_alive,
        })
    }

    // SAFETY: Core has validated all packet handles, lanes, and lifetime
    // ownership. This implementation consumes each recorded command buffer once
    // and retains every JS resource through the returned queue completion.
    unsafe fn submit(
        device: &WebGpuDevice,
        packet: api::SubmitPacket<Self>,
    ) -> api::SubmitOutcome<Self> {
        if device.0.lost.get() {
            return api::SubmitOutcome::Rejected {
                error: Error::DeviceLost,
                work: api::RetryableSubmit::Fresh(packet),
            };
        }
        if let Some(message) = device.0.validation_error.borrow().clone() {
            return api::SubmitOutcome::Rejected {
                error: Error::Backend(message),
                work: api::RetryableSubmit::Fresh(packet),
            };
        }
        // WebGPU exposes one ordered queue; canvas handoffs are automatic at
        // the browser's rendering update. Sparse binding has no WebGPU API.
        let shape_ok = packet
            .steps
            .iter()
            .all(|bound| !matches!(bound.payload, api::StepPayload::SparseBind(_)));
        if !shape_ok {
            return api::SubmitOutcome::Rejected {
                error: Error::Unsupported("WebGPU submission plan shape"),
                work: api::RetryableSubmit::Fresh(packet),
            };
        }
        retire(device);
        let api::SubmitPacket {
            plan,
            mut steps,
            completion,
        } = packet;
        let commands = Array::new();
        let mut resources = Vec::new();
        for bound in &mut steps {
            if let api::StepPayload::Commands(work) = &mut bound.payload {
                for work in std::mem::take(work) {
                    // SAFETY: the packet was validated for this device/lane and
                    // `work` is consumed exactly once by this submit call.
                    let native = unsafe { work.into_native() };
                    commands.push(&native.command_buffer);
                    resources.push(native.command_buffer);
                    resources.extend(native.keep_alive);
                }
            }
        }
        if let Err(error) = js::call1(&device.0.queue, "submit", &commands.into()) {
            device.0.uncertain_resources.borrow_mut().extend(resources);
            return api::SubmitOutcome::Indeterminate {
                error: Error::Backend(js::message(&error)),
                // SAFETY: queue submission was entered; the retained resource
                // set marks this device's native issue boundary as uncertain.
                uncertain: unsafe { api::NativeUncertainty::from_native(device.0.id, ()) },
            };
        }
        let point = match queue_completion(device) {
            Ok(point) => point,
            Err(error) => {
                device.0.uncertain_resources.borrow_mut().extend(resources);
                return api::SubmitOutcome::Indeterminate {
                    error,
                    // SAFETY: queue submission succeeded but completion fence
                    // creation failed, leaving this device's issue boundary uncertain.
                    uncertain: unsafe { api::NativeUncertainty::from_native(device.0.id, ()) },
                };
            }
        };
        device.0.in_flight.borrow_mut().push(InFlight {
            completion: point.clone(),
            _resources: resources,
        });
        let mut presents = Vec::new();
        let mut accepted = Vec::new();
        for mut bound in steps {
            accepted.push(bound.step);
            if let api::StepPayload::Present(binding) = &mut bound.payload {
                // SAFETY: packet validation proved this is the live exclusive
                // lease; after queue submission, failed handoff is indeterminate.
                if let Err(error) = unsafe { binding.try_handoff() } {
                    return api::SubmitOutcome::Indeterminate {
                        error,
                        // SAFETY: GPU commands were already accepted, so this
                        // device owns the indeterminate presentation boundary.
                        uncertain: unsafe { api::NativeUncertainty::from_native(device.0.id, ()) },
                    };
                }
                presents.push(api::PresentReceipt {
                    step: Some(bound.step),
                    accepted: true,
                });
            }
        }
        let last = accepted.last().copied();
        let checkpoints = match (&completion, last) {
            (api::CompletionRequest::None, _) | (_, None) => Vec::new(),
            (_, Some(frontier)) => vec![api::Checkpoint {
                // SAFETY: `point` was created from this device's queue after
                // accepted submission and therefore belongs to `device.0.id`.
                point: unsafe { api::CompletionPoint::from_native(device.0.id, point) },
                covers: api::CompletionCoverage {
                    lane_frontiers: vec![(plan.steps()[frontier.index as usize].lane, frontier)],
                },
            }],
        };
        let index = device.0.next_submission.get();
        device.0.next_submission.set(index.saturating_add(1));
        api::SubmitOutcome::Accepted(api::SubmissionReceipt {
            device: device.0.id,
            index: Some(api::SubmissionIndex {
                device: device.0.id,
                value: index,
            }),
            accepted,
            checkpoints,
            presents,
        })
    }

    // SAFETY: Core validates the original plan and caller waits for the prior
    // accepted frontier. Native work in the suffix is consumed exactly once.
    unsafe fn resume(
        device: &WebGpuDevice,
        tail: api::DeferredTail<Self>,
    ) -> api::SubmitOutcome<Self> {
        let api::DeferredTail {
            plan,
            next_step,
            pending,
            accepted_frontier,
            completion,
        } = tail;
        let suffix_completion = match &completion {
            api::CompletionRequest::None => api::CompletionRequest::None,
            api::CompletionRequest::End => api::CompletionRequest::End,
            api::CompletionRequest::Steps(steps) => api::CompletionRequest::Steps(
                steps
                    .iter()
                    .copied()
                    .filter(|step| step.index >= next_step)
                    .collect::<Vec<_>>()
                    .into(),
            ),
        };
        // SAFETY: the suffix was validated against the original plan and its
        // completed prefix satisfies every dependency on earlier steps.
        match unsafe {
            Self::submit(
                device,
                api::SubmitPacket {
                    plan,
                    steps: pending,
                    completion: suffix_completion,
                },
            )
        } {
            api::SubmitOutcome::Rejected {
                error,
                work: api::RetryableSubmit::Fresh(packet),
            } => api::SubmitOutcome::Rejected {
                error,
                work: api::RetryableSubmit::Tail(api::DeferredTail {
                    plan: packet.plan,
                    next_step,
                    pending: packet.steps,
                    accepted_frontier,
                    completion,
                }),
            },
            api::SubmitOutcome::Accepted(mut receipt) => {
                if matches!(completion, api::CompletionRequest::Steps(_))
                    || (matches!(completion, api::CompletionRequest::End)
                        && receipt.checkpoints.is_empty())
                {
                    receipt.checkpoints.splice(0..0, accepted_frontier);
                }
                api::SubmitOutcome::Accepted(receipt)
            }
            other => other,
        }
    }
}
