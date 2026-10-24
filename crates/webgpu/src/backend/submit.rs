//! One-shot native command buffers and browser queue completion.
//!
//! Finished command buffers are submitted once. Their JS objects stay alive in
//! `in_flight` until `onSubmittedWorkDone` settles; loss and validation failure
//! settle the same completion path, so a caller never observes a fake success.

use std::{
    cell::RefCell,
    collections::BTreeMap,
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

pub struct Completion(Rc<RefCell<CompletionInner>>, Option<u64>);
impl Clone for Completion {
    fn clone(&self) -> Self {
        Self(self.0.clone(), None)
    }
}
impl Drop for Completion {
    fn drop(&mut self) {
        if let Some(id) = self.1.take() {
            self.0.borrow_mut().wakers.remove(&id);
        }
    }
}

struct CompletionInner {
    state: api::CompletionState,
    // Validation errors can fail a point before the native queue is done.
    // Only the queue promise supplies retirement evidence.
    native_settled: bool,
    wakers: BTreeMap<u64, Waker>,
    next_registration: u64,
}

pub(super) struct InFlight {
    pub(super) completion: Completion,
    _resources: Vec<JsValue>,
}

impl Completion {
    fn new(promise: Promise, device: WebGpuDevice) -> Self {
        let result = Self(
            Rc::new(RefCell::new(CompletionInner {
                state: api::CompletionState::Pending,
                native_settled: false,
                wakers: BTreeMap::new(),
                next_registration: 0,
            })),
            None,
        );
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
            pending.0.borrow_mut().native_settled = true;
            pending.settled(state);
            // Promise settlement is the retirement trigger, including None
            // submissions whose caller never asks for another RHI operation.
            retire(&device);
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
        for waker in wakers.into_values() {
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
        let this = self.get_mut();
        let mut inner = this.0.borrow_mut();
        match &inner.state {
            api::CompletionState::Complete => Poll::Ready(Ok(())),
            api::CompletionState::Failed(error) => Poll::Ready(Err(error.clone())),
            api::CompletionState::Pending => {
                let id = match this.1 {
                    Some(id) => id,
                    None => {
                        let Some(next) = inner.next_registration.checked_add(1) else {
                            return Poll::Ready(Err(Error::InvalidRange));
                        };
                        let id = inner.next_registration;
                        inner.next_registration = next;
                        this.1 = Some(id);
                        id
                    }
                };
                match inner.wakers.get_mut(&id) {
                    Some(waker) if waker.will_wake(context.waker()) => {}
                    Some(waker) => *waker = context.waker().clone(),
                    None => {
                        inner.wakers.insert(id, context.waker().clone());
                    }
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
        .retain(|flight| !flight.completion.0.borrow().native_settled);
}

/// Issues one browser queue boundary and retains its native objects until the
/// queue's promise settles. Once `queue.submit` has been called, every error
/// is necessarily indeterminate: JavaScript cannot report whether the queue
/// accepted the boundary before throwing or before completion observation was
/// installed.
fn submit_batch(
    device: &WebGpuDevice,
    commands: Array,
    resources: Vec<JsValue>,
) -> Result<Completion> {
    if let Err(error) = js::call1(&device.0.queue, "submit", &commands.into()) {
        device.0.uncertain_resources.borrow_mut().extend(resources);
        return Err(Error::Backend(js::message(&error)));
    }
    let point = match queue_completion(device) {
        Ok(point) => point,
        Err(error) => {
            device.0.uncertain_resources.borrow_mut().extend(resources);
            return Err(error);
        }
    };
    device.0.in_flight.borrow_mut().push(InFlight {
        completion: point.clone(),
        _resources: resources,
    });
    Ok(point)
}

fn indeterminate(device: &WebGpuDevice, error: Error) -> api::SubmitOutcome<WebGpu> {
    api::SubmitOutcome::Indeterminate {
        error,
        // SAFETY: callers invoke this only after crossing a browser queue or
        // present-handoff boundary, so Core must retain recovery ownership.
        uncertain: unsafe { api::NativeUncertainty::from_native(device.0.id, ()) },
    }
}

fn flush_batch(
    device: &WebGpuDevice,
    frontier: api::StepId,
    commands: &mut Array,
    resources: &mut Vec<JsValue>,
    pending_presents: &mut Vec<(api::StepId, api::PresentBinding<WebGpu>)>,
    presents: &mut Vec<api::PresentReceipt>,
    issued_boundaries: &mut Vec<(api::StepId, Completion)>,
) -> Result<()> {
    let point = submit_batch(
        device,
        std::mem::replace(commands, Array::new()),
        std::mem::take(resources),
    )?;
    for (step, mut binding) in pending_presents.drain(..) {
        // SAFETY: queue submission for this batch succeeded first; a failed
        // browser handoff therefore leaves an indeterminate native boundary
        // and the in-flight batch remains retained.
        unsafe { binding.try_handoff() }?;
        presents.push(api::PresentReceipt {
            step: Some(step),
            accepted: true,
        });
    }
    issued_boundaries.push((frontier, point));
    Ok(())
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
            steps,
            completion,
        } = packet;
        let requested = match &completion {
            api::CompletionRequest::Steps(steps) => steps.to_vec(),
            _ => Vec::new(),
        };
        let segmented = matches!(completion, api::CompletionRequest::Steps(_));
        let mut commands = Array::new();
        let mut resources = Vec::new();
        let mut presents = Vec::new();
        let mut accepted = Vec::new();
        let mut pending_presents: Vec<(api::StepId, api::PresentBinding<WebGpu>)> = Vec::new();
        let mut checkpoints = Vec::new();
        let mut issued_boundaries = Vec::new();
        for bound in steps {
            let step = bound.step;
            match bound.payload {
                api::StepPayload::Commands(work) => {
                    for work in work {
                        // SAFETY: the packet was validated for this device/lane
                        // and `work` is consumed exactly once by this submit call.
                        let native = unsafe { work.into_native() };
                        commands.push(&native.command_buffer);
                        resources.push(native.command_buffer);
                        resources.extend(native.keep_alive);
                    }
                }
                api::StepPayload::Present(binding) => pending_presents.push((step, binding)),
                api::StepPayload::SparseBind(_) => unreachable!("prevalidated WebGPU plan"),
            }
            accepted.push(step);
            if segmented && requested.contains(&step) {
                if let Err(error) = flush_batch(
                    device,
                    step,
                    &mut commands,
                    &mut resources,
                    &mut pending_presents,
                    &mut presents,
                    &mut issued_boundaries,
                ) {
                    return indeterminate(device, error);
                }
            }
        }
        // `None`/`End` deliberately remain one browser submission.  `Steps`
        // emits a tail boundary only when work or a present follows its final
        // requested checkpoint, retaining that tail without exposing a token.
        if !segmented || !resources.is_empty() || !pending_presents.is_empty() {
            let frontier = accepted.last().copied().unwrap_or(api::StepId {
                plan: plan.id(),
                index: 0,
            });
            if let Err(error) = flush_batch(
                device,
                frontier,
                &mut commands,
                &mut resources,
                &mut pending_presents,
                &mut presents,
                &mut issued_boundaries,
            ) {
                return indeterminate(device, error);
            }
        }
        // A request list may contain duplicates or be deliberately ordered
        // differently from execution. Each requested item refers to the one
        // native boundary emitted immediately after that exact plan step.
        for requested_step in &requested {
            let (_, point) = issued_boundaries
                .iter()
                .find(|(frontier, _)| frontier == requested_step)
                .expect("validated requested WebGPU completion boundary");
            checkpoints.push(api::Checkpoint {
                // SAFETY: this promise was installed after the matching queue
                // submit boundary on this device.
                point: unsafe { api::CompletionPoint::from_native(device.0.id, point.clone()) },
                covers: api::CompletionCoverage {
                    lane_frontiers: vec![(
                        plan.steps()[requested_step.index as usize].lane,
                        *requested_step,
                    )],
                },
            });
        }
        if matches!(completion, api::CompletionRequest::End) {
            if let Some(frontier) = accepted.last().copied() {
                let (_, point) = issued_boundaries
                    .last()
                    .expect("one WebGPU batch was issued for accepted work");
                checkpoints.push(api::Checkpoint {
                    // SAFETY: the final queue promise follows the full packet.
                    point: unsafe { api::CompletionPoint::from_native(device.0.id, point.clone()) },
                    covers: api::CompletionCoverage {
                        lane_frontiers: vec![(
                            plan.steps()[frontier.index as usize].lane,
                            frontier,
                        )],
                    },
                });
            }
        }
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

#[cfg(test)]
#[path = "submit/tests.rs"]
mod tests;
