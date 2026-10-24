//! Autonomous native-fence retirement for submitted DX12 owners.
//!
//! A completed submission must release retained command work even when its
//! caller drops every public `Device` and completion receipt.  The one-shot
//! Windows thread-pool wait owns the only extra `Dx12Device` reference; it is
//! never stored back in the device, avoiding a State -> work -> State cycle.

#[cfg(test)]
use std::cell::Cell;
use std::{
    ffi::c_void,
    sync::{
        Arc,
        atomic::{AtomicIsize, Ordering},
    },
};

use fluxel_rhi_core::api::{Error, Result};

use windows::Win32::{
    Foundation::{CloseHandle, HANDLE},
    Graphics::Direct3D12::ID3D12Fence,
    System::Threading::{
        CloseThreadpoolWait, CreateEventW, CreateThreadpoolWait,
        DisassociateCurrentThreadFromCallback, PTP_CALLBACK_INSTANCE, PTP_WAIT, SetThreadpoolWait,
    },
};

use super::{Dx12Device, native_error, submit::retire};

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FailureStage {
    Event,
    Wait,
    SetEvent,
}

#[cfg(test)]
thread_local! {
    static FAILURE_STAGE: Cell<Option<FailureStage>> = const { Cell::new(None) };
}

#[cfg(test)]
fn fail_here(stage: FailureStage) -> bool {
    FAILURE_STAGE.with(|slot| {
        if slot.get() == Some(stage) {
            slot.set(None);
            true
        } else {
            false
        }
    })
}

#[cfg(test)]
fn inject_failure(stage: FailureStage) {
    FAILURE_STAGE.with(|slot| slot.set(Some(stage)));
}

struct RetirementContext {
    device: Dx12Device,
    event: Option<HANDLE>,
    wait: AtomicIsize,
}

impl Drop for RetirementContext {
    fn drop(&mut self) {
        if let Some(event) = self.event.take() {
            // SAFETY: this context exclusively owns the event unless setup
            // transferred it to the abandoned-event store.
            unsafe {
                let _ = CloseHandle(event);
            }
        }
    }
}

unsafe extern "system" fn retire_on_fence(
    callback: PTP_CALLBACK_INSTANCE,
    context: *mut c_void,
    _: PTP_WAIT,
    _: u32,
) {
    // The raw Arc is transferred exactly once to this one-shot callback. Its
    // strong device reference retains all native state until retirement.
    let context = unsafe { Arc::from_raw(context.cast::<RetirementContext>()) };
    let wait = PTP_WAIT(context.wait.load(Ordering::Acquire));
    debug_assert_ne!(wait.0, 0);
    // Never drain this wait from its own callback: WaitForThreadpoolWaitCallbacks
    // would deadlock. Disassociation permits teardown without waiting on self.
    unsafe {
        DisassociateCurrentThreadFromCallback(callback);
        SetThreadpoolWait(wait, None, None);
        CloseThreadpoolWait(wait);
    }
    retire(&context.device);
    // Dropping context releases its final registration-owned Device Arc.
}

fn completed(fence: &ID3D12Fence, value: u64) -> Option<bool> {
    // SAFETY: the registration owns a cloned device fence; querying is pure.
    match unsafe { fence.GetCompletedValue() } {
        u64::MAX => None,
        completed => Some(completed >= value),
    }
}

/// Unwinds an unarmed wait after D3D12 rejected its event registration.
///
/// The native submission has already happened, so the fence/event is retained
/// with a recovery-only device owner rather than abandoning its in-flight
/// work. No callback can race this path because `SetThreadpoolWait` has not
/// run.
fn failed_setup(
    device: &Dx12Device,
    fence: ID3D12Fence,
    value: u64,
    raw: *const RetirementContext,
    wait: PTP_WAIT,
    error: Error,
) -> Result<()> {
    // SAFETY: this is the sole raw-Arc owner while the wait is unarmed.
    unsafe {
        CloseThreadpoolWait(wait);
        let mut context = Arc::from_raw(raw);
        let event = Arc::get_mut(&mut context)
            .expect("unarmed retirement context has one owner")
            .event
            .take()
            .expect("retirement event ownership");
        super::fence_wait::retain_failed_retirement_event(device, fence, value, Some(event));
        drop(context);
    }
    Err(error)
}

fn failed_before_wait(
    device: &Dx12Device,
    fence: ID3D12Fence,
    value: u64,
    event: Option<HANDLE>,
    error: Error,
) -> Result<()> {
    super::fence_wait::retain_failed_retirement_event(device, fence, value, event);
    Err(error)
}

/// Registers one terminal lane fence without CPU polling or a user worker.
///
/// A registration failure leaves the existing in-flight owner intact rather
/// than claiming completion. Device removal (`u64::MAX`) remains quarantined.
pub(super) fn register(device: &Dx12Device, ordinal: u32, value: u64) -> Result<()> {
    if value == u64::MAX {
        return Ok(());
    }
    let fence = match super::super::queue::fence(device, ordinal) {
        Ok(fence) => fence.clone(),
        Err(error) => return Err(error),
    };
    match completed(&fence, value) {
        Some(true) => {
            retire(device);
            return Ok(());
        }
        None => {
            // Native handoff already happened. A sparse-only packet need not
            // own an encoder/device Arc, so loss also requires the error-only
            // owner quarantine before reporting uncertainty.
            super::fence_wait::retain_failed_retirement_event(device, fence, value, None);
            return Err(Error::DeviceLost);
        }
        Some(false) => {}
    }
    // SAFETY: the event is newly created, nonsignaled, and owned until the
    // callback closes it or registration setup unwinds it.
    #[cfg(test)]
    if fail_here(FailureStage::Event) {
        return failed_before_wait(
            device,
            fence,
            value,
            None,
            Error::Backend("injected CreateEventW failure".into()),
        );
    }
    let event = match unsafe { CreateEventW(None, false, false, None) } {
        Ok(event) => event,
        Err(error) => return failed_before_wait(device, fence, value, None, native_error(error)),
    };
    let context = Arc::new(RetirementContext {
        device: device.clone(),
        event: Some(event),
        wait: AtomicIsize::new(0),
    });
    let raw = Arc::into_raw(context);
    // SAFETY: `raw` remains owned by this setup path until it is transferred to
    // the callback; the callback ABI matches retire_on_fence.
    #[cfg(test)]
    if fail_here(FailureStage::Wait) {
        // SAFETY: this setup path still exclusively owns the raw Arc.
        let mut context = unsafe { Arc::from_raw(raw) };
        let event = Arc::get_mut(&mut context)
            .expect("unarmed retirement context has one owner")
            .event
            .take();
        drop(context);
        return failed_before_wait(
            device,
            fence,
            value,
            event,
            Error::Backend("injected CreateThreadpoolWait failure".into()),
        );
    }
    let wait = match unsafe {
        CreateThreadpoolWait(Some(retire_on_fence), Some(raw.cast_mut().cast()), None)
    } {
        Ok(wait) => wait,
        Err(error) => {
            // SAFETY: no wait was created, so this setup path exclusively
            // owns the raw Arc and can transfer its event to quarantine.
            let mut context = unsafe { Arc::from_raw(raw) };
            let event = Arc::get_mut(&mut context)
                .expect("unarmed retirement context has one owner")
                .event
                .take();
            drop(context);
            return failed_before_wait(device, fence, value, event, native_error(error));
        }
    };
    // The wait is intentionally not armed yet. D3D12 owns the event
    // registration first, so a SetEventOnCompletion failure has no callback
    // race and can transfer the event to device-owned retirement safely.
    unsafe { (*raw).wait.store(wait.0, Ordering::Release) };
    #[cfg(test)]
    if fail_here(FailureStage::SetEvent) {
        return failed_setup(
            device,
            fence,
            value,
            raw,
            wait,
            Error::Backend("injected SetEventOnCompletion failure".into()),
        );
    }
    if let Err(error) = unsafe { fence.SetEventOnCompletion(value, event) } {
        return failed_setup(device, fence, value, raw, wait, native_error(error));
    }
    // SAFETY: the native fence now owns event signaling. If it already
    // completed, arming observes the signaled event and the callback owns
    // context teardown; setup touches no context after this call.
    unsafe { SetThreadpoolWait(wait, Some(event), None) };
    Ok(())
}

#[cfg(all(test, feature = "hardware-tests"))]
mod hardware_tests {
    use crate::backend::Dx12Instance;
    use fluxel_rhi_core::api::*;
    use std::{
        future::Future,
        sync::Arc,
        thread,
        time::{Duration, Instant},
    };
    use windows::Win32::Graphics::Direct3D12::{D3D12_FENCE_FLAG_NONE, ID3D12Fence};

    /// Releases the private native gate even while an assertion unwinds, so a
    /// failed retention assertion cannot strand either queue in this process.
    struct GateRelease {
        fence: ID3D12Fence,
        released: bool,
    }

    impl GateRelease {
        fn new(fence: ID3D12Fence) -> Self {
            Self {
                fence,
                released: false,
            }
        }

        fn release(&mut self) {
            if !std::mem::replace(&mut self.released, true) {
                // SAFETY: this fixture owns the sole CPU producer signal.
                unsafe { self.fence.Signal(1) }.expect("release retirement test gate");
            }
        }
    }

    impl Drop for GateRelease {
        fn drop(&mut self) {
            if !std::mem::replace(&mut self.released, true) {
                // SAFETY: best-effort unwinding cleanup for the same signal.
                let _ = unsafe { self.fence.Signal(1) };
            }
        }
    }

    fn hold_native_queues(device: &Device<crate::backend::Dx12>, compute: bool) -> GateRelease {
        // SAFETY: the test owns no concurrent raw queue access; the device
        // keeps these native interfaces valid through the two Wait calls.
        let native = unsafe { device.native_ref() };
        let fence = unsafe {
            native
                .0
                .native
                .CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE)
        }
        .expect("create private retirement test fence");
        unsafe { native.0.queue.Wait(&fence, 1) }.expect("hold direct queue");
        if compute {
            unsafe { native.0.multi_queue.queue.Wait(&fence, 1) }.expect("hold compute queue");
        }
        GateRelease::new(fence)
    }

    fn block<F: Future>(future: F) -> F::Output {
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                std::task::Poll::Ready(value) => return value,
                std::task::Poll::Pending => thread::sleep(Duration::from_millis(1)),
            }
        }
    }

    #[test]
    fn none_submission_retires_direct_and_compute_after_external_device_drop() {
        let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
        let mut gate = hold_native_queues(&device, true);
        let direct = device
            .lanes()
            .iter()
            .find(|lane| lane.class == LaneClass::Graphics)
            .unwrap()
            .id;
        let compute = device
            .lanes()
            .iter()
            .find(|lane| lane.class == LaneClass::Compute)
            .unwrap()
            .id;
        let direct_buffer = device
            .create_buffer(&BufferDesc {
                label: None,
                size: 4,
                usage: BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let compute_buffer = device
            .create_buffer(&BufferDesc {
                label: None,
                size: 4,
                usage: BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let direct_weak = Arc::downgrade(&unsafe { direct_buffer.native_ref() }.mapped);
        let compute_weak = Arc::downgrade(&unsafe { compute_buffer.native_ref() }.mapped);
        let state_weak = Arc::downgrade(&unsafe { device.native_ref() }.0);
        let mut direct_encoder = device.encoder(direct).unwrap();
        direct_encoder
            .write_buffer(&direct_buffer, 0, &[1, 2, 3, 4])
            .unwrap();
        let mut compute_encoder = device.encoder(compute).unwrap();
        compute_encoder
            .write_buffer(&compute_buffer, 0, &[5, 6, 7, 8])
            .unwrap();
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: device.id(),
                    ordinal: 99_941,
                },
                Arc::from([
                    StepDesc {
                        lane: direct,
                        kind: StepKind::Commands,
                        waits: vec![],
                    },
                    StepDesc {
                        lane: compute,
                        kind: StepKind::Commands,
                        waits: vec![],
                    },
                ]),
            )
        };
        let first = plan.step(0);
        let second = plan.step(1);
        // SAFETY: independent one-shot lists use their declared lanes only.
        let receipt = unsafe {
            device.queue().submit(SubmitPacket {
                plan,
                steps: vec![
                    BoundStep {
                        step: first,
                        payload: StepPayload::Commands(vec![direct_encoder.finish().unwrap()]),
                    },
                    BoundStep {
                        step: second,
                        payload: StepPayload::Commands(vec![compute_encoder.finish().unwrap()]),
                    },
                ],
                completion: CompletionRequest::None,
            })
        };
        assert!(matches!(receipt, SubmitOutcome::Accepted(_)));
        drop(receipt);
        drop(direct_buffer);
        drop(compute_buffer);
        drop(device);

        // The queues cannot pass their terminal signals yet, so this proves
        // retention comes from the autonomous callback/in-flight stores, not
        // from a fast synchronous `GetCompletedValue` path.
        assert!(state_weak.upgrade().is_some());
        assert!(direct_weak.upgrade().is_some());
        assert!(compute_weak.upgrade().is_some());
        gate.release();
        drop(gate);

        let deadline = Instant::now() + Duration::from_secs(5);
        while state_weak.upgrade().is_some()
            || direct_weak.upgrade().is_some()
            || compute_weak.upgrade().is_some()
        {
            assert!(
                Instant::now() < deadline,
                "native fence retirement did not break retention"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn every_failed_setup_stage_quarantines_until_explicit_idle_recovery() {
        for (offset, stage) in [
            super::FailureStage::Event,
            super::FailureStage::Wait,
            super::FailureStage::SetEvent,
        ]
        .into_iter()
        .enumerate()
        {
            failed_stage_quarantines_until_idle(stage, offset as u64);
        }
    }

    fn failed_stage_quarantines_until_idle(stage: super::FailureStage, offset: u64) {
        let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
        let mut gate = hold_native_queues(&device, false);
        let direct = device
            .lanes()
            .iter()
            .find(|lane| lane.class == LaneClass::Graphics)
            .unwrap()
            .id;
        let buffer = device
            .create_buffer(&BufferDesc {
                label: None,
                size: 4,
                usage: BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let buffer_weak = Arc::downgrade(&unsafe { buffer.native_ref() }.mapped);
        let state_weak = Arc::downgrade(&unsafe { device.native_ref() }.0);
        let mut encoder = device.encoder(direct).unwrap();
        encoder.write_buffer(&buffer, 0, &[9, 8, 7, 6]).unwrap();
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: device.id(),
                    ordinal: 99_942 + offset,
                },
                Arc::from([StepDesc {
                    lane: direct,
                    kind: StepKind::Commands,
                    waits: vec![],
                }]),
            )
        };
        super::inject_failure(stage);
        // SAFETY: the completed command list belongs to this plan's direct lane.
        let receipt = unsafe {
            device.queue().submit(SubmitPacket {
                plan: plan.clone(),
                steps: vec![BoundStep {
                    step: plan.step(0),
                    payload: StepPayload::Commands(vec![encoder.finish().unwrap()]),
                }],
                completion: CompletionRequest::None,
            })
        };
        assert!(matches!(receipt, SubmitOutcome::Indeterminate { .. }));
        drop(receipt);
        assert_eq!(
            unsafe { device.native_ref() }
                .0
                .abandoned_fence_events
                .lock()
                .unwrap()
                .len(),
            1,
            "failed unarmed setup must retain the accepted submission"
        );

        // The queue is still behind the private fence. This prevents a fast
        // GPU from bypassing all injected setup stages through the synchronous
        // completion fast path.
        assert!(state_weak.upgrade().is_some());
        assert!(buffer_weak.upgrade().is_some());
        gate.release();
        drop(gate);

        // This is the explicit recovery operation available after an
        // indeterminate handoff. It observes completion and releases both the
        // event and the error-only self-retention Arc.
        block(device.wait_idle()).unwrap();
        assert!(
            unsafe { device.native_ref() }
                .0
                .abandoned_fence_events
                .lock()
                .unwrap()
                .is_empty()
        );
        drop(buffer);
        drop(device);

        let deadline = Instant::now() + Duration::from_secs(5);
        while state_weak.upgrade().is_some() || buffer_weak.upgrade().is_some() {
            assert!(
                Instant::now() < deadline,
                "explicit recovery left a failed-setup device retention cycle"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }
}
