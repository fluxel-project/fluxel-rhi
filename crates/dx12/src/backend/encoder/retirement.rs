//! Autonomous native-fence retirement for submitted DX12 owners.
//!
//! A completed submission must release retained command work even when its
//! caller drops every public `Device` and completion receipt.  The one-shot
//! Windows thread-pool wait owns the only extra `Dx12Device` reference; it is
//! never stored back in the device, avoiding a State -> work -> State cycle.

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
#[path = "../../tests/backend/encoder/retirement_support.rs"]
mod test_support;
#[cfg(all(test, feature = "hardware-tests"))]
use test_support::inject_failure;
#[cfg(test)]
use test_support::{FailureStage, fail_here};

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
#[path = "../../tests/backend/encoder/retirement.rs"]
mod hardware_tests;
