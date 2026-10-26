//! Native fence and explicit memory-visibility execution.
//!
//! Fence observations distinguish signaled, pending, and unrecognized native
//! statuses. Driver-error observation is deferred to the presentation boundary
//! so polling a fence cannot itself become a per-frame synchronization point.
//! Query-object execution has its own child module because its active-query
//! invariant is independent of fence lifetime.

use super::provider::NativeGlProvider;
use crate::backend::api::{
    GlContextLifecycle, GlError, GlFamilyApi as _, GlFenceLease, GlFenceStatus, GlMemoryBarrier,
    GlSyncApi, GlWaitBound, SyncId,
};

/// `client_waitSync` accepts a 64-bit timeout, but glow 0.18 binds it as
/// `i32` nanoseconds. A larger bounded wait is clamped to that ceiling
/// (about two seconds); a longer effective wait stays expressible through
/// repeated polls, and zero remains a pure poll.
const MAX_WAIT_TIMEOUT_NS: i32 = i32::MAX;

impl GlSyncApi for NativeGlProvider {
    fn create_fence(&mut self) -> Result<GlFenceLease, GlError> {
        use glow::HasContext as _;
        const OP: &str = "create-fence";
        self.assert_ready(OP)?;
        // SAFETY: current-context contract; the fence is deleted if the Fluxel
        // lease book rejects the issue.
        let raw = unsafe { self.gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0) }.map_err(
            |message| GlError::Driver {
                operation: OP,
                message,
            },
        )?;
        let slot = match self.slot(OP) {
            Ok(slot) => slot,
            Err(error) => {
                // SAFETY: the fence was created by this current context but
                // cannot be published without a Fluxel identity. Roll back
                // this private, otherwise-unreachable GL object before
                // returning the allocation failure.
                unsafe { self.gl.delete_sync(raw) };
                return Err(error);
            }
        };
        let id = SyncId::new(self.context_stamp(), slot, 0);
        self.syncs.insert(id, raw);
        self.commands_pending_submission.set(true);
        match self.fences.issue(id) {
            Ok(lease) => Ok(lease),
            Err(error) => {
                if let Some(entry) = self.syncs.remove(&id) {
                    // SAFETY: current-context contract; rollback of this
                    // transaction's own object.
                    unsafe { self.gl.delete_sync(entry) };
                }
                Err(error)
            }
        }
    }

    fn destroy_fence(&mut self, fence: GlFenceLease) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "destroy-fence";
        self.assert_ready(OP)?;
        self.validate_object_context(OP, fence.fence.context)?;
        self.fences.validate(fence)?;
        let raw = self.sync(OP, fence.fence)?;
        // SAFETY: current-context contract; liveness was checked first.
        unsafe { self.gl.delete_sync(raw) };
        self.syncs.remove(&fence.fence);
        self.fences.revoke(fence);
        self.defer_readonly_error(OP)
    }

    fn poll_fence(&mut self, fence: GlFenceLease) -> Result<GlFenceStatus, GlError> {
        const OP: &str = "poll-fence";
        self.assert_ready(OP)?;
        self.validate_object_context(OP, fence.fence.context)?;
        self.fences.validate(fence)?;
        let raw = self.sync(OP, fence.fence)?;
        let submitted = self.submit_pending_commands(OP);
        let status = self.sync_status(OP, raw)?;
        // A headless fence needs one publication boundary. Only that first
        // submission observes the deferred stream error; later timer-driven
        // polls remain pure non-blocking status checks.
        if submitted {
            self.take_deferred_driver_error(OP)?;
        }
        Ok(status)
    }

    fn wait_fence(
        &mut self,
        fence: GlFenceLease,
        bound: GlWaitBound,
    ) -> Result<GlFenceStatus, GlError> {
        use glow::HasContext as _;
        const OP: &str = "wait-fence";
        self.assert_ready(OP)?;
        self.validate_object_context(OP, fence.fence.context)?;
        self.fences.validate(fence)?;
        let raw = self.sync(OP, fence.fence)?;
        if bound.nanoseconds == 0 {
            let submitted = self.submit_pending_commands(OP);
            if submitted {
                self.take_deferred_driver_error(OP)?;
            }
        }
        // A zero bound is a poll: no flush flag, no wait.
        let flags = if bound.nanoseconds == 0 {
            0
        } else {
            glow::SYNC_FLUSH_COMMANDS_BIT
        };
        let timeout = if bound.nanoseconds == 0 {
            0
        } else {
            bound
                .nanoseconds
                .min(u64::try_from(MAX_WAIT_TIMEOUT_NS).unwrap_or(i32::MAX as u64))
                as i32
        };
        // SAFETY: current-context contract; the fence is live.
        let result = unsafe { self.gl.client_wait_sync(raw, flags, timeout) };
        if bound.nanoseconds != 0 {
            // The flush flag makes all preceding work, including this fence,
            // visible to the server as part of the explicit wait.
            self.commands_pending_submission.set(false);
        }
        let status = match result {
            glow::ALREADY_SIGNALED | glow::CONDITION_SATISFIED => GlFenceStatus::Complete,
            glow::TIMEOUT_EXPIRED => GlFenceStatus::Pending,
            // WAIT_FAILED and any other answer stay Unknown so accepted work
            // keeps its lease until the caller observes a usable status.
            _ => GlFenceStatus::Unknown,
        };
        if status == GlFenceStatus::Unknown {
            return self.abnormal_wait_failure(OP);
        }
        if bound.nanoseconds != 0 {
            // An explicit bounded wait is a caller-selected slow boundary.
            self.take_deferred_driver_error(OP)?;
        }
        Ok(status)
    }

    fn flush(&mut self) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "flush";
        self.assert_ready(OP)?;
        // SAFETY: current-context contract.
        unsafe { self.gl.flush() };
        self.defer_readonly_error(OP)?;
        self.commands_pending_submission.set(false);
        self.take_deferred_driver_error(OP)
    }

    fn memory_barrier(&mut self, barriers: GlMemoryBarrier) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "memory-barrier";
        self.assert_ready(OP)?;
        barriers.validate_nonempty()?;
        // SAFETY: the current context owns every preceding direct command;
        // validated nonempty classes are translated only to GL barrier bits.
        unsafe {
            self.gl
                .memory_barrier(super::exec_compute::native_barrier_bits(barriers.0));
        }
        self.defer_driver_error(OP)
    }

    fn memory_barrier_by_region(&mut self, barriers: GlMemoryBarrier) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "memory-barrier-by-region";
        self.assert_ready(OP)?;
        barriers.validate_by_region()?;
        // SAFETY: this current context supports the admitted regional barrier
        // route and validation restricted the mask to its legal classes.
        unsafe {
            self.gl
                .memory_barrier_by_region(super::exec_compute::native_barrier_bits(barriers.0));
        }
        self.defer_driver_error(OP)
    }

    fn texture_barrier(&mut self) -> Result<(), GlError> {
        const OP: &str = "texture-barrier";
        self.assert_ready(OP)?;
        Err(GlError::Unsupported {
            operation: OP,
            reason: "texture-barrier requires separately admitted native entry-point lowering",
        })
    }
}

impl NativeGlProvider {
    /// Observes one fence with a zero-timeout client wait.
    ///
    /// Zero flags avoid requesting a flush; zero timeout avoids waiting for
    /// GPU completion. This is still a driver status observation, so callers
    /// must keep polling bounded rather than treating it as a free CPU read.
    fn sync_status(
        &mut self,
        operation: &'static str,
        raw: glow::NativeFence,
    ) -> Result<GlFenceStatus, GlError> {
        use glow::HasContext as _;
        // SAFETY: current-context contract; the fence is live and the zero
        // flags/timeout make this a pure, non-blocking poll.
        let value = unsafe { self.gl.client_wait_sync(raw, 0, 0) };
        let status = match value {
            glow::ALREADY_SIGNALED | glow::CONDITION_SATISFIED => GlFenceStatus::Complete,
            glow::TIMEOUT_EXPIRED => GlFenceStatus::Pending,
            _ => GlFenceStatus::Unknown,
        };
        if status == GlFenceStatus::Unknown {
            return self.abnormal_wait_failure(operation);
        }
        Ok(status)
    }

    /// Converts an abnormal client-wait result into terminal provider failure.
    /// This slow diagnostic path is deliberately separate from ordinary
    /// zero-time pending polls: even a clean error flag cannot make a
    /// WAIT_FAILED context safe for optimistic caches or existing fences.
    fn abnormal_wait_failure(&mut self, operation: &'static str) -> Result<GlFenceStatus, GlError> {
        self.take_deferred_driver_error(operation)?;
        self.lifecycle = GlContextLifecycle::Lost;
        self.reset_executable_state();
        Err(GlError::Driver {
            operation,
            message: "glClientWaitSync returned WAIT_FAILED or an unknown status".into(),
        })
    }

    fn sync(&self, operation: &'static str, id: SyncId) -> Result<glow::NativeFence, GlError> {
        self.validate_object_context(operation, id.context)?;
        // Map keys are full identities, so a hit implies the same generation.
        self.syncs
            .get(&id)
            .copied()
            .ok_or_else(|| Self::validation(operation, "fence is not live"))
    }
}

#[path = "exec_query.rs"]
mod query;
