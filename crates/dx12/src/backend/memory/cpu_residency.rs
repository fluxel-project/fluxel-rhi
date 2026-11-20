//! CPU map and native residency exclusion for one explicit physical heap.
//!
//! Placed buffers share this state with their heap. A map lease prevents any
//! residency transition until Unmap finishes. A pending native page-in keeps
//! new maps out until its own fence completes. The renderer still orders GPU
//! uses and chooses when to change residency references.

use super::super::encoder::{Dx12Completion, fence_wait::fence_progress};
use fluxel_rhi_core::api::{Error, Result};
use std::ops::Range;
use std::sync::{Arc, Mutex};

pub(in crate::backend) struct CpuHeapAccess {
    state: Mutex<State>,
}

struct State {
    maps: Vec<Range<u64>>,
    references: u64,
    pending: Option<Dx12Completion>,
}

impl State {
    fn check_pending(&mut self) -> Result<()> {
        if let Some(point) = &self.pending {
            // SAFETY: the accepted operation owns the live fence for this
            // allocation. Observing it does not block or mutate GPU work.
            let completed = unsafe { point.fence.GetCompletedValue() };
            if !fence_progress(completed, point.value)? {
                return Err(Error::InvalidState);
            }
            self.pending = None;
        }
        Ok(())
    }

    fn check_transition(&mut self) -> Result<()> {
        self.check_pending()?;
        if !self.maps.is_empty() {
            return Err(Error::InvalidState);
        }
        Ok(())
    }
}

impl CpuHeapAccess {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(State {
                maps: Vec::new(),
                references: 1,
                pending: None,
            }),
        }
    }

    pub(in crate::backend) fn acquire_map(
        self: &Arc<Self>,
        range: Range<u64>,
    ) -> Result<CpuHeapMapLease> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.check_pending()?;
        if state.references == 0
            || state
                .maps
                .iter()
                .any(|mapped| range.start < mapped.end && mapped.start < range.end)
        {
            return Err(Error::InvalidState);
        }
        state.maps.push(range.clone());
        Ok(CpuHeapMapLease {
            access: self.clone(),
            range,
        })
    }

    pub(super) fn make_resident(&self, native: impl FnOnce() -> Result<()>) -> Result<()> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.check_transition()?;
        let references = state.references.checked_add(1).ok_or(Error::InvalidRange)?;
        native()?;
        state.references = references;
        Ok(())
    }

    pub(super) fn enqueue(
        &self,
        native: impl FnOnce() -> Result<Dx12Completion>,
    ) -> Result<Dx12Completion> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.check_transition()?;
        let references = state.references.checked_add(1).ok_or(Error::InvalidRange)?;
        let point = native()?;
        state.pending = Some(point.clone());
        state.references = references;
        Ok(point)
    }

    pub(super) fn evict(&self, native: impl FnOnce() -> Result<()>) -> Result<()> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.check_transition()?;
        let references = state.references.checked_sub(1).ok_or(Error::InvalidState)?;
        native()?;
        state.references = references;
        Ok(())
    }
}

pub(in crate::backend) struct CpuHeapMapLease {
    access: Arc<CpuHeapAccess>,
    range: Range<u64>,
}

impl Drop for CpuHeapMapLease {
    fn drop(&mut self) {
        let mut state = self
            .access
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let index = state
            .maps
            .iter()
            .position(|range| *range == self.range)
            .expect("live map lease has a registered range");
        state.maps.swap_remove(index);
    }
}
