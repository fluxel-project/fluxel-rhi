//! Synchronous WebGL transfer primitives for CPU-owned mapping leases.
//!
//! Host memory is a snapshot; only explicitly flushed ranges are uploaded.
use super::owner::{BrowserBuffer, BrowserOwner};
use fluxel_rhi_core::api::{Error, Result};
use std::rc::Rc;
use web_sys::WebGl2RenderingContext as Gl;

impl BrowserOwner {
    pub(crate) fn write_buffer_lease(
        &self,
        destination: &BrowserBuffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()> {
        self.drain_pending_deletes()?;
        if !Rc::ptr_eq(&self.state, &destination.owner.state) {
            return Err(Error::WrongDevice);
        }
        if offset
            .checked_add(bytes.len() as u64)
            .is_none_or(|end| end > destination.size)
            || offset > (1u64 << 53)
        {
            return Err(Error::InvalidRange);
        }
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        let raw = &state.context.raw;
        raw.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&destination.raw));
        raw.buffer_sub_data_with_f64_and_u8_array(Gl::COPY_WRITE_BUFFER, offset as f64, bytes);
        // Mapping leases are commonly flushed once per frame. The RHI owns
        // this context and every later copy binds its destination explicitly,
        // so neutralize the target instead of a synchronous getParameter.
        raw.bind_buffer(Gl::COPY_WRITE_BUFFER, None);
        if raw.is_context_lost() {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        // The publication boundary consumes the sticky error once for the
        // whole frame; checking here would turn every mapped-buffer update
        // into a synchronous round trip.
        Ok(())
    }
}
