//! Synchronous WebGL transfer primitives for CPU-owned mapping leases.
//!
//! Host memory is a snapshot; only explicitly flushed ranges are uploaded.
use super::owner::{BrowserBuffer, BrowserOwner};
use fluxel_rhi_core::api::{Error, Result};
use std::rc::Rc;
use wasm_bindgen::JsCast;
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
        let previous = raw
            .get_parameter(Gl::COPY_WRITE_BUFFER_BINDING)
            .ok()
            .and_then(|v| v.dyn_into::<web_sys::WebGlBuffer>().ok());
        raw.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&destination.raw));
        raw.buffer_sub_data_with_f64_and_u8_array(Gl::COPY_WRITE_BUFFER, offset as f64, bytes);
        raw.bind_buffer(Gl::COPY_WRITE_BUFFER, previous.as_ref());
        if raw.is_context_lost() {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        let error = raw.get_error();
        match error {
            Gl::NO_ERROR => Ok(()),
            Gl::OUT_OF_MEMORY => Err(Error::OutOfMemory),
            _ => Err(Error::Backend(format!(
                "WebGL2 mapping flush failed with GL error {error}"
            ))),
        }
    }
}
