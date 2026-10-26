//! Direct browser staging for asynchronous host buffer maps.
use super::*;
use wasm_bindgen::JsCast;

const MAX_EXACT_WEBGL_OFFSET: u64 = (1_u64 << 53) - 1;

fn bound_buffer(raw: &Gl, target: u32) -> Option<web_sys::WebGlBuffer> {
    raw.get_parameter(target).ok()?.dyn_into().ok()
}

impl BrowserOwner {
    /// Copies one source range into a ticket-owned staging buffer and fences
    /// that copy. The returned completion is the sole GPU wait; callers do
    /// not retain a CPU command stream.
    pub(crate) fn stage_map_read(
        &self,
        source: &BrowserBuffer,
        offset: u64,
        size: u64,
    ) -> Result<(BrowserBuffer, super::super::completion::BrowserCompletion)> {
        if !Rc::ptr_eq(&self.state, &source.owner.state) {
            return Err(Error::WrongDevice);
        }
        if offset.checked_add(size).is_none_or(|end| end > source.size) {
            return Err(Error::InvalidRange);
        }
        if offset > MAX_EXACT_WEBGL_OFFSET
            || size > MAX_EXACT_WEBGL_OFFSET
            || offset
                .checked_add(size)
                .is_none_or(|end| end > MAX_EXACT_WEBGL_OFFSET)
        {
            return Err(Error::InvalidRange);
        }
        let staging = self.create_buffer(crate::backend::api::GlBufferDesc {
            size,
            usage: crate::backend::api::GlBufferUsage::COPY_DESTINATION
                | crate::backend::api::GlBufferUsage::MAP_READ,
        })?;
        {
            let mut state = self.state.borrow_mut();
            state.ensure_active()?;
            if state.active_encoder.is_some() {
                return Err(Error::InvalidState);
            }
            let raw = &state.context.raw;
            let prior_read = bound_buffer(raw, Gl::COPY_READ_BUFFER_BINDING);
            let prior_write = bound_buffer(raw, Gl::COPY_WRITE_BUFFER_BINDING);
            raw.bind_buffer(Gl::COPY_READ_BUFFER, Some(&source.raw));
            raw.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&staging.raw));
            raw.copy_buffer_sub_data_with_f64_and_f64_and_f64(
                Gl::COPY_READ_BUFFER,
                Gl::COPY_WRITE_BUFFER,
                offset as f64,
                0.0,
                size as f64,
            );
            let lost = raw.is_context_lost();
            let error = raw.get_error();
            raw.bind_buffer(Gl::COPY_READ_BUFFER, prior_read.as_ref());
            raw.bind_buffer(Gl::COPY_WRITE_BUFFER, prior_write.as_ref());
            if lost {
                state.mark_lost();
                return Err(Error::DeviceLost);
            }
            if error != Gl::NO_ERROR {
                return Err(Error::Backend("WebGL2 map staging copy failed".into()));
            }
        }
        let completion = self.completion_after_current_commands()?;
        Ok((staging, completion))
    }
}
