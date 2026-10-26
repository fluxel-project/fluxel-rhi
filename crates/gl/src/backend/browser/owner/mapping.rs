//! Direct browser staging for asynchronous host buffer maps.
use super::*;

const MAX_EXACT_WEBGL_OFFSET: u64 = (1_u64 << 53) - 1;

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
            // This context is exclusively owned by the RHI.  Do not query
            // driver state merely to restore it: later transfer verbs bind
            // their operands explicitly, so leave these private targets
            // neutral after the staging copy.
            raw.bind_buffer(Gl::COPY_READ_BUFFER, None);
            raw.bind_buffer(Gl::COPY_WRITE_BUFFER, None);
            if lost {
                state.mark_lost();
                return Err(Error::DeviceLost);
            }
        }
        let completion = self.completion_after_current_commands()?;
        Ok((staging, completion))
    }
}
