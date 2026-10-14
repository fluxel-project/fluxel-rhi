//! Synchronous readback tickets for the owner-thread GL backend.
//!
//! GL command verbs execute at the call site, so buffer readback copies bytes
//! into a ticket immediately. Core still requires a covering completion point
//! before exposing the ticket, preserving the public lifetime contract.

use std::{future::Ready, ops::Range, rc::Rc};

use fluxel_rhi_core::api::{Error, ReadbackBackend, ReadbackData, Result};

use super::{
    Gl,
    spi::native_error,
    types::{GlBuffer, GlDevice, GlDeviceState, GlEncoder},
};

/// CPU bytes captured at the exact immediate command position.
pub struct GlReadbackTicket {
    _state: Rc<GlDeviceState>,
    bytes: Vec<u8>,
}

impl ReadbackBackend for Gl {
    type NativeReadback = GlReadbackTicket;
    type ReadbackFuture<'a> = Ready<Result<ReadbackData>>;

    fn encode_readback(
        encoder: &mut GlEncoder,
        source: &GlBuffer,
        range: Range<u64>,
    ) -> Result<GlReadbackTicket> {
        if !Rc::ptr_eq(&encoder.state, &source.state)
            || !(source
                .desc
                .usage
                .contains(crate::backend::gl::api::GlBufferUsage::COPY_SOURCE)
                || source
                    .desc
                    .usage
                    .contains(crate::backend::gl::api::GlBufferUsage::MAP_READ))
            || range.start > range.end
            || range.end > source.desc.size
        {
            return Err(Error::InvalidRange);
        }
        let size = range
            .end
            .checked_sub(range.start)
            .ok_or(Error::InvalidRange)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::{GlBufferRange, GlCopyDomainApi as _};
            let result = encoder.state.with_native("read GL buffer", |provider| {
                provider.read_buffer(GlBufferRange {
                    buffer: source.id,
                    offset: range.start,
                    size,
                })
            });
            let bytes = match result {
                Ok(bytes) => bytes,
                Err(error) => {
                    encoder.indeterminate = true;
                    return Err(native_error(error));
                }
            };
            encoder.issued = true;
            return Ok(GlReadbackTicket {
                _state: encoder.state.clone(),
                bytes,
            });
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            let _ = (encoder, source, range, size);
            Err(Error::Unsupported("GL buffer readback platform"))
        }
    }

    fn map_readback<'a>(_: &'a GlDevice, ticket: GlReadbackTicket) -> Self::ReadbackFuture<'a> {
        let len = ticket.bytes.len() as u64;
        std::future::ready(Ok(ReadbackData {
            bytes: ticket.bytes,
            row_pitch: len,
            slice_pitch: len,
        }))
    }
}
