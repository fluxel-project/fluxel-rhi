//! Synchronous readback tickets for the owner-thread GL backend.
//!
//! GL command verbs execute at the call site. Native paths capture their bytes
//! immediately; WebGL2 copies into ticket-owned GPU staging at this command
//! position and reads those immutable bytes after the completion proof.

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
    source: GlReadbackSource,
}

enum GlReadbackSource {
    Bytes(Vec<u8>),
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    Browser {
        buffer: crate::backend::gl::browser::BrowserBuffer,
        offset: u64,
        size: u64,
    },
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
        if size == 0 {
            return Ok(GlReadbackTicket {
                _state: encoder.state.clone(),
                source: GlReadbackSource::Bytes(Vec::new()),
            });
        }
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
                source: GlReadbackSource::Bytes(bytes),
            });
        }
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            use crate::backend::gl::api::{GlBufferDesc, GlBufferUsage};
            let staging = encoder.state.owner.create_buffer(GlBufferDesc {
                size,
                usage: GlBufferUsage::COPY_DESTINATION | GlBufferUsage::MAP_READ,
            })?;
            let result = encoder
                .browser
                .as_mut()
                .ok_or(Error::InvalidState)?
                .copy_browser_buffer(&source.browser, range.start, &staging, 0, size);
            if let Err(error) = result {
                encoder.indeterminate = true;
                return Err(error);
            }
            encoder.issued = true;
            return Ok(GlReadbackTicket {
                _state: encoder.state.clone(),
                source: GlReadbackSource::Browser {
                    buffer: staging,
                    offset: 0,
                    size,
                },
            });
        }
        #[cfg(not(any(
            all(windows, feature = "native-gl-wgl"),
            feature = "native-gles-egl",
            all(target_arch = "wasm32", feature = "webgl2")
        )))]
        {
            let _ = (encoder, source, range, size);
            Err(Error::Unsupported("GL buffer readback platform"))
        }
    }

    fn map_readback<'a>(
        device: &'a GlDevice,
        ticket: GlReadbackTicket,
    ) -> Self::ReadbackFuture<'a> {
        let result = match ticket.source {
            GlReadbackSource::Bytes(bytes) => Ok(bytes),
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            GlReadbackSource::Browser {
                buffer,
                offset,
                size,
            } => device.0.owner.read_buffer(&buffer, offset, size),
        };
        std::future::ready(result.map(|bytes| {
            let len = bytes.len() as u64;
            ReadbackData {
                bytes,
                row_pitch: len,
                slice_pitch: len,
            }
        }))
    }
}
