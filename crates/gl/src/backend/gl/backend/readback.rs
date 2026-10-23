//! Staged readback tickets for the owner-thread GL backend.
//!
//! GL command verbs execute at the call site. Native paths capture their bytes
//! into ticket-owned GPU staging at this command position and read those
//! immutable bytes only after the caller supplies a covering completion proof.

use std::{future::Ready, ops::Range, rc::Rc};

use fluxel_rhi_core::api::{Error, ReadbackBackend, ReadbackData, Result};

use super::{
    Gl,
    spi::native_error,
    types::{GlBuffer, GlDevice, GlDeviceState, GlEncoder},
};

/// Ticket-owned storage captured at the exact immediate command position.
pub struct GlReadbackTicket {
    _state: Rc<GlDeviceState>,
    source: GlReadbackSource,
}

enum GlReadbackSource {
    Bytes(Vec<u8>),
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    Native {
        buffer: GlBuffer,
        size: u64,
    },
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
            use super::types::GlBufferInner;
            use crate::backend::gl::api::{
                GlBufferDesc, GlBufferRange, GlBufferUsage, GlCopyDomainApi as _,
                GlResourceApi as _,
            };
            let desc = GlBufferDesc {
                size,
                usage: GlBufferUsage::COPY_DESTINATION | GlBufferUsage::MAP_READ,
            };
            let id = encoder
                .state
                .with_native("create readback staging", |provider| {
                    provider.create_buffer_resource(desc)
                })
                .map_err(native_error)?;
            let buffer = GlBuffer::from_inner(GlBufferInner {
                state: encoder.state.clone(),
                id,
                desc,
            });
            let result = encoder
                .state
                .with_native("stage GL buffer readback", |provider| {
                    provider.copy_buffer_range(
                        GlBufferRange {
                            buffer: source.id,
                            offset: range.start,
                            size,
                        },
                        GlBufferRange {
                            buffer: buffer.id,
                            offset: 0,
                            size,
                        },
                    )
                });
            if let Err(error) = result {
                encoder.indeterminate = true;
                return Err(native_error(error));
            }
            encoder.issued = true;
            return Ok(GlReadbackTicket {
                _state: encoder.state.clone(),
                source: GlReadbackSource::Native { buffer, size },
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
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            GlReadbackSource::Native { buffer, size } => {
                use crate::backend::gl::api::{GlBufferRange, GlCopyDomainApi as _};
                if !Rc::ptr_eq(&device.0, &buffer.state) {
                    Err(Error::WrongDevice)
                } else {
                    device
                        .0
                        .with_native("map GL readback staging", |provider| {
                            provider.read_buffer(GlBufferRange {
                                buffer: buffer.id,
                                offset: 0,
                                size,
                            })
                        })
                        .map_err(native_error)
                }
            }
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
