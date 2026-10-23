//! Optional raster operations issued on the immediate GL context.
//!
//! The static type exposes the Core SPI shape, while per-context discovery
//! controls whether `Device::raster_extras` is available. A supported indirect
//! draw validates one argument record and immediately invokes the native
//! provider. Multi-draw issues each GPU indirect record immediately; the count
//! variant uses the native parameter-buffer entry point without CPU readback.

use fluxel_rhi_core::api::{self, Error, RasterExtrasOps, Result};

use super::types::{Gl, GlBuffer, GlEncoder};

/// Native GL indirect draw operations; unsupported optional verbs stay closed.
pub struct GlExtras;

impl RasterExtrasOps<Gl> for GlExtras {
    const AVAILABLE: bool = true;

    fn draw_indirect(
        encoder: &mut GlEncoder,
        arguments: &GlBuffer,
        offset: u64,
        indexed: bool,
    ) -> Result<()> {
        let record_size = if indexed { 20_u64 } else { 16_u64 };
        validate(encoder, arguments, offset, 1, record_size as u32, indexed)?;
        super::raster::cube_mirror::refresh(encoder)?;

        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::{
                GlBufferRange, GlDrawIndirectApi as _, GlIndirectAbi, GlIndirectCommandRange,
            };

            let command = GlIndirectCommandRange {
                range: GlBufferRange {
                    buffer: arguments.id,
                    offset: 0,
                    size: arguments.desc.size & !3,
                },
                command_offset: offset,
                draw_count: 1,
                stride: record_size as u32,
                abi: if indexed {
                    GlIndirectAbi::Indexed
                } else {
                    GlIndirectAbi::NonIndexed
                },
            };
            let result = encoder
                .state
                .with_native("draw GL indirect", |provider| {
                    provider.draw_indirect(command)
                })
                .map_err(indirect_error);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }

        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            let _ = (encoder, arguments, offset, indexed);
            Err(Error::Unsupported("GL indirect draw platform"))
        }
    }

    fn multi_draw(
        encoder: &mut GlEncoder,
        arguments: &GlBuffer,
        offset: u64,
        count: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<()> {
        validate(encoder, arguments, offset, count, stride, indexed)?;
        if count == 0 {
            return Ok(());
        }
        super::raster::cube_mirror::refresh(encoder)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::{
                GlBufferRange, GlDrawIndirectApi as _, GlIndirectAbi, GlIndirectCommandRange,
            };
            let result = encoder
                .state
                .with_native("multi draw GL indirect", |provider| {
                    for i in 0..count {
                        provider.draw_indirect(GlIndirectCommandRange {
                            range: GlBufferRange {
                                buffer: arguments.id,
                                offset: 0,
                                size: arguments.desc.size & !3,
                            },
                            command_offset: offset + u64::from(i) * u64::from(stride),
                            draw_count: 1,
                            stride,
                            abi: if indexed {
                                GlIndirectAbi::Indexed
                            } else {
                                GlIndirectAbi::NonIndexed
                            },
                        })?;
                    }
                    Ok(())
                })
                .map_err(indirect_error);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        Err(Error::Unsupported("GL multi-draw indirect platform"))
    }

    fn multi_draw_count(
        encoder: &mut GlEncoder,
        args: &GlBuffer,
        args_offset: u64,
        count: &GlBuffer,
        count_offset: u64,
        max: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<()> {
        validate(encoder, args, args_offset, max, stride, indexed)?;
        if !std::rc::Rc::ptr_eq(&encoder.state, &count.state) {
            return Err(Error::WrongDevice);
        }
        if !count
            .desc
            .usage
            .contains(crate::backend::gl::api::GlBufferUsage::INDIRECT)
        {
            return Err(Error::InvalidState);
        }
        if count_offset % 4 != 0
            || count_offset
                .checked_add(4)
                .is_none_or(|n| n > count.desc.size)
        {
            return Err(Error::InvalidRange);
        }
        if max == 0 {
            return Ok(());
        }
        super::raster::cube_mirror::refresh(encoder)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::{
                GlBufferRange, GlIndirectAbi, GlIndirectCommandRange, GlIndirectCountRange,
            };
            let command = GlIndirectCommandRange {
                range: GlBufferRange {
                    buffer: args.id,
                    offset: 0,
                    size: args.desc.size & !3,
                },
                command_offset: args_offset,
                draw_count: max,
                stride,
                abi: if indexed {
                    GlIndirectAbi::Indexed
                } else {
                    GlIndirectAbi::NonIndexed
                },
            };
            let count = GlIndirectCountRange {
                range: GlBufferRange {
                    buffer: count.id,
                    offset: 0,
                    size: count.desc.size & !3,
                },
                count_offset,
                max_draw_count: max,
            };
            let result = encoder
                .state
                .with_native("multi draw GL indirect count", |provider| {
                    provider.multi_draw_count(command, count)
                })
                .map_err(indirect_error);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        Err(Error::Unsupported("GL multi-draw indirect count platform"))
    }

    fn set_push_constants(
        encoder: &mut GlEncoder,
        stages: api::StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> Result<()> {
        super::push::write(encoder, stages, offset, bytes, false)
    }
}

fn validate(
    encoder: &GlEncoder,
    buffer: &GlBuffer,
    offset: u64,
    count: u32,
    stride: u32,
    indexed: bool,
) -> Result<()> {
    use crate::backend::gl::api::GlBufferUsage;
    if !std::rc::Rc::ptr_eq(&encoder.state, &buffer.state) {
        return Err(Error::WrongDevice);
    }
    if encoder.raster_extent.is_none() || encoder.raster_pipeline.is_none() {
        return Err(Error::InvalidState);
    }
    if !buffer.desc.usage.contains(GlBufferUsage::INDIRECT) {
        return Err(Error::InvalidState);
    }
    let size = if indexed { 20_u64 } else { 16_u64 };
    if offset % 4 != 0 || (count > 0 && (u64::from(stride) < size || stride % 4 != 0)) {
        return Err(Error::InvalidRange);
    }
    let end = if count == 0 {
        Some(offset)
    } else {
        offset
            .checked_add(u64::from(count - 1) * u64::from(stride))
            .and_then(|n| n.checked_add(size))
    };
    if end.is_none_or(|end| end > buffer.desc.size || end > i32::MAX as u64) {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
fn indirect_error(error: crate::backend::gl::api::GlError) -> Error {
    match error {
        crate::backend::gl::api::GlError::Unsupported { reason, .. } => Error::Unsupported(reason),
        other => super::spi::native_error(other),
    }
}
