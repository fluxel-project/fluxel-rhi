//! Optional raster operations issued on the immediate GL context.
//!
//! The static type exposes the Core SPI shape, while per-context discovery
//! controls whether `Device::raster_extras` is available. A supported indirect
//! draw validates one argument record and immediately invokes the native
//! provider; this module owns no draw list, replay path, or multi-draw fallback.

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
        if encoder.raster_extent.is_none()
            || encoder.raster_pipeline.is_none()
            || !std::rc::Rc::ptr_eq(&encoder.state, &arguments.state)
        {
            return Err(Error::InvalidState);
        }

        let record_size = if indexed { 20_u64 } else { 16_u64 };
        if offset % 4 != 0
            || offset
                .checked_add(record_size)
                .is_none_or(|end| end > arguments.desc.size)
        {
            return Err(Error::InvalidRange);
        }

        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::{
                GlBufferRange, GlDrawIndirectApi as _, GlIndirectAbi, GlIndirectCommandRange,
            };

            let command = GlIndirectCommandRange {
                range: GlBufferRange {
                    buffer: arguments.id,
                    offset: 0,
                    size: arguments.desc.size,
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
                .map_err(super::spi::native_error);
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

    fn multi_draw(_: &mut GlEncoder, _: &GlBuffer, _: u64, _: u32, _: u32, _: bool) -> Result<()> {
        Err(Error::Unsupported("GL multi-draw indirect"))
    }

    fn multi_draw_count(
        _: &mut GlEncoder,
        _: &GlBuffer,
        _: u64,
        _: &GlBuffer,
        _: u64,
        _: u32,
        _: u32,
        _: bool,
    ) -> Result<()> {
        Err(Error::Unsupported("GL multi-draw indirect count"))
    }

    fn set_push_constants(_: &mut GlEncoder, _: api::StageMask, _: u32, _: &[u8]) -> Result<()> {
        Err(Error::Unsupported("GL raster push constants"))
    }
}
