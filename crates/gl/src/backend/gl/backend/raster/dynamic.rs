//! Immediate GL viewport and scissor lowering.
//!
//! These are mutable context state, so this module validates Core values then
//! applies them directly. It owns neither pass lifetime nor any draw record.

use fluxel_rhi_core::api::{Error, Rect, Result, Viewport};

use crate::backend::gl::backend::spi::native_error;
use crate::backend::gl::backend::types::GlEncoder;

pub(super) fn set_viewport(encoder: &mut GlEncoder, viewport: Viewport) -> Result<()> {
    if encoder.raster_extent.is_none() {
        return Err(Error::InvalidState);
    }
    let native = crate::backend::gl::api::GlViewport {
        x: exact_u32(viewport.x)?,
        y: exact_u32(viewport.y)?,
        width: exact_u32(viewport.width)?,
        height: exact_u32(viewport.height)?,
        min_depth: viewport.min_depth.to_bits(),
        max_depth: viewport.max_depth.to_bits(),
    };
    if !viewport.min_depth.is_finite()
        || !viewport.max_depth.is_finite()
        || !(0.0..=1.0).contains(&viewport.min_depth)
        || !(0.0..=1.0).contains(&viewport.max_depth)
        || viewport.min_depth > viewport.max_depth
    {
        return Err(Error::InvalidRange);
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlRasterCommandApi as _;
        let result = encoder
            .state
            .with_native("set GL viewport", |provider| provider.set_viewport(native))
            .map_err(native_error);
        if result.is_ok() {
            encoder.issued = true;
        } else {
            encoder.indeterminate = true;
        }
        result
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            let result = encoder
                .browser
                .as_mut()
                .ok_or(Error::InvalidState)?
                .set_viewport(viewport);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        {
            let _ = native;
            Err(Error::Unsupported("GL platform context"))
        }
    }
}

pub(super) fn set_scissor(encoder: &mut GlEncoder, scissor: Rect) -> Result<()> {
    if encoder.raster_extent.is_none() {
        return Err(Error::InvalidState);
    }
    let native = crate::backend::gl::api::GlScissorRect {
        x: scissor.x,
        y: scissor.y,
        width: scissor.width,
        height: scissor.height,
    };
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlRasterCommandApi as _;
        let result = encoder
            .state
            .with_native("set GL scissor", |provider| {
                provider.set_scissor(Some(native))
            })
            .map_err(native_error);
        if result.is_ok() {
            encoder.issued = true;
        } else {
            encoder.indeterminate = true;
        }
        result
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            let result = encoder
                .browser
                .as_mut()
                .ok_or(Error::InvalidState)?
                .set_scissor(scissor);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        {
            let _ = native;
            Err(Error::Unsupported("GL platform context"))
        }
    }
}

/// The common-profile provider represents viewport coordinates as integer
/// pixels. Reject fractional values instead of silently changing Core state.
fn exact_u32(value: f32) -> Result<u32> {
    if !value.is_finite() || value < 0.0 || value > u32::MAX as f32 || value.fract() != 0.0 {
        return Err(Error::InvalidRange);
    }
    Ok(value as u32)
}
