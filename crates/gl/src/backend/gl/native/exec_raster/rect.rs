//! GL framebuffer coordinate conversion for raster setters.
//!
//! Converts public upper-left rectangles at the GL boundary.

use super::{GlError, NativeGlProvider};

/// Converts one Core upper-left rectangle into GL's lower-left integer space.
///
/// The height is copied from the *currently active* render pass immediately
/// before issuing state. It is deliberately not kept in raster state: a later
/// pass may bind another framebuffer or a resized drawable with a new extent.
pub(super) fn native_rect(
    operation: &'static str,
    framebuffer_height: u32,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<(i32, i32, i32, i32), GlError> {
    let upper_edge = y
        .checked_add(height)
        .ok_or_else(|| NativeGlProvider::validation(operation, "rectangle y range overflows"))?;
    let gl_y = framebuffer_height.checked_sub(upper_edge).ok_or_else(|| {
        NativeGlProvider::validation(operation, "rectangle exceeds active framebuffer height")
    })?;
    Ok((
        i32::try_from(x)
            .map_err(|_| NativeGlProvider::validation(operation, "rectangle x exceeds GLint"))?,
        i32::try_from(gl_y)
            .map_err(|_| NativeGlProvider::validation(operation, "rectangle y exceeds GLint"))?,
        i32::try_from(width).map_err(|_| {
            NativeGlProvider::validation(operation, "rectangle width exceeds GLsizei")
        })?,
        i32::try_from(height).map_err(|_| {
            NativeGlProvider::validation(operation, "rectangle height exceeds GLsizei")
        })?,
    ))
}
