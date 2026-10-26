//! Validation and D3D12 lowering for public framebuffer coordinates.
//!
//! Keeping these conversions here makes direct lists and bundle recorders
//! reject exactly the same invalid public coordinates before native calls.

use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::{Foundation::RECT, Graphics::Direct3D12::D3D12_VIEWPORT};

pub(crate) fn viewport(viewport: api::Viewport) -> Result<D3D12_VIEWPORT> {
    if !viewport.x.is_finite()
        || !viewport.y.is_finite()
        || !viewport.width.is_finite()
        || !viewport.height.is_finite()
        || !viewport.min_depth.is_finite()
        || !viewport.max_depth.is_finite()
        || viewport.width <= 0.0
        || viewport.height <= 0.0
        || viewport.min_depth < 0.0
        || viewport.min_depth > viewport.max_depth
        || viewport.max_depth > 1.0
    {
        return Err(Error::InvalidRange);
    }
    Ok(D3D12_VIEWPORT {
        TopLeftX: viewport.x,
        TopLeftY: viewport.y,
        Width: viewport.width,
        Height: viewport.height,
        MinDepth: viewport.min_depth,
        MaxDepth: viewport.max_depth,
    })
}

pub(crate) fn scissor(rect: api::Rect) -> Result<RECT> {
    if rect.width == 0 || rect.height == 0 {
        return Err(Error::InvalidRange);
    }
    let right = rect.x.checked_add(rect.width).ok_or(Error::InvalidRange)?;
    let bottom = rect.y.checked_add(rect.height).ok_or(Error::InvalidRange)?;
    if right > i32::MAX as u32 || bottom > i32::MAX as u32 {
        return Err(Error::InvalidRange);
    }
    Ok(RECT {
        left: rect.x as i32,
        top: rect.y as i32,
        right: right as i32,
        bottom: bottom as i32,
    })
}

#[cfg(test)]
#[path = "../../tests/backend/encoder/coordinates.rs"]
mod tests;
