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
mod tests {
    use super::*;

    fn valid_viewport() -> api::Viewport {
        api::Viewport {
            x: 2.0,
            y: 3.0,
            width: 4.0,
            height: 5.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    #[test]
    fn validates_viewport_bounds_before_native_lowering() {
        assert!(viewport(valid_viewport()).is_ok());
        for candidate in [
            api::Viewport {
                width: 0.0,
                ..valid_viewport()
            },
            api::Viewport {
                min_depth: -0.1,
                ..valid_viewport()
            },
            api::Viewport {
                max_depth: 1.1,
                ..valid_viewport()
            },
            api::Viewport {
                min_depth: 0.8,
                max_depth: 0.2,
                ..valid_viewport()
            },
            api::Viewport {
                x: f32::NAN,
                ..valid_viewport()
            },
        ] {
            assert!(matches!(viewport(candidate), Err(Error::InvalidRange)));
        }
    }

    #[test]
    fn validates_scissor_representability_before_native_lowering() {
        assert!(
            scissor(api::Rect {
                x: 2,
                y: 3,
                width: 4,
                height: 5,
            })
            .is_ok()
        );
        for rect in [
            api::Rect {
                x: 0,
                y: 0,
                width: 0,
                height: 1,
            },
            api::Rect {
                x: i32::MAX as u32,
                y: 0,
                width: 1,
                height: 1,
            },
            api::Rect {
                x: u32::MAX,
                y: 0,
                width: 1,
                height: 1,
            },
        ] {
            assert!(matches!(scissor(rect), Err(Error::InvalidRange)));
        }
    }
}
