//! Presents one current DXGI image and qualifies explicit damage rectangles.

use fluxel_rhi_core::{
    api::{Error, PresentMode, Rect, Result},
    backend::NativeImageLease,
};
use windows::{
    Win32::{
        Foundation::{RECT, S_OK},
        Graphics::Dxgi::{
            DXGI_PRESENT, DXGI_PRESENT_ALLOW_TEARING, DXGI_PRESENT_PARAMETERS, IDXGISwapChain3,
        },
    },
    core::HRESULT,
};

use super::{Dx12SurfaceImage, Inner, error, handle_surface_failure};

// DXGI status from dxgitype.h, omitted by the Windows bindings.
const DXGI_STATUS_MODE_CHANGED: HRESULT = HRESULT(0x087a_0007);

// SAFETY: SurfaceState serializes generation changes, handoff and lease drop.
// Each lease owns one acquired slot and releases it at most once.
unsafe impl NativeImageLease for Dx12SurfaceImage {
    fn try_handoff(&mut self, generation: u64) -> Result<()> {
        self.try_handoff_with_damage(generation, &[])
    }

    fn damage_support(&self, generation: u64) -> Result<()> {
        let inner = self.0.state.inner.lock().unwrap_or_else(|p| p.into_inner());
        self.require_current(&inner, generation)?;
        require_damage(&inner)
    }

    fn try_handoff_with_damage(&mut self, generation: u64, damage: &[Rect]) -> Result<()> {
        let mut inner = self.0.state.inner.lock().unwrap_or_else(|p| p.into_inner());
        self.require_current(&inner, generation)?;
        if !damage.is_empty() {
            require_damage(&inner)?;
        }
        let (sync, flags) = if inner.present_mode == PresentMode::Immediate {
            (0, DXGI_PRESENT_ALLOW_TEARING)
        } else {
            (1, DXGI_PRESENT(0))
        };
        let result = inner
            .chain
            .as_ref()
            .map_or(Err(Error::SurfaceLost), |chain| {
                present(chain, sync, flags, damage)
            });
        if let Err(error) = result {
            // An uncertain handoff cannot establish the previous frame that
            // DXGI needs for a later partial present.
            inner.full_presented = false;
            return Err(handle_surface_failure(&mut inner, error));
        }
        inner.full_presented = inner.preserved;
        inner.acquired = false;
        self.0.handed = true;
        Ok(())
    }

    fn is_current(&self, generation: u64) -> bool {
        let inner = self.0.state.inner.lock().unwrap_or_else(|p| p.into_inner());
        self.require_current(&inner, generation).is_ok()
    }
}

impl Dx12SurfaceImage {
    fn require_current(&self, inner: &Inner, generation: u64) -> Result<()> {
        if self.0.generation != generation
            || self.0.handed
            || inner.generation != generation
            || !inner.acquired
        {
            return Err(Error::SurfaceOutdated);
        }
        Ok(())
    }
}

fn require_damage(inner: &Inner) -> Result<()> {
    if !inner.preserved {
        return Err(Error::Unsupported("DXGI damage on a flip-discard surface"));
    }
    if !inner.full_presented {
        return Err(Error::InvalidState);
    }
    Ok(())
}

fn present(chain: &IDXGISwapChain3, sync: u32, flags: DXGI_PRESENT, damage: &[Rect]) -> Result<()> {
    let status = if damage.is_empty() {
        // SAFETY: the caller holds the surface lock and owns the current image.
        unsafe { chain.Present(sync, flags) }
    } else {
        let count = u32::try_from(damage.len()).map_err(|_| Error::InvalidRange)?;
        let mut rects = damage.iter().map(native_rect).collect::<Result<Vec<_>>>()?;
        let params = DXGI_PRESENT_PARAMETERS {
            DirtyRectsCount: count,
            pDirtyRects: rects.as_mut_ptr(),
            ..Default::default()
        };
        // SAFETY: the locked swapchain uses flip-sequential. The rectangle
        // storage remains live and DXGI reads it during the synchronous call.
        unsafe { chain.Present1(sync, flags, &params) }
    };
    if status == S_OK {
        Ok(())
    } else if status == DXGI_STATUS_MODE_CHANGED {
        Err(Error::SurfaceOutdated)
    } else if status.is_err() {
        Err(error(windows::core::Error::from_hresult(status)))
    } else {
        // Flip-model Present has no documented occluded success result.
        Err(Error::Backend(format!(
            "Unexpected DXGI Present status: {status:?}"
        )))
    }
}

fn native_rect(rect: &Rect) -> Result<RECT> {
    let edge = |position: u32, extent: u32| {
        position
            .checked_add(extent)
            .and_then(|end| i32::try_from(end).ok())
            .ok_or(Error::InvalidRange)
    };
    Ok(RECT {
        left: edge(rect.x, 0)?,
        top: edge(rect.y, 0)?,
        right: edge(rect.x, rect.width)?,
        bottom: edge(rect.y, rect.height)?,
    })
}
