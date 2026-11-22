//! Explicit HDR10 metadata for a configured DXGI surface generation.

use fluxel_rhi_core::api::{ColorSpace, Error, Hdr10Metadata, Result, Support};
use windows::{Win32::Graphics::Dxgi::*, core::Interface};

use super::{Inner, SurfaceState, error, output};

const UNSUPPORTED: &str = "DXGI HDR10 surface metadata";

fn configured_chain(inner: &Inner, generation: u64) -> Result<&IDXGISwapChain3> {
    if inner.generation != generation {
        return Err(Error::SurfaceOutdated);
    }
    if inner.color_space != ColorSpace::Hdr10 {
        return Err(Error::Unsupported(UNSUPPORTED));
    }
    inner.chain.as_ref().ok_or(Error::SurfaceLost)
}

pub(super) fn support(surface: &SurfaceState, generation: u64) -> Support<()> {
    let inner = surface.inner.lock().unwrap_or_else(|p| p.into_inner());
    let ready = configured_chain(&inner, generation).is_ok_and(|chain| {
        chain.cast::<IDXGISwapChain4>().is_ok()
            && surface.device.require_active().is_ok()
            && output::hdr10_presentation_supported(surface, chain).unwrap_or(false)
    });
    if ready {
        Support::Supported(())
    } else {
        Support::Unsupported(UNSUPPORTED)
    }
}

pub(super) fn set(
    surface: &SurfaceState,
    generation: u64,
    value: Option<&Hdr10Metadata>,
) -> Result<()> {
    let mut inner = surface.inner.lock().unwrap_or_else(|p| p.into_inner());
    let chain = configured_chain(&inner, generation)?;
    surface.device.require_active()?;
    if let Some(value) = value {
        if !output::hdr10_presentation_supported(surface, chain)? {
            return Err(Error::Unsupported(UNSUPPORTED));
        }
        let chain = metadata_chain(chain)?;
        let data = native_payload(value);
        const _: () = assert!(std::mem::size_of::<DXGI_HDR_METADATA_HDR10>() == 28);
        // SAFETY: the C payload has 28 bytes of initialized fields and no
        // padding. DXGI consumes the borrowed bytes during this call.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(&data).cast(),
                std::mem::size_of_val(&data),
            )
        };
        // SAFETY: the locked chain is current and bytes match the HDR10 type.
        unsafe { chain.SetHDRMetaData(DXGI_HDR_METADATA_TYPE_HDR10, Some(bytes)) }
            .map_err(error)?;
    } else {
        clear(chain)?;
    }
    inner.hdr10_metadata_set = value.is_some();
    Ok(())
}

pub(super) fn clear(chain: &IDXGISwapChain3) -> Result<()> {
    let chain = metadata_chain(chain)?;
    // SAFETY: NONE requires a null payload and callers retain the swapchain.
    unsafe { chain.SetHDRMetaData(DXGI_HDR_METADATA_TYPE_NONE, None) }.map_err(error)
}

fn metadata_chain(chain: &IDXGISwapChain3) -> Result<IDXGISwapChain4> {
    chain.cast().map_err(|_| Error::Unsupported(UNSUPPORTED))
}

fn native_payload(value: &Hdr10Metadata) -> DXGI_HDR_METADATA_HDR10 {
    let [red, green, blue] = value.primaries;
    DXGI_HDR_METADATA_HDR10 {
        RedPrimary: red,
        GreenPrimary: green,
        BluePrimary: blue,
        WhitePoint: value.white_point,
        MaxMasteringLuminance: value.max_mastering_nits,
        MinMasteringLuminance: value.min_mastering_nits_x10000,
        MaxContentLightLevel: value.max_content_nits,
        MaxFrameAverageLightLevel: value.max_frame_average_nits,
    }
}
