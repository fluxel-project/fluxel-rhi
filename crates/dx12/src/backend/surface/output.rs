//! Resolves the host HWND's current monitor to an active DXGI output.

use fluxel_rhi_core::api::{ColorSpace, Error, Result, SurfaceOutputInfo, TextureFormat};
use windows::{
    Win32::Graphics::{
        Dxgi::{
            Common::{
                DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709, DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
                DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, DXGI_COLOR_SPACE_TYPE,
            },
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_ERROR_NOT_FOUND, DXGI_OUTPUT_DESC1,
            DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT, IDXGIFactory4, IDXGIOutput6,
            IDXGISwapChain3,
        },
        Gdi::{MONITOR_DEFAULTTONULL, MonitorFromWindow},
    },
    core::Interface,
};

use super::{SurfaceState, error, swapchain_format};

pub(super) fn hdr_active(surface: &SurfaceState) -> Result<bool> {
    Ok(current_output(surface)?
        .is_some_and(|output| output.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020))
}

fn format_accepts_color_space(format: TextureFormat, color_space: ColorSpace) -> bool {
    match color_space {
        ColorSpace::Srgb => format != TextureFormat::Rgba16Float,
        ColorSpace::ExtendedLinearSrgb => format == TextureFormat::Rgba16Float,
        ColorSpace::Hdr10 => format == TextureFormat::Rgb10a2Unorm,
        ColorSpace::DisplayP3 => false,
    }
}

pub(super) fn supports_color_space(
    surface: &SurfaceState,
    format: TextureFormat,
    color_space: ColorSpace,
) -> bool {
    if !format_accepts_color_space(format, color_space) {
        return false;
    }
    // DXGI's flip formats use these color spaces by default. DWM can
    // compose an scRGB swapchain even on an SDR output.
    if color_space != ColorSpace::Hdr10 {
        return true;
    }
    if !hdr_active(surface).unwrap_or(false) {
        return false;
    }
    let chain = surface
        .inner
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .chain
        .clone();
    let Some(chain) = chain else { return true };
    // SAFETY: the cloned chain is live and GetDesc1 queries its descriptor only.
    let Ok(desc) = (unsafe { chain.GetDesc1() }) else {
        return false;
    };
    // A different prospective format can only be queried after ResizeBuffers.
    if Some(desc.Format) != swapchain_format(format) {
        return true;
    }
    native_color_space(color_space)
        .is_some_and(|native| supports_presentation(&chain, native).unwrap_or(false))
}

pub(super) fn set_swapchain_color_space(
    chain: &IDXGISwapChain3,
    color_space: ColorSpace,
) -> Result<()> {
    let native =
        native_color_space(color_space).ok_or(Error::Unsupported("DXGI surface color space"))?;
    if color_space == ColorSpace::Hdr10 && !supports_presentation(chain, native)? {
        return Err(Error::Unsupported("DXGI swapchain HDR presentation"));
    }
    // SAFETY: the new/resized chain is retained and no image is acquired
    // during configuration. `native` is a selected DXGI color space.
    unsafe { chain.SetColorSpace1(native) }.map_err(error)
}

fn native_color_space(color_space: ColorSpace) -> Option<DXGI_COLOR_SPACE_TYPE> {
    Some(match color_space {
        ColorSpace::Srgb => DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
        ColorSpace::ExtendedLinearSrgb => DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,
        ColorSpace::Hdr10 => DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
        ColorSpace::DisplayP3 => return None,
    })
}

fn supports_presentation(chain: &IDXGISwapChain3, native: DXGI_COLOR_SPACE_TYPE) -> Result<bool> {
    // SAFETY: callers retain the chain and the query does not mutate it.
    let support = unsafe { chain.CheckColorSpaceSupport(native) }.map_err(error)?;
    Ok(support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT.0 as u32 != 0)
}

pub(super) fn info(surface: &SurfaceState) -> Result<Option<SurfaceOutputInfo>> {
    Ok(current_output(surface)?.map(|output| SurfaceOutputInfo {
        color_space: match output.ColorSpace {
            DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709 => Some(ColorSpace::Srgb),
            DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709 => Some(ColorSpace::ExtendedLinearSrgb),
            DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 => Some(ColorSpace::Hdr10),
            _ => None,
        },
        bits_per_color: output.BitsPerColor,
        primaries: [output.RedPrimary, output.GreenPrimary, output.BluePrimary],
        white_point: output.WhitePoint,
        min_luminance_nits: output.MinLuminance,
        max_luminance_nits: output.MaxLuminance,
        max_full_frame_luminance_nits: output.MaxFullFrameLuminance,
        sdr_white_level_nits: (output.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020)
            .then(|| super::sdr_white::white_level(&output.DeviceName))
            .flatten(),
    }))
}

pub(super) fn current_output(surface: &SurfaceState) -> Result<Option<DXGI_OUTPUT_DESC1>> {
    // SAFETY: the surface's host contract keeps the HWND alive while we query it.
    let monitor = unsafe { MonitorFromWindow(surface.hwnd, MONITOR_DEFAULTTONULL) };
    if monitor.is_invalid() {
        return Ok(None);
    }
    // SAFETY: the factory is retained by the surface and the query has no output pointers.
    let factory = if unsafe { surface.factory.IsCurrent() }.as_bool() {
        surface.factory.clone()
    } else {
        // SAFETY: a fresh factory has no borrowed inputs. DXGI can invalidate
        // cached outputs after display topology or active color state changes.
        unsafe { CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .map_err(error)?
    };
    // Enumerate all desktop adapters. The adapter driving this HWND's
    // monitor may differ from the render adapter on hybrid systems.
    for adapter_index in 0.. {
        // SAFETY: the factory is live and DXGI returns an owned adapter reference.
        let adapter = match unsafe { factory.EnumAdapters1(adapter_index) } {
            Ok(adapter) => adapter,
            Err(reason) if reason.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(reason) => return Err(error(reason)),
        };
        for output_index in 0.. {
            // SAFETY: the retained adapter owns its output enumeration.
            let output = match unsafe { adapter.EnumOutputs(output_index) } {
                Ok(output) => output,
                Err(reason) if reason.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(reason) => return Err(error(reason)),
            };
            // SAFETY: the retained output writes its descriptor into a local return value.
            let desc = unsafe { output.GetDesc() }.map_err(error)?;
            if desc.Monitor == monitor && desc.AttachedToDesktop.as_bool() {
                let Ok(output) = output.cast::<IDXGIOutput6>() else {
                    return Ok(None);
                };
                // SAFETY: the matched output remains live for the description query.
                let desc = unsafe { output.GetDesc1() }.map_err(error)?;
                return Ok(
                    (desc.Monitor == monitor && desc.AttachedToDesktop.as_bool()).then_some(desc),
                );
            }
        }
    }
    Ok(None)
}
