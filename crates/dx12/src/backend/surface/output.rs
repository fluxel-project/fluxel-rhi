//! Resolves the host HWND's current monitor to an active DXGI output.

use fluxel_rhi_core::api::{ColorSpace, Result, SurfaceOutputInfo};
use windows::{
    Win32::Graphics::{
        Dxgi::{
            Common::{
                DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709, DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
                DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
            },
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_ERROR_NOT_FOUND, DXGI_OUTPUT_DESC1,
            IDXGIFactory4, IDXGIOutput6,
        },
        Gdi::{MONITOR_DEFAULTTONULL, MonitorFromWindow},
    },
    core::Interface,
};

use super::{SurfaceState, error};

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
