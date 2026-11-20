//! Pre-device Win32 target observations in the instance's adapter namespace.

use super::{Dx12, Dx12Instance, adapter::probe_d3d12, adapter_at, native_error, surface};
use fluxel_rhi_core::{
    api::{AdapterId, Result, SurfaceCompatibility, SurfaceProbeUncertainty},
    backend::SurfaceProbeBackend,
};
use raw_window_handle::WindowHandle;
use windows::Win32::Graphics::{
    Dxgi::{DXGI_ERROR_NOT_CURRENTLY_AVAILABLE, DXGI_ERROR_NOT_FOUND, DXGI_ERROR_UNSUPPORTED},
    Gdi::{MONITOR_DEFAULTTONULL, MonitorFromWindow},
};

impl SurfaceProbeBackend for Dx12 {
    type Target<'a> = WindowHandle<'a>;

    fn probe_surface(
        instance: &Dx12Instance,
        id: AdapterId,
        target: WindowHandle<'_>,
    ) -> Result<SurfaceCompatibility> {
        use SurfaceProbeUncertainty::*;

        let hwnd = surface::live_hwnd(target.as_raw())?;
        let adapter = adapter_at(instance, id.ordinal)?;
        match probe_d3d12(&adapter) {
            Ok(()) => {}
            Err(error) if error.code() == DXGI_ERROR_UNSUPPORTED => {
                return Ok(SurfaceCompatibility::Incompatible(
                    "D3D12 feature level 11_0",
                ));
            }
            Err(error) => return Err(native_error(error)),
        }

        // SAFETY: the host's borrowed WindowHandle keeps this HWND valid.
        let monitor = unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONULL) };
        // SAFETY: the queried factory remains owned by the instance.
        let current = unsafe { instance.factory.IsCurrent() }.as_bool();
        if monitor.is_invalid() || !current {
            return Ok(SurfaceCompatibility::Unknown(OutputUnavailable));
        }
        for index in 0.. {
            // SAFETY: the adapter is live and DXGI returns an owned reference.
            let output = match unsafe { adapter.EnumOutputs(index) } {
                Ok(output) => output,
                Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(error) if error.code() == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE => {
                    return Ok(SurfaceCompatibility::Unknown(OutputUnavailable));
                }
                Err(error) => return Err(native_error(error)),
            };
            // SAFETY: the output remains live while obtaining its descriptor.
            let desc = unsafe { output.GetDesc() }.map_err(native_error)?;
            if desc.Monitor == monitor && desc.AttachedToDesktop.as_bool() {
                return Ok(SurfaceCompatibility::Unknown(OutputOnAdapter));
            }
        }
        Ok(SurfaceCompatibility::Unknown(OutputNotReportedOnAdapter))
    }
}
