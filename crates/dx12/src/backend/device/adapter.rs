//! DXGI adapter discovery and selection in the ordinary ordinal namespace.
//! Preferences never change the identity used when opening an adapter.

use super::*;

use windows::Win32::Graphics::Dxgi::{
    DXGI_ADAPTER_DESC1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND,
    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, DXGI_GPU_PREFERENCE_MINIMUM_POWER, IDXGIAdapter1,
    IDXGIDevice, IDXGIFactory4, IDXGIFactory6,
};

pub(super) const REQUESTED_ADAPTER_UNAVAILABLE: &str = "the requested DXGI adapter";
// Software selection must not reuse the ordinal of a physical DXGI adapter.
const WARP_ORDINAL: u64 = u32::MAX as u64 + 1;

/// Resolves a public DXGI ordinal without narrowing an untrusted `u64`.
///
/// DXGI ordinals are dense `u32` values and end with `DXGI_ERROR_NOT_FOUND`.
/// Keeping that translation here makes `enumerate` and `open` agree on both
/// the ordinal namespace and the structured error returned for stale IDs.
pub(super) fn adapter_at(instance: &Dx12Instance, ordinal: u64) -> Result<IDXGIAdapter1> {
    if ordinal == WARP_ORDINAL {
        let factory = instance
            .factory
            .cast::<IDXGIFactory4>()
            .map_err(|_| Error::Unsupported("DXGI WARP selection"))?;
        // SAFETY: the factory is live and DXGI returns an owned adapter reference.
        return unsafe { factory.EnumWarpAdapter() }.map_err(native_error);
    }
    let ordinal =
        u32::try_from(ordinal).map_err(|_| Error::Unsupported(REQUESTED_ADAPTER_UNAVAILABLE))?;
    // SAFETY: the factory is owned by `instance`; the returned COM reference
    // is transferred to the caller on success.
    match unsafe { instance.factory.EnumAdapters1(ordinal) } {
        Ok(adapter) => Ok(adapter),
        Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => {
            Err(Error::Unsupported(REQUESTED_ADAPTER_UNAVAILABLE))
        }
        Err(error) => Err(native_error(error)),
    }
}

fn adapter_desc(adapter: &IDXGIAdapter1) -> Result<DXGI_ADAPTER_DESC1> {
    // SAFETY: `adapter` is a live COM interface and GetDesc1 only writes an
    // initialized descriptor returned by the Windows binding.
    unsafe { adapter.GetDesc1() }.map_err(native_error)
}

fn desc_name(desc: &DXGI_ADAPTER_DESC1) -> String {
    let end = desc
        .Description
        .iter()
        .position(|x| *x == 0)
        .unwrap_or(desc.Description.len());
    String::from_utf16_lossy(&desc.Description[..end])
}

pub(super) fn adapter_name(adapter: &IDXGIAdapter1) -> Result<String> {
    Ok(desc_name(&adapter_desc(adapter)?))
}

fn is_software(desc: &DXGI_ADAPTER_DESC1) -> bool {
    desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
}

fn supports_d3d12(adapter: &IDXGIAdapter1) -> bool {
    // SAFETY: a null output asks D3D12 to check the requested minimum feature
    // level without creating a device. The input adapter remains live.
    unsafe {
        D3D12CreateDevice::<_, ID3D12Device>(adapter, D3D_FEATURE_LEVEL_11_0, std::ptr::null_mut())
    }
    .is_ok()
}

pub(super) fn select_adapter(
    instance: &Dx12Instance,
    scope: InstanceId,
    preference: api::AdapterPreference,
) -> Result<AdapterInfo> {
    use api::AdapterPreference;

    if preference == AdapterPreference::Software {
        let adapter = adapter_at(instance, WARP_ORDINAL)?;
        if !supports_d3d12(&adapter) {
            return Err(Error::Unsupported("D3D12 on the WARP adapter"));
        }
        return Ok(AdapterInfo {
            id: AdapterId {
                instance: scope,
                ordinal: WARP_ORDINAL,
            },
            name: adapter_name(&adapter)?,
        });
    }

    if preference == AdapterPreference::Default {
        return enumerate(instance, scope)?
            .into_iter()
            .next()
            .ok_or(Error::Unsupported("a Direct3D 12 adapter"));
    }

    let preference = match preference {
        AdapterPreference::LowPower => DXGI_GPU_PREFERENCE_MINIMUM_POWER,
        AdapterPreference::HighPerformance => DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
        _ => unreachable!("default and software selection are handled above"),
    };
    let factory = instance
        .factory
        .cast::<IDXGIFactory6>()
        .map_err(|_| Error::Unsupported("DXGI GPU preference selection"))?;
    for rank in 0..=u32::MAX {
        // SAFETY: this is a live factory and DXGI returns an owned adapter.
        let adapter: IDXGIAdapter1 =
            match unsafe { factory.EnumAdapterByGpuPreference(rank, preference) } {
                Ok(adapter) => adapter,
                Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(error) => return Err(native_error(error)),
            };
        let desc = adapter_desc(&adapter)?;
        if is_software(&desc) || !supports_d3d12(&adapter) {
            continue;
        }
        // Preference order is different from enumeration order. Match native
        // identities before returning an ID that Instance::open can resolve.
        for ordinal in 0..=u32::MAX {
            match adapter_at(instance, u64::from(ordinal)) {
                Ok(candidate) if adapter_desc(&candidate)?.AdapterLuid == desc.AdapterLuid => {
                    return Ok(AdapterInfo {
                        id: AdapterId {
                            instance: scope,
                            ordinal: u64::from(ordinal),
                        },
                        name: desc_name(&desc),
                    });
                }
                Ok(_) => {}
                Err(Error::Unsupported(REQUESTED_ADAPTER_UNAVAILABLE)) => break,
                Err(error) => return Err(error),
            }
        }
    }
    Err(Error::Unsupported(
        "a Direct3D 12 adapter for the requested preference",
    ))
}

pub(super) fn enumerate(instance: &Dx12Instance, scope: InstanceId) -> Result<Vec<AdapterInfo>> {
    let mut adapters = Vec::new();
    for ordinal in 0..=u32::MAX {
        match adapter_at(instance, u64::from(ordinal)) {
            Ok(adapter) => adapters.push(AdapterInfo {
                id: AdapterId {
                    instance: scope,
                    ordinal: u64::from(ordinal),
                },
                name: adapter_name(&adapter)?,
            }),
            Err(Error::Unsupported(REQUESTED_ADAPTER_UNAVAILABLE)) => break,
            Err(error) => return Err(error),
        }
    }
    Ok(adapters)
}
pub(super) fn properties(instance: &Dx12Instance, id: AdapterId) -> Result<api::AdapterProperties> {
    let adapter = adapter_at(instance, id.ordinal)?;
    let desc = adapter_desc(&adapter)?;
    // SAFETY: the adapter and IID remain live. The optional DXGI query
    // returns a version by value.
    let driver_version = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID) }
        .ok()
        .map(|version| {
            let version = version as u64;
            [48, 32, 16, 0].map(|shift| (version >> shift) as u16)
        });
    Ok(api::AdapterProperties {
        backend: "dx12",
        vendor_id: Some(desc.VendorId),
        device_id: Some(desc.DeviceId),
        driver_version,
        is_software: Some(is_software(&desc)),
        dedicated_video_memory: Some(desc.DedicatedVideoMemory as u64),
        shared_system_memory: Some(desc.SharedSystemMemory as u64),
    })
}
