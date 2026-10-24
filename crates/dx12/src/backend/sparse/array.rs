//! Preflight array mip restrictions before native reserved-resource creation.
use super::*;

pub(super) fn supports_packed_array_tails(device: &Dx12Device) -> api::Result<bool> {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: options has exactly the structure required by this synchronous query.
    unsafe {
        device.0.native.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS).cast(),
            std::mem::size_of_val(&options) as u32,
        )
    }
    .map_err(native_error)?;
    Ok(options.TiledResourcesTier.0 >= D3D12_TILED_RESOURCES_TIER_4.0)
}

pub(super) fn validate_array_mips(
    device: &Dx12Device,
    desc: &api::TextureDesc,
    native: D3D12_RESOURCE_DESC,
    depth: bool,
) -> api::Result<()> {
    // Tiers below 4 reject a texture array if any mip is smaller than a
    // standard tile in either dimension. Query the actual device tile shape
    // using a valid single-slice, single-mip reserved resource before asking
    // the driver to create the array. Tier 4 is handled by the returned
    // tiling metadata below instead of assuming the older restriction.
    if desc.array_layers > 1 {
        if !supports_packed_array_tails(device)? {
            let probe_desc = D3D12_RESOURCE_DESC {
                DepthOrArraySize: 1,
                MipLevels: 1,
                ..native
            };
            let mut probe = None;
            // SAFETY: the probe matches the validated descriptor except that
            // it has a single slice and mip, and result storage is valid.
            unsafe {
                device.0.native.CreateReservedResource(
                    &probe_desc,
                    if depth {
                        D3D12_RESOURCE_STATE_DEPTH_WRITE
                    } else {
                        D3D12_RESOURCE_STATE_COMMON
                    },
                    None,
                    &mut probe,
                )
            }
            .map_err(native_error)?;
            let probe: ID3D12Resource = probe.ok_or(api::Error::OutOfMemory)?;
            let mut shape = D3D12_TILE_SHAPE::default();
            // SAFETY: only the tile-shape output is requested; the resource
            // remains live for this synchronous native query.
            unsafe {
                device.0.native.GetResourceTiling(
                    &probe,
                    None,
                    None,
                    Some(&mut shape),
                    None,
                    0,
                    std::ptr::null_mut(),
                );
            }
            let last = desc.mip_levels - 1;
            if shape.WidthInTexels == 0 || shape.HeightInTexels == 0 {
                return Err(api::Error::Backend(
                    "DX12 sparse tile shape unavailable".into(),
                ));
            }
            if (desc.extent[0] >> last).max(1) < shape.WidthInTexels
                || (desc.extent[1] >> last).max(1) < shape.HeightInTexels
            {
                return Err(api::Error::Unsupported(
                    "DX12 sparse array mip smaller than tile",
                ));
            }
        }
    }
    Ok(())
}
