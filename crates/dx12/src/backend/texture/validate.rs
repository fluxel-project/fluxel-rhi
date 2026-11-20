//! Structural DX12 texture descriptor validation.
use super::*;

pub(super) fn max_extent(dimension: api::TextureDimension) -> [u32; 3] {
    match dimension {
        api::TextureDimension::D1 => [D3D12_REQ_TEXTURE1D_U_DIMENSION, 1, 1],
        api::TextureDimension::D2 => [
            D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
            D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
            1,
        ],
        api::TextureDimension::D3 => [D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION; 3],
    }
}

pub(super) fn max_array_layers(d: &api::TextureDesc) -> u32 {
    match d.dimension {
        api::TextureDimension::D1
            if d.usage.0 & !(api::TextureUsage::COPY_SRC.0 | api::TextureUsage::COPY_DST.0)
                == 0 =>
        {
            D3D12_REQ_TEXTURE1D_ARRAY_AXIS_DIMENSION
        }
        api::TextureDimension::D2 => D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION,
        _ => 1,
    }
}

pub(super) fn valid(d: &api::TextureDesc) -> std::result::Result<(), &'static str> {
    let depth = is_depth(d.format);
    let compressed = bc_block_bytes(d.format).is_some();
    let d3 = d.dimension == api::TextureDimension::D3;
    let d1 = d.dimension == api::TextureDimension::D1;
    if d.view_formats
        .iter()
        .any(|&view| !super::format::compatible_color_view(d.format, view))
    {
        return Err("DX12 incompatible texture view format");
    }
    if !(compressed
        || color_texel_bytes(d.format).is_some()
        || matches!(
            d.format,
            api::TextureFormat::Rgba8Unorm
                | api::TextureFormat::Bgra8Unorm
                | api::TextureFormat::Depth32Float
                | api::TextureFormat::Depth24Plus
                | api::TextureFormat::Depth16Unorm
                | api::TextureFormat::Depth24PlusStencil8
                | api::TextureFormat::Depth32FloatStencil8
        ))
        || d.extent
            .iter()
            .zip(max_extent(d.dimension))
            .any(|(&extent, maximum)| extent == 0 || extent > maximum)
        || (d3 && depth)
        || ((d1 || d3) && (compressed || d.samples != 1))
        || d.array_layers == 0
        || d.array_layers > max_array_layers(d)
        || !matches!(d.samples, 1 | 2 | 4 | 8 | 16 | 32)
    {
        return Err("DX12 RGBA8/Depth32Float D2 texture subset");
    }
    if compressed
        && (d.samples != 1
            || d.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
            || d.usage.contains(api::TextureUsage::STORAGE_BINDING))
    {
        return Err("DX12 BC texture subset");
    }
    if d.samples > 1 && d.usage.contains(api::TextureUsage::STORAGE_BINDING) {
        return Err("DX12 multisample texture usage");
    }
    if depth && d.usage.contains(api::TextureUsage::STORAGE_BINDING) {
        return Err("DX12 depth texture usage");
    }
    if d.mip_levels == 0 || d.mip_levels > max_mips(d) {
        return Err("DX12 mip count");
    }
    Ok(())
}
pub(super) fn max_mips(desc: &api::TextureDesc) -> u32 {
    if desc.samples > 1 {
        1
    } else {
        let [width, height, depth] = desc.extent;
        1 + width.max(height).max(depth).ilog2()
    }
}

pub(super) fn view(
    dv: &Dx12Device,
    desc: &api::TextureDesc,
    v: &api::TextureViewDesc,
    usage: api::TextureUsage,
) -> Result<()> {
    if desc.dimension == api::TextureDimension::D1 && desc.array_layers > 1 {
        return Err(Error::Unsupported("DX12 D1 array texture views"));
    }
    let depth = is_depth(desc.format);
    let array = desc.array_layers > 1 || v.dimension == api::TextureViewDimension::D2Array;
    let cube = matches!(
        v.dimension,
        api::TextureViewDimension::Cube | api::TextureViewDimension::CubeArray
    );
    if !match desc.dimension {
        api::TextureDimension::D1 => v.dimension == api::TextureViewDimension::D1,
        api::TextureDimension::D2 => matches!(
            v.dimension,
            api::TextureViewDimension::D2
                | api::TextureViewDimension::D2Array
                | api::TextureViewDimension::Cube
                | api::TextureViewDimension::CubeArray
        ),
        api::TextureDimension::D3 => v.dimension == api::TextureViewDimension::D3,
    } || (v.dimension == api::TextureViewDimension::D2
        && v.layers.start.checked_add(1) != Some(v.layers.end))
        || (cube
            && (desc.extent[0] != desc.extent[1]
                || desc.samples != 1
                || v.layers.start % 6 != 0
                || (v.layers.end.saturating_sub(v.layers.start)) % 6 != 0
                || (v.dimension == api::TextureViewDimension::Cube
                    && (v.layers.start != 0 || v.layers.end != 6))
                || usage.contains(api::TextureUsage::STORAGE_BINDING)
                || usage.contains(api::TextureUsage::RENDER_ATTACHMENT)))
        || v.mips.start >= v.mips.end
        || v.mips.end > desc.mip_levels
        || v.layers.start >= v.layers.end
        || v.layers.end > desc.array_layers
        || (usage.contains(api::TextureUsage::STORAGE_BINDING)
            && v.mips.start.checked_add(1) != Some(v.mips.end))
        || if depth {
            let all = depth_aspects(desc.format);
            v.aspects != all
                && !(all != api::TextureAspects::DEPTH
                    && (v.aspects == api::TextureAspects::DEPTH
                        || v.aspects == api::TextureAspects::STENCIL))
        } else {
            v.aspects != api::TextureAspects::COLOR
        }
    {
        return Err(Error::Unsupported("DX12 D2/D2Array texture view"));
    }
    // A D2 descriptor cannot select a slice of a native array resource.
    // Storage arrays need a Texture2DArray UAV and matching shader binding.
    if array
        && v.dimension == api::TextureViewDimension::D2
        && (usage.contains(api::TextureUsage::TEXTURE_BINDING)
            || usage.contains(api::TextureUsage::STORAGE_BINDING))
    {
        return Err(Error::Unsupported("DX12 array texture binding view"));
    }
    if usage.contains(api::TextureUsage::RENDER_ATTACHMENT) {
        if depth && v.aspects == depth_aspects(desc.format) {
            depth_dsv_desc(v, desc.samples, array)?;
        } else if !depth && v.mips.start.checked_add(1) != Some(v.mips.end) {
            return Err(Error::Unsupported("DX12 RTV view requires one mip"));
        }
    }
    if !view_format_available(dv, desc, v.format, usage) {
        return Err(Error::Unsupported("DX12 texture view format"));
    }
    if cube && usage.contains(api::TextureUsage::TEXTURE_BINDING) {
        if v.aspects == (api::TextureAspects::DEPTH | api::TextureAspects::STENCIL) {
            return Err(Error::Unsupported(
                "DX12 sampled cube view needs one aspect",
            ));
        }
        let format = if depth {
            depth_view::srv_format(desc.format, v.aspects)
        } else {
            dxgi_format(v.format)
        };
        if !format::native_format_has_support(dv, format, D3D12_FORMAT_SUPPORT1_TEXTURECUBE) {
            return Err(Error::Unsupported("DX12 sampled cube view format"));
        }
    }
    Ok(())
}
