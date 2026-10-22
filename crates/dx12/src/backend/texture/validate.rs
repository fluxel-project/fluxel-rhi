//! Structural DX12 texture descriptor validation.
use super::*;

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
        || d.extent[0] == 0
        || d.extent[1] == 0
        || d.extent[0] > 16_384
        || d.extent[1] > 16_384
        || if d3 {
            d.extent[0] > 2048
                || d.extent[1] > 2048
                || d.extent[2] == 0
                || d.extent[2] > 2048
                || d.array_layers != 1
                || depth
                || compressed
                || d.samples != 1
        } else if d1 {
            d.extent[1] != 1
                || d.extent[2] != 1
                || compressed
                || d.samples != 1
                || (d.array_layers > 1
                    && d.usage.0 & !(api::TextureUsage::COPY_SRC.0 | api::TextureUsage::COPY_DST.0)
                        != 0)
        } else {
            d.extent[2] != 1
        }
        || d.array_layers == 0
        || d.array_layers > 2_048
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
    if d.samples > 1 && (d.mip_levels != 1 || d.usage.contains(api::TextureUsage::STORAGE_BINDING))
    {
        return Err("DX12 multisample texture usage");
    }
    if depth && d.usage.contains(api::TextureUsage::STORAGE_BINDING) {
        return Err("DX12 depth texture usage");
    }
    if d.mip_levels == 0 || d.mip_levels > max_mips(d.extent[0], d.extent[1].max(d.extent[2])) {
        return Err("DX12 mip count");
    }
    Ok(())
}
pub(super) fn max_mips(width: u32, height: u32) -> u32 {
    1 + width.max(height).ilog2()
}
