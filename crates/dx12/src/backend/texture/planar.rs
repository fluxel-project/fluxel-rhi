//! Admission and plane metadata for native NV12 and P010 textures.

use super::*;
use api::{TextureAspects as A, TextureFormat as F, TextureUsage as U};

pub(in crate::backend) fn is_planar(format: F) -> bool {
    matches!(format, F::Nv12 | F::P010)
}

pub(in crate::backend) fn supports_layout(layout: api::ImageLayout) -> bool {
    use api::ImageLayout::*;
    matches!(
        layout,
        Undefined | Common | ShaderReadOnly | CopySrc | CopyDst
    )
}

pub(super) fn view_formats(format: F) -> impl Iterator<Item = F> {
    [A::PLANE_0, A::PLANE_1]
        .into_iter()
        .filter_map(move |aspect| format.planar_view_format(aspect))
}

pub(in crate::backend) fn device_supports(device: &Dx12Device, format: F) -> bool {
    match format {
        F::Nv12 => device.0.planar_nv12,
        F::P010 => device.0.planar_p010,
        _ => false,
    }
}

pub(super) fn validate_texture(d: &api::TextureDesc) -> std::result::Result<(), &'static str> {
    let [width, height, depth] = d.extent;
    let max_axis = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
    let usage = U::TEXTURE_BINDING | U::COPY_SRC | U::COPY_DST;
    if d.dimension != api::TextureDimension::D2
        || d.samples != 1
        || d.mip_levels != 1
        || depth != 1
        || d.array_layers == 0
        || d.array_layers > max_array_layers(d)
        || width == 0
        || width > max_axis
        || width % 2 != 0
        || height == 0
        || height > max_axis
        || height % 2 != 0
        || d.usage.0 == 0
        || !usage.contains(d.usage)
        || !d.view_formats.is_empty()
    {
        return Err("DX12 planar texture requires even 2D dimensions and sampled or copy usage");
    }
    Ok(())
}

pub(super) fn validate_view(
    device: &Dx12Device,
    texture: &api::TextureDesc,
    view: &api::TextureViewDesc,
    usage: U,
) -> Result<()> {
    if !device_supports(device, texture.format)
        || !usage.contains(U::TEXTURE_BINDING)
        || usage.0 & (U::STORAGE_BINDING | U::RENDER_ATTACHMENT).0 != 0
        || texture.format.planar_view_format(view.aspects) != Some(view.format)
        || !match view.dimension {
            api::TextureViewDimension::D2 => texture.array_layers == 1,
            api::TextureViewDimension::D2Array => true,
            _ => false,
        }
    {
        return Err(Error::Unsupported("DX12 planar sampled view"));
    }
    if view.mips != (0..1)
        || view.layers.start >= view.layers.end
        || view.layers.end > texture.array_layers
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}
