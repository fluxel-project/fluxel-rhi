//! Standalone whole-subresource color and depth clears.
//! The caller supplies the attachment state through explicit transitions.
use super::*;
use std::ops::Range;

pub(crate) fn texture(
    encoder: &mut Dx12Encoder,
    texture: &Dx12Texture,
    mips: Range<u32>,
    layers: Range<u32>,
    value: api::TextureClearValue,
) -> Result<(), Error> {
    let desc = &texture.0.desc;
    if mips.start >= mips.end
        || mips.end > desc.mip_levels
        || layers.start >= layers.end
        || layers.end > desc.array_layers
    {
        return Err(Error::InvalidRange);
    }
    if !desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT) {
        return Err(Error::Unsupported(
            "DX12 clear requires render attachment usage",
        ));
    }
    let (aspect, depth) = match (desc.format, value) {
        (api::TextureFormat::Depth32Float, api::TextureClearValue::Depth(x))
            if x.is_finite() && (0.0..=1.0).contains(&x) =>
        {
            (api::TextureAspects::DEPTH, true)
        }
        (
            api::TextureFormat::Rgba8Unorm | api::TextureFormat::Bgra8Unorm,
            api::TextureClearValue::Color(api::ClearColor::Float(_)),
        ) => (api::TextureAspects::COLOR, false),
        _ => return Err(Error::Unsupported("DX12 clear format/value")),
    };
    let dimension = if desc.array_layers > 1 {
        api::TextureViewDimension::D2Array
    } else {
        api::TextureViewDimension::D2
    };
    // Allocate every view before recording so descriptor allocation failure
    // cannot leave a partially encoded multi-mip clear.
    let mut views = Vec::new();
    for mip in mips {
        views.push(super::super::texture::create_view(
            &encoder.device,
            texture,
            &api::TextureViewDesc {
                format: desc.format,
                dimension,
                aspects: aspect,
                mips: mip..mip + 1,
                layers: layers.clone(),
            },
        )?);
    }
    for view in views {
        if depth {
            let handle = view
                .0
                .dsv
                .ok_or(Error::Unsupported("DX12 clear depth view"))?;
            let heap = view.0.dsv_heap.as_ref().ok_or(Error::InvalidState)?.clone();
            let api::TextureClearValue::Depth(clear) = value else {
                unreachable!()
            };
            // SAFETY: descriptor and resource are retained below and the caller
            // supplies DEPTH_WRITE state for all selected subresources.
            unsafe {
                encoder
                    .list
                    .ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH, clear, 0, None);
            }
            encoder.clear_heaps.push(heap);
        } else {
            let handle = view
                .0
                .rtv
                .ok_or(Error::Unsupported("DX12 clear color view"))?;
            let heap = view.0.rtv_heap.as_ref().ok_or(Error::InvalidState)?.clone();
            let api::TextureClearValue::Color(api::ClearColor::Float(clear)) = value else {
                unreachable!()
            };
            // SAFETY: descriptor and resource are retained below and the caller
            // supplies RENDER_TARGET state for all selected subresources.
            unsafe {
                encoder.list.ClearRenderTargetView(handle, &clear, None);
            }
            encoder.clear_heaps.push(heap);
        }
    }
    encoder.textures.push(texture.clone());
    Ok(())
}
