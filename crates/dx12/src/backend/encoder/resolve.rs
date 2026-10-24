//! Native standalone multisample color resolves.
//!
//! Resource states remain entirely caller owned: source is `RESOLVE_SOURCE`
//! and destination `RESOLVE_DEST` on entry and on return.  This module only
//! validates the D3D12 resolve shape and records `ResolveSubresourceRegion`.

use super::{Dx12Encoder, copy};
use crate::backend::{Dx12, Dx12Texture};
use fluxel_rhi_core::api::{self, Error, Result};
use windows::{
    Win32::{
        Foundation::RECT,
        Graphics::Direct3D12::{
            D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE, D3D12_RESOLVE_MODE_AVERAGE,
            ID3D12GraphicsCommandList1,
        },
    },
    core::Interface,
};

pub(crate) fn texture(
    encoder: &mut Dx12Encoder,
    src: api::TextureRegion<'_, Dx12>,
    dst: api::TextureRegion<'_, Dx12>,
) -> Result<()> {
    if encoder.is_compute {
        return Err(Error::Unsupported("DX12 resolve requires graphics queue"));
    }
    // SAFETY: regions are tied to this typed device.  Cloning retains native
    // resources until the submitted command list's fence retires.
    let source = unsafe { src.texture.native_ref() }.clone();
    let target = unsafe { dst.texture.native_ref() }.clone();
    let source_desc = &source.0.desc;
    let target_desc = &target.0.desc;
    if !(source_desc
        .usage
        .contains(api::TextureUsage::RENDER_ATTACHMENT)
        || source_desc.usage.contains(api::TextureUsage::COPY_SRC))
        || !(target_desc
            .usage
            .contains(api::TextureUsage::RENDER_ATTACHMENT)
            || target_desc.usage.contains(api::TextureUsage::COPY_DST))
        || source_desc.samples <= 1
        || target_desc.samples != 1
        || source_desc.dimension != api::TextureDimension::D2
        || target_desc.dimension != api::TextureDimension::D2
        || source_desc.format != target_desc.format
        || src.aspect != api::TextureAspects::COLOR
        || dst.aspect != api::TextureAspects::COLOR
        || src.extent != dst.extent
    {
        return Err(Error::Unsupported(
            "DX12 resolve requires matching 2D multisample color source and single-sample destination usages",
        ));
    }
    if super::super::texture::is_depth(source_desc.format)
        || super::super::texture::bc_block_bytes(source_desc.format).is_some()
        || !super::super::texture::format_has_support(
            &encoder.device,
            source_desc.format,
            D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE,
        )
    {
        return Err(Error::Unsupported("DX12 color resolve format"));
    }
    let source_subresource = copy::validate_region(&Dx12Texture(source.0.clone()), &src)?;
    let target_subresource = copy::validate_region(&Dx12Texture(target.0.clone()), &dst)?;
    if source.0.resource.as_raw() == target.0.resource.as_raw() {
        return Err(Error::Unsupported("DX12 same-resource resolve"));
    }
    let layers = src.extent[2];
    let full = |desc: &api::TextureDesc, region: &api::TextureRegion<'_, Dx12>| {
        region.origin[0] == 0
            && region.origin[1] == 0
            && region.extent[0] == (desc.extent[0] >> region.mip).max(1)
            && region.extent[1] == (desc.extent[1] >> region.mip).max(1)
    };
    let full = full(source_desc, &src) && full(target_desc, &dst);
    let list1 = if full {
        None
    } else {
        Some(
            encoder
                .list
                .cast::<ID3D12GraphicsCommandList1>()
                .map_err(|_| Error::Unsupported("DX12 partial resolve command list"))?,
        )
    };
    for layer in 0..layers {
        let source_subresource = source_subresource + layer * source_desc.mip_levels;
        let target_subresource = target_subresource + layer * target_desc.mip_levels;
        let rect = RECT {
            left: i32::try_from(src.origin[0]).map_err(|_| Error::InvalidRange)?,
            top: i32::try_from(src.origin[1]).map_err(|_| Error::InvalidRange)?,
            right: i32::try_from(src.origin[0] + src.extent[0]).map_err(|_| Error::InvalidRange)?,
            bottom: i32::try_from(src.origin[1] + src.extent[1])
                .map_err(|_| Error::InvalidRange)?,
        };
        // SAFETY: region/subresource validation above establishes valid bounds.
        // The caller supplies native resolve states; both COM resources are
        // retained by the encoder after this call.
        unsafe {
            if let Some(list) = &list1 {
                list.ResolveSubresourceRegion(
                    &target.0.resource,
                    target_subresource,
                    dst.origin[0],
                    dst.origin[1],
                    &source.0.resource,
                    source_subresource,
                    Some(&rect),
                    super::super::texture::dxgi_format(source_desc.format),
                    D3D12_RESOLVE_MODE_AVERAGE,
                );
            } else {
                encoder.list.ResolveSubresource(
                    &target.0.resource,
                    target_subresource,
                    &source.0.resource,
                    source_subresource,
                    super::super::texture::dxgi_format(source_desc.format),
                );
            }
        }
    }
    encoder.textures.push(source);
    encoder.textures.push(target);
    Ok(())
}
