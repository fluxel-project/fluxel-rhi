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
    if encoder.is_compute {
        return Err(Error::Unsupported(
            "DX12 texture clear requires a graphics lane",
        ));
    }
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
    if let api::TextureClearValue::Color(color) = value {
        if super::super::texture::integer_color(desc.format) {
            let (signed, bits) = super::super::texture::integer_clear_bits(desc.format, color)
                .ok_or(Error::InvalidState)?;
            let pipeline = encoder.device.0.integer_clear_cache.get_or_create(
                &encoder.device,
                desc.format,
                desc.samples,
                signed,
            )?;
            let mut views = Vec::new();
            for mip in mips {
                if desc.dimension == api::TextureDimension::D3 {
                    for slice in 0..(desc.extent[2] >> mip).max(1) {
                        let (heap, handle) = super::integer_clear::volume_slice_rtv(
                            &encoder.device,
                            &texture.0,
                            mip,
                            slice,
                        )?;
                        views.push((mip, Some(heap), handle));
                    }
                    continue;
                }
                for layer in layers.clone() {
                    let view = super::super::texture::create_view(
                        &encoder.device,
                        texture,
                        &api::TextureViewDesc {
                            format: desc.format,
                            dimension: if desc.dimension == api::TextureDimension::D1 {
                                api::TextureViewDimension::D1
                            } else if desc.array_layers > 1 {
                                api::TextureViewDimension::D2Array
                            } else {
                                api::TextureViewDimension::D2
                            },
                            aspects: api::TextureAspects::COLOR,
                            mips: mip..mip + 1,
                            layers: layer..layer + 1,
                        },
                    )?;
                    encoder.view_descriptor_leases.push(
                        view.0
                            .rtv_lease
                            .as_ref()
                            .ok_or(Error::InvalidState)?
                            .clone(),
                    );
                    views.push((mip, None, view.0.rtv.ok_or(Error::InvalidState)?));
                }
            }
            for (mip, heap, handle) in views {
                let width = (desc.extent[0] >> mip).max(1);
                let height = (desc.extent[1] >> mip).max(1);
                let rect = windows::Win32::Foundation::RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                };
                super::integer_clear::record(
                    &encoder.list,
                    &pipeline,
                    handle,
                    width,
                    height,
                    rect,
                    bits,
                )?;
                if let Some(heap) = heap {
                    encoder.clear_heaps.push(heap);
                }
            }
            super::restore_after_internal_draw(encoder)?;
            encoder.integer_clear_pipelines.push(pipeline);
            encoder.textures.push(texture.clone());
            return Ok(());
        }
    }
    let (aspect, depth) = match (desc.format, value) {
        (format, api::TextureClearValue::Depth(x))
            if super::super::texture::is_depth(format)
                && x.is_finite()
                && (0.0..=1.0).contains(&x) =>
        {
            (super::super::texture::depth_aspects(format), true)
        }
        (format, api::TextureClearValue::Color(api::ClearColor::Float(_)))
            if super::super::texture::color_texel_bytes(format).is_some()
                && !super::super::texture::integer_color(format) =>
        {
            (api::TextureAspects::COLOR, false)
        }
        _ => return Err(Error::Unsupported("DX12 clear format/value")),
    };
    let dimension = if desc.dimension == api::TextureDimension::D1 {
        api::TextureViewDimension::D1
    } else if desc.dimension == api::TextureDimension::D3 {
        api::TextureViewDimension::D3
    } else if desc.array_layers > 1 {
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
            let heap = view
                .0
                .dsv_lease
                .as_ref()
                .ok_or(Error::InvalidState)?
                .clone();
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
            encoder.view_descriptor_leases.push(heap);
        } else {
            let handle = view
                .0
                .rtv
                .ok_or(Error::Unsupported("DX12 clear color view"))?;
            let heap = view
                .0
                .rtv_lease
                .as_ref()
                .ok_or(Error::InvalidState)?
                .clone();
            let api::TextureClearValue::Color(api::ClearColor::Float(clear)) = value else {
                unreachable!()
            };
            // SAFETY: descriptor and resource are retained below and the caller
            // supplies RENDER_TARGET state for all selected subresources.
            unsafe {
                encoder.list.ClearRenderTargetView(handle, &clear, None);
            }
            encoder.view_descriptor_leases.push(heap);
        }
    }
    encoder.textures.push(texture.clone());
    Ok(())
}
