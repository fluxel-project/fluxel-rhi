//! Standalone attachment clears for selected texture subresources.
use super::{super::MetalTexture, MetalCommandEncoder};
use fluxel_rhi_core::api::{self, Error, Result};
use std::ops::Range;

fn color_value(format: api::TextureFormat, color: api::ClearColor) -> Result<[f64; 4]> {
    use api::{ClearColor, TextureFormat::*};
    let unsigned = matches!(
        format,
        R8Uint
            | Rg8Uint
            | Rgba8Uint
            | R16Uint
            | Rg16Uint
            | Rgba16Uint
            | R32Uint
            | Rg32Uint
            | Rgba32Uint
            | Rgb10a2Uint
    );
    let signed = matches!(
        format,
        R8Sint
            | Rg8Sint
            | Rgba8Sint
            | R16Sint
            | Rg16Sint
            | Rgba16Sint
            | R32Sint
            | Rg32Sint
            | Rgba32Sint
    );
    match color {
        ClearColor::Uint(c) if unsigned => Ok(c.map(f64::from)),
        ClearColor::Sint(c) if signed => Ok(c.map(f64::from)),
        ClearColor::Float(c) if !unsigned && !signed => Ok(c.map(f64::from)),
        _ => Err(Error::InvalidState),
    }
}

pub(super) fn clear_texture(
    encoder: &mut MetalCommandEncoder,
    texture: &MetalTexture,
    mips: Range<u32>,
    layers: Range<u32>,
    value: api::TextureClearValue,
) -> Result<()> {
    let desc = texture.desc();
    if mips.start >= mips.end
        || mips.end > desc.mip_levels
        || layers.start >= layers.end
        || layers.end > desc.array_layers
    {
        return Err(Error::InvalidRange);
    }
    if !desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT) {
        return Err(Error::InvalidState);
    }
    let combined = matches!(
        desc.format,
        api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8
    );
    let color = match value {
        api::TextureClearValue::Color(value)
            if super::super::texture::color_attachment_format(desc.format) =>
        {
            Some(color_value(desc.format, value)?)
        }
        api::TextureClearValue::Depth(value)
            if matches!(
                desc.format,
                api::TextureFormat::Depth16Unorm
                    | api::TextureFormat::Depth24Plus
                    | api::TextureFormat::Depth24PlusStencil8
                    | api::TextureFormat::Depth32Float
                    | api::TextureFormat::Depth32FloatStencil8
            ) && value.is_finite()
                && (0.0..=1.0).contains(&value) =>
        {
            None
        }
        _ => return Err(Error::InvalidState),
    };
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::{
            MTLClearColor, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLLoadAction,
            MTLRenderPassDescriptor, MTLStoreAction,
        };
        if encoder.render.is_some() || encoder.compute.is_some() {
            return Err(Error::InvalidState);
        }
        for mip in mips {
            let slices = if desc.dimension == api::TextureDimension::D3 {
                0..(desc.extent[2] >> mip).max(1)
            } else {
                layers.clone()
            };
            for slice in slices {
                let pass = MTLRenderPassDescriptor::renderPassDescriptor();
                if let Some(c) = color {
                    // SAFETY: attachment zero exists in Metal's fixed attachment array.
                    let a = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
                    a.setTexture(Some(&texture.raw));
                    a.setLevel(mip as usize);
                    if desc.dimension == api::TextureDimension::D3 {
                        a.setDepthPlane(slice as usize);
                    } else {
                        a.setSlice(slice as usize);
                    }
                    a.setLoadAction(MTLLoadAction::Clear);
                    a.setStoreAction(MTLStoreAction::Store);
                    a.setClearColor(MTLClearColor {
                        red: c[0],
                        green: c[1],
                        blue: c[2],
                        alpha: c[3],
                    });
                } else if let api::TextureClearValue::Depth(depth) = value {
                    let a = pass.depthAttachment();
                    a.setTexture(Some(&texture.raw));
                    a.setLevel(mip as usize);
                    if desc.dimension == api::TextureDimension::D3 {
                        a.setDepthPlane(slice as usize);
                    } else {
                        a.setSlice(slice as usize);
                    }
                    a.setLoadAction(MTLLoadAction::Clear);
                    a.setStoreAction(MTLStoreAction::Store);
                    a.setClearDepth(depth as f64);
                    if combined {
                        let s = pass.stencilAttachment();
                        s.setTexture(Some(&texture.raw));
                        s.setLevel(mip as usize);
                        if desc.dimension == api::TextureDimension::D3 {
                            s.setDepthPlane(slice as usize);
                        } else {
                            s.setSlice(slice as usize);
                        }
                        s.setLoadAction(MTLLoadAction::Load);
                        s.setStoreAction(MTLStoreAction::Store);
                    }
                }
                let render = encoder
                    .command
                    .renderCommandEncoderWithDescriptor(&pass)
                    .ok_or(Error::OutOfMemory)?;
                render.endEncoding();
            }
        }
        Ok(())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (encoder, texture, mips, layers, value, color, combined);
        Err(Error::Unsupported("Metal requires an Apple target"))
    }
}
