//! Standalone clears through one render pass per selected subresource.
use super::{super::MetalTexture, MetalCommandEncoder};
use fluxel_rhi_core::api::{self, Error, Result};
use std::ops::Range;

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
    if encoder.lane != 0 || !desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT) {
        return Err(Error::Unsupported(
            "Metal clear requires a graphics lane and render attachment",
        ));
    }
    match (desc.format, value) {
        (
            api::TextureFormat::Rgba8Unorm
            | api::TextureFormat::Rgba8UnormSrgb
            | api::TextureFormat::Bgra8Unorm
            | api::TextureFormat::Bgra8UnormSrgb
            | api::TextureFormat::Rgba16Float,
            api::TextureClearValue::Color(api::ClearColor::Float(_)),
        ) => {}
        (
            api::TextureFormat::Depth32Float | api::TextureFormat::Depth32FloatStencil8,
            api::TextureClearValue::Depth(v),
        ) if v.is_finite() && (0.0..=1.0).contains(&v) => {}
        _ => return Err(Error::Unsupported("Metal clear format or value")),
    }
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
            for layer in layers.clone() {
                let pass = MTLRenderPassDescriptor::renderPassDescriptor();
                match value {
                    api::TextureClearValue::Color(api::ClearColor::Float(c)) => {
                        // SAFETY: attachment zero exists in Metal's fixed attachment array.
                        let attachment =
                            unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
                        attachment.setTexture(Some(&texture.raw));
                        attachment.setLevel(mip as usize);
                        attachment.setSlice(layer as usize);
                        attachment.setLoadAction(MTLLoadAction::Clear);
                        attachment.setStoreAction(MTLStoreAction::Store);
                        attachment.setClearColor(MTLClearColor {
                            red: c[0] as f64,
                            green: c[1] as f64,
                            blue: c[2] as f64,
                            alpha: c[3] as f64,
                        });
                    }
                    api::TextureClearValue::Depth(depth) => {
                        let attachment = pass.depthAttachment();
                        attachment.setTexture(Some(&texture.raw));
                        attachment.setLevel(mip as usize);
                        attachment.setSlice(layer as usize);
                        attachment.setLoadAction(MTLLoadAction::Clear);
                        attachment.setStoreAction(MTLStoreAction::Store);
                        attachment.setClearDepth(depth as f64);
                        if desc.format == api::TextureFormat::Depth32FloatStencil8 {
                            let stencil = pass.stencilAttachment();
                            stencil.setTexture(Some(&texture.raw));
                            stencil.setLevel(mip as usize);
                            stencil.setSlice(layer as usize);
                            stencil.setLoadAction(MTLLoadAction::Load);
                            stencil.setStoreAction(MTLStoreAction::Store);
                        }
                    }
                    _ => unreachable!("validated clear value"),
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
        let _ = (encoder, texture, mips, layers, value);
        Err(Error::Unsupported("Metal requires an Apple target"))
    }
}
