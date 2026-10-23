//! Immediate standalone texture clears through temporary framebuffers.
use super::super::owner::{BrowserEncoder, BrowserTexture};
use super::attachments::{BrowserColorAttachment, BrowserDepthAttachment};
use crate::backend::gl::api::{GlFormatResourceKind, GlTextureDimension};
use fluxel_rhi_core::api::{Error, LoadOp, Rect, Result, StoreOp, TextureClearValue};

impl BrowserEncoder {
    pub(crate) fn clear_texture_subresources(
        &mut self,
        texture: &BrowserTexture,
        mips: std::ops::Range<u32>,
        layers: std::ops::Range<u32>,
        value: TextureClearValue,
    ) -> Result<()> {
        self.same_owner(&texture.owner)?;
        let desc = texture.desc;
        let kind = if texture.renderbuffer.is_some() {
            GlFormatResourceKind::Renderbuffer
        } else {
            GlFormatResourceKind::Texture
        };
        if !self
            .owner
            .discovery_snapshot()
            .formats()
            .get_for(kind, desc.format, desc.sample_count)
            .is_some_and(|facts| facts.renderable)
        {
            let TextureClearValue::Color(color) = value else {
                return Err(Error::Unsupported("WebGL2 non-renderable depth clear"));
            };
            let texel = crate::backend::gl::clear_texel::encode(desc.format, color)?;
            // Keep the staging allocation bounded while immediately uploading
            // every selected mip/layer; no compressed encoding is attempted.
            for mip in mips {
                let extent = desc.mip_extent(mip).ok_or(Error::InvalidRange)?;
                let slices = if desc.dimension == GlTextureDimension::D3 {
                    0..extent.depth_or_layers
                } else {
                    layers.clone()
                };
                let row_bytes = (extent.width as usize)
                    .checked_mul(texel.len())
                    .ok_or(Error::InvalidRange)?;
                let rows = (1024 * 1024 / row_bytes.max(1))
                    .max(1)
                    .min(extent.height as usize);
                let length = row_bytes.checked_mul(rows).ok_or(Error::InvalidRange)?;
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(length)
                    .map_err(|_| Error::OutOfMemory)?;
                while bytes.len() < length {
                    bytes.extend_from_slice(&texel);
                }
                for layer in slices {
                    let mut y = 0;
                    while y < extent.height {
                        let count = (extent.height - y).min(rows as u32);
                        self.write_texture_region(
                            texture,
                            mip,
                            [0, y, layer],
                            [extent.width, count, 1],
                            u32::try_from(row_bytes).map_err(|_| Error::InvalidRange)?,
                            count,
                            &bytes,
                        )?;
                        y += count;
                    }
                }
            }
            return Ok(());
        }
        for mip in mips {
            let extent = desc.mip_extent(mip).ok_or(Error::InvalidRange)?;
            let slices = if desc.dimension == GlTextureDimension::D3 {
                0..extent.depth_or_layers
            } else {
                layers.clone()
            };
            for layer in slices {
                let area = Rect {
                    x: 0,
                    y: 0,
                    width: extent.width,
                    height: extent.height,
                };
                match value {
                    TextureClearValue::Color(color) => {
                        let colors = [Some(BrowserColorAttachment {
                            texture: Some(texture),
                            surface_format: None,
                            mip_level: mip,
                            layer,
                            load: LoadOp::Clear(color),
                            store: StoreOp::Store,
                            resolve: None,
                        })];
                        self.begin_offscreen_attachments(
                            &colors,
                            None,
                            [extent.width, extent.height],
                            area,
                        )?;
                    }
                    TextureClearValue::Depth(depth) => {
                        let attachment = BrowserDepthAttachment {
                            texture,
                            aspect: crate::backend::gl::api::GlTextureAspect::All,
                            mip_level: mip,
                            layer,
                            depth: Some((LoadOp::Clear(depth), StoreOp::Store)),
                            stencil: None,
                        };
                        self.begin_offscreen_attachments(
                            &[],
                            Some(attachment),
                            [extent.width, extent.height],
                            area,
                        )?;
                    }
                }
                self.end_offscreen_raster()?;
            }
        }
        Ok(())
    }
}
