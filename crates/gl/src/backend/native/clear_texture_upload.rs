//! Bounded constant uploads for non-renderable textures on older contexts.
use super::provider::NativeGlProvider;
use crate::backend::api::{
    GlCopyDomainApi as _, GlError, GlExtent3d, GlTextureAspect, GlTextureDesc, GlTextureDimension,
    GlTextureRegion, GlTextureSubresource, TextureId,
};
use fluxel_rhi_core::api::{ClearColor, TexelCopyBufferLayout};

impl NativeGlProvider {
    pub(super) fn clear_color_upload(
        &mut self,
        id: TextureId,
        desc: GlTextureDesc,
        mips: std::ops::Range<u32>,
        layers: std::ops::Range<u32>,
        color: ClearColor,
    ) -> Result<(), GlError> {
        let texel =
            crate::backend::clear_texel::encode(desc.format, color).map_err(
                |error| match error {
                    fluxel_rhi_core::api::Error::OutOfMemory => GlError::OutOfMemory {
                        operation: "clear-texture",
                    },
                    _ => GlError::Unsupported {
                        operation: "clear-texture",
                        reason: "color clear has no native upload encoding",
                    },
                },
            )?;
        for mip in mips {
            let extent = desc
                .mip_extent(mip)
                .ok_or_else(|| Self::validation("clear-texture", "invalid mip"))?;
            let slices = if desc.dimension == GlTextureDimension::D3 {
                0..extent.depth_or_layers
            } else {
                layers.clone()
            };
            let row_bytes = (extent.width as usize)
                .checked_mul(texel.len())
                .ok_or_else(|| Self::validation("clear-texture", "row size overflow"))?;
            let rows = (1024 * 1024 / row_bytes.max(1))
                .max(1)
                .min(extent.height as usize);
            let length = row_bytes
                .checked_mul(rows)
                .ok_or_else(|| Self::validation("clear-texture", "staging size overflow"))?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(length)
                .map_err(|_| GlError::OutOfMemory {
                    operation: "clear-texture",
                })?;
            while bytes.len() < length {
                bytes.extend_from_slice(&texel);
            }
            for layer in slices {
                let mut y = 0;
                while y < extent.height {
                    let count = (extent.height - y).min(rows as u32);
                    let layout = crate::backend::translate::lower_texel_copy_layout(
                        TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: row_bytes as u32,
                            rows_per_image: count,
                        },
                        desc.format,
                    )
                    .map_err(|_| Self::validation("clear-texture", "invalid upload layout"))?;
                    let volume = desc.dimension == GlTextureDimension::D3;
                    self.upload_texture(
                        GlTextureRegion {
                            subresource: GlTextureSubresource {
                                texture: id,
                                aspect: GlTextureAspect::Color,
                                mip_level: mip,
                                base_layer: if volume { 0 } else { layer },
                                layer_count: 1,
                            },
                            origin: [0, y, if volume { layer } else { 0 }],
                            extent: GlExtent3d {
                                width: extent.width,
                                height: count,
                                depth_or_layers: 1,
                            },
                        },
                        layout,
                        &bytes,
                    )?;
                    y += count;
                }
            }
        }
        Ok(())
    }
}
