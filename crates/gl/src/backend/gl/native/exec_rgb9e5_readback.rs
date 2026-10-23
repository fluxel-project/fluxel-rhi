//! Bit-preserving packed and signed-normalized readback through integer images.
use super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    GlBufferRange, GlError, GlExtent3d, GlFamilyApi as _, GlFamilyProfile, GlFormat,
    GlFormatResourceKind, GlPixelFormat, GlPixelLayout, GlResourceApi as _, GlTextureDesc,
    GlTextureDimension, GlTextureRegion, GlTextureSubresource, GlTextureUsage,
};

/// Compatible integer formats in the same GL texture view-size class.
pub(super) fn exact_integer_alias(format: GlFormat) -> Option<GlFormat> {
    Some(match format {
        GlFormat::R8Snorm => GlFormat::R8Uint,
        GlFormat::Rg8Snorm | GlFormat::R16Snorm => GlFormat::R16Uint,
        GlFormat::Rgba8Snorm | GlFormat::Rg16Snorm | GlFormat::Rgb9e5Ufloat => GlFormat::R32Uint,
        GlFormat::Rgba16Snorm => GlFormat::Rg32Uint,
        _ => return None,
    })
}

impl NativeGlProvider {
    /// Seeds test-only SNORM storage by copying raw bits from a matching integer image.
    #[cfg(test)]
    pub(crate) fn seed_exact_alias_for_test(
        &mut self,
        source: crate::backend::gl::api::TextureId,
        destination: crate::backend::gl::api::TextureId,
    ) -> Result<(), GlError> {
        let (src, src_desc) = self.texture("seed alias", source)?;
        let (dst, dst_desc) = self.texture("seed alias", destination)?;
        if exact_integer_alias(dst_desc.format) != Some(src_desc.format) {
            return Err(Self::validation("seed alias", "incompatible formats"));
        }
        let region = |texture| GlTextureRegion {
            subresource: GlTextureSubresource {
                texture,
                aspect: crate::backend::gl::api::GlTextureAspect::All,
                mip_level: 1,
                base_layer: 1,
                layer_count: 1,
            },
            origin: [1, 1, 0],
            extent: GlExtent3d {
                width: 2,
                height: 2,
                depth_or_layers: 1,
            },
        };
        self.copy_texture_direct(
            "seed alias",
            region(source),
            region(destination),
            src_desc,
            dst_desc,
            src,
            dst,
        )
    }

    pub(super) fn copy_exact_alias_texture_to_buffer(
        &mut self,
        texture: glow::NativeTexture,
        desc: GlTextureDesc,
        source: GlTextureRegion,
        destination: GlBufferRange,
        layout: GlPixelLayout,
    ) -> Result<(), GlError> {
        const OP: &str = "exact-alias-texture-to-buffer";
        let alias = exact_integer_alias(desc.format)
            .ok_or_else(|| Self::validation(OP, "missing integer alias"))?;
        let alias_layout = super::exec_copy::upload_pixel_format(alias)
            .ok_or_else(|| Self::validation(OP, "missing alias layout"))?;
        if !matches!(
            self.discovery.context().profile(),
            GlFamilyProfile::Desktop { .. }
        ) || !self
            .discovery
            .formats()
            .get_for(GlFormatResourceKind::Texture, desc.format, 1)
            .is_some_and(|f| f.copy_source)
            || desc.sample_count != 1
            || super::exec_copy::upload_pixel_format(desc.format) != Some(layout.format)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "bit-preserving alias readback unavailable",
            });
        }
        let end = layout
            .required_bytes(source)
            .map_err(|_| Self::validation(OP, "invalid layout"))?;
        if end > destination.size
            || destination
                .offset
                .checked_add(end)
                .is_none_or(|v| v > u32::MAX as u64)
        {
            return Err(Self::validation(OP, "destination range overflow"));
        }
        let stride = u64::from(layout.rows_per_image) * u64::from(layout.bytes_per_row);
        let scratch_desc = GlTextureDesc {
            dimension: GlTextureDimension::D2,
            extent: GlExtent3d {
                width: source.extent.width,
                height: source.extent.height,
                depth_or_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            format: alias,
            usage: GlTextureUsage::COPY_SOURCE | GlTextureUsage::COPY_DESTINATION,
        };
        let id = self.create_texture_resource(scratch_desc)?;
        let result = (|| {
            let (scratch, _) = self.texture(OP, id)?;
            let scratch_region = GlTextureRegion {
                subresource: GlTextureSubresource {
                    texture: id,
                    aspect: source.subresource.aspect,
                    mip_level: 0,
                    base_layer: 0,
                    layer_count: 1,
                },
                origin: [0, 0, 0],
                extent: scratch_desc.extent,
            };
            for layer in 0..source.extent.depth_or_layers {
                let mut src = source;
                src.extent.depth_or_layers = 1;
                src.origin[2] = source.origin[2]
                    .checked_add(layer)
                    .ok_or_else(|| Self::validation(OP, "layer overflow"))?;
                let offset = layout
                    .offset
                    .checked_add(stride * u64::from(layer))
                    .ok_or_else(|| Self::validation(OP, "offset overflow"))?;
                // Both formats share a texture view-size class. Direct image
                // copy reinterprets storage without value conversion.
                self.copy_texture_direct(
                    OP,
                    src,
                    scratch_region,
                    desc,
                    scratch_desc,
                    texture,
                    scratch,
                )?;
                self.copy_texture_to_buffer(
                    scratch_region,
                    destination,
                    GlPixelLayout {
                        offset,
                        format: alias_layout,
                        ..layout
                    },
                )?;
            }
            Ok(())
        })();
        let cleanup = self.destroy_texture_resource(id);
        result.and(cleanup)
    }
}
