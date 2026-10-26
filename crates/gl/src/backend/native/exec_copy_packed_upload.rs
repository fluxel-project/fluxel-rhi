//! Packed depth/stencil plane uploads through a same-format scratch texture.
use super::super::provider::NativeGlProvider;
use super::{UnpackDirection, packed_unpack_rows};
use crate::backend::api::{
    GlBufferRange, GlError, GlFamilyProfile, GlFormat, GlFormatResourceKind, GlPixelFormat,
    GlPixelLayout, GlTextureAspect, GlTextureDesc, GlTextureDimension, GlTextureRegion,
    GlTextureUsage,
};
use glow::HasContext as _;

pub(in crate::backend::native) enum PackedSource<'a> {
    Cpu(&'a [u8]),
    Buffer(glow::NativeBuffer, GlBufferRange),
}

impl NativeGlProvider {
    pub(in crate::backend::native) fn upload_packed_plane(
        &mut self,
        op: &'static str,
        destination: glow::NativeTexture,
        desc: GlTextureDesc,
        region: GlTextureRegion,
        layout: GlPixelLayout,
        source: PackedSource<'_>,
    ) -> Result<(), GlError> {
        let embedded = matches!(
            self.discovery.context().profile(),
            GlFamilyProfile::Embedded { .. }
        );
        if !desc.usage.contains(GlTextureUsage::COPY_DESTINATION) || desc.sample_count != 1 {
            return Err(Self::validation(
                op,
                "packed texture missing copy usage or is multisampled",
            ));
        }
        if !self
            .discovery
            .formats()
            .get_for(GlFormatResourceKind::Texture, desc.format, 1)
            .is_some_and(|f| f.copy_destination && f.renderable)
        {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "packed texture lacks copy and framebuffer facts",
            });
        }
        let standalone = matches!(desc.format, GlFormat::Depth16Unorm | GlFormat::Depth32Float);
        let stencil = region.subresource.aspect == GlTextureAspect::StencilOnly;
        if !(standalone
            && region.subresource.aspect == GlTextureAspect::DepthOnly
            && matches!(
                (desc.format, layout.format),
                (GlFormat::Depth16Unorm, GlPixelFormat::Depth16)
                    | (GlFormat::Depth32Float, GlPixelFormat::Depth32Float)
            )
            || stencil && layout.format == GlPixelFormat::Stencil8
            || desc.format == GlFormat::Depth32FloatStencil8
                && region.subresource.aspect == GlTextureAspect::DepthOnly
                && layout.format == GlPixelFormat::Depth32Float)
        {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "packed plane upload representation",
            });
        }
        if !matches!(
            desc.dimension,
            GlTextureDimension::D1
                | GlTextureDimension::D2
                | GlTextureDimension::D2Array
                | GlTextureDimension::D3
        ) {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "packed plane upload dimension",
            });
        }
        let end = layout
            .required_bytes(region)
            .map_err(|_| Self::validation(op, "invalid packed plane layout"))?;
        match &source {
            PackedSource::Cpu(bytes) if end > bytes.len() as u64 => {
                return Err(Self::validation(op, "packed source too short"));
            }
            PackedSource::Buffer(_, range) if end > range.size => {
                return Err(Self::validation(op, "packed source buffer too short"));
            }
            _ => {}
        }
        if let PackedSource::Buffer(_, range) = &source {
            // Validate the last byte before issuing any scratch upload or blit.
            // This also bounds every row start and the signed glow copy offsets.
            if !pbo_range_fits(range.offset, end) {
                return Err(Self::validation(op, "PBO source range exceeds GLint"));
            }
        }
        let w = i32::try_from(region.extent.width)
            .map_err(|_| Self::validation(op, "width exceeds GLint"))?;
        let h = i32::try_from(region.extent.height)
            .map_err(|_| Self::validation(op, "height exceeds GLint"))?;
        let pixel_count = usize::try_from(region.extent.width)
            .ok()
            .and_then(|w| w.checked_mul(region.extent.height as usize))
            .ok_or_else(|| Self::validation(op, "scratch size overflow"))?;
        let cpu = match &source {
            PackedSource::Cpu(bytes) => Some(packed_unpack_rows(op, layout, region, bytes)?),
            _ => None,
        };
        let plane_bytes = if stencil {
            1
        } else if desc.format == GlFormat::Depth16Unorm {
            2
        } else {
            4
        };
        let row_bytes = usize::try_from(region.extent.width)
            .ok()
            .and_then(|w| w.checked_mul(plane_bytes))
            .ok_or_else(|| Self::validation(op, "row size overflow"))?;
        let pack_plane = stencil || (embedded && !standalone);
        let packed_bytes = if pack_plane && desc.format == GlFormat::Depth32FloatStencil8 {
            8
        } else {
            4
        };
        let depth_size = pixel_count
            .checked_mul(packed_bytes)
            .and_then(|v| i32::try_from(v).ok())
            .ok_or_else(|| Self::validation(op, "scratch PBO size overflow"))?;
        if pack_plane && matches!(source, PackedSource::Buffer(..)) {
            self.stencil_pack_preflight(op, pixel_count, depth_size as usize)?;
        }
        let input_size = pixel_count
            .checked_mul(plane_bytes)
            .and_then(|n| n.checked_add(3))
            .map(|n| n & !3)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| Self::validation(op, "input scratch size overflow"))?;
        let mut scratch_desc = desc;
        scratch_desc.dimension = GlTextureDimension::D2;
        scratch_desc.extent = region.extent;
        scratch_desc.extent.depth_or_layers = 1;
        scratch_desc.mip_level_count = 1;
        let mut scratch_region = region;
        scratch_region.origin = [0, 0, 0];
        scratch_region.extent.depth_or_layers = 1;
        scratch_region.subresource.mip_level = 0;
        scratch_region.subresource.base_layer = 0;
        scratch_region.subresource.layer_count = 1;
        // SAFETY: context is current, scratch objects are private, every row
        // and offset has been checked, and bindings/pixel state are restored.
        unsafe {
            let tex = self
                .gl
                .create_texture()
                .map_err(|_| GlError::OutOfMemory { operation: op })?;
            let saved = self.pixel_store;
            self.gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            self.gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
            self.apply_tight_unpack();
            let (internal, kind) = match desc.format {
                GlFormat::Depth16Unorm => (glow::DEPTH_COMPONENT16, glow::UNSIGNED_SHORT),
                GlFormat::Depth32Float => (glow::DEPTH_COMPONENT32F, glow::FLOAT),
                GlFormat::Depth24PlusStencil8 => (glow::DEPTH24_STENCIL8, glow::UNSIGNED_INT_24_8),
                _ => (
                    glow::DEPTH32F_STENCIL8,
                    glow::FLOAT_32_UNSIGNED_INT_24_8_REV,
                ),
            };
            self.gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                internal as i32,
                w,
                h,
                0,
                if standalone {
                    glow::DEPTH_COMPONENT
                } else {
                    glow::DEPTH_STENCIL
                },
                kind,
                glow::PixelUnpackData::Slice(None),
            );
            let pbo = if matches!(source, PackedSource::Buffer(..)) {
                match self.gl.create_buffer() {
                    Ok(b) => {
                        self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(b));
                        self.gl.buffer_data_size(
                            glow::COPY_WRITE_BUFFER,
                            depth_size,
                            glow::STREAM_DRAW,
                        );
                        Some(b)
                    }
                    Err(_) => None,
                }
            } else {
                None
            };
            let input = if pack_plane && matches!(source, PackedSource::Buffer(..)) {
                match self.gl.create_buffer() {
                    Ok(b) => {
                        self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(b));
                        self.gl.buffer_data_size(
                            glow::COPY_WRITE_BUFFER,
                            input_size,
                            glow::STREAM_DRAW,
                        );
                        Some(b)
                    }
                    Err(_) => None,
                }
            } else {
                None
            };
            let mut result = Ok(());
            if pack_plane && matches!(source, PackedSource::Buffer(..)) && input.is_none() {
                result = Err(GlError::OutOfMemory { operation: op });
            }
            if matches!(source, PackedSource::Buffer(..)) && pbo.is_none() {
                result = Err(GlError::OutOfMemory { operation: op });
            }
            for layer in 0..region.extent.depth_or_layers {
                if result.is_err() {
                    break;
                }
                let mut packed = Vec::new();
                let (format, upload_type) = if pack_plane {
                    if let Some(bytes) = cpu.as_ref() {
                        let start = layer as usize * pixel_count * plane_bytes;
                        let element = if desc.format == GlFormat::Depth24PlusStencil8 {
                            4
                        } else {
                            8
                        };
                        if packed.try_reserve_exact(pixel_count * element).is_err() {
                            result = Err(GlError::OutOfMemory { operation: op });
                            break;
                        }
                        packed.resize(pixel_count * element, 0);
                        if stencil {
                            for (i, &v) in bytes[start..start + pixel_count].iter().enumerate() {
                                let offset = i * element + if element == 4 { 0 } else { 4 };
                                packed[offset..offset + 4]
                                    .copy_from_slice(&u32::from(v).to_ne_bytes());
                            }
                        } else {
                            for i in 0..pixel_count {
                                packed[i * 8..i * 8 + 4]
                                    .copy_from_slice(&bytes[start + i * 4..start + i * 4 + 4]);
                            }
                        }
                    }
                    (glow::DEPTH_STENCIL, kind)
                } else {
                    (
                        glow::DEPTH_COMPONENT,
                        if standalone { kind } else { glow::FLOAT },
                    )
                };
                self.gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                match &source {
                    PackedSource::Cpu(_) => {
                        self.gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
                        let payload = if pack_plane {
                            packed.as_slice()
                        } else {
                            let bytes = cpu.as_ref().unwrap();
                            let start = layer as usize * pixel_count * plane_bytes;
                            &bytes[start..start + pixel_count * plane_bytes]
                        };
                        self.gl.tex_sub_image_2d(
                            glow::TEXTURE_2D,
                            0,
                            0,
                            0,
                            w,
                            h,
                            format,
                            upload_type,
                            glow::PixelUnpackData::Slice(Some(payload)),
                        );
                    }
                    PackedSource::Buffer(src, range) => {
                        let scratch = pbo.expect("scratch checked");
                        self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(*src));
                        let copy_target = input.unwrap_or(scratch);
                        self.gl
                            .bind_buffer(glow::COPY_WRITE_BUFFER, Some(copy_target));
                        for row in 0..region.extent.height {
                            let offset = range.offset
                                + layout.offset
                                + u64::from(layout.bytes_per_row)
                                    * (u64::from(layout.rows_per_image) * u64::from(layer)
                                        + u64::from(row));
                            self.copy_buffer_sub_data_wide(
                                glow::COPY_READ_BUFFER,
                                glow::COPY_WRITE_BUFFER,
                                offset,
                                u64::from(row)
                                    * u64::try_from(row_bytes).expect("row size fits u64"),
                                u64::try_from(row_bytes).expect("row size fits u64"),
                                op,
                            )?;
                        }
                        if result.is_err() {
                            break;
                        }
                        if pack_plane {
                            if let Err(e) = self.pack_stencil_gpu(
                                op,
                                input.expect("stencil input checked"),
                                scratch,
                                pixel_count,
                                desc.format == GlFormat::Depth32FloatStencil8,
                                !stencil,
                            ) {
                                result = Err(e);
                                break;
                            }
                        }
                        self.gl
                            .bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(scratch));
                        self.gl.tex_sub_image_2d(
                            glow::TEXTURE_2D,
                            0,
                            0,
                            0,
                            w,
                            h,
                            format,
                            upload_type,
                            glow::PixelUnpackData::BufferOffset(0),
                        );
                    }
                }
                let mut dest = region;
                dest.extent.depth_or_layers = 1;
                if desc.dimension == GlTextureDimension::D3 {
                    dest.origin[2] += layer;
                } else {
                    dest.subresource.base_layer += layer;
                }
                dest.subresource.layer_count = 1;
                result = self.copy_packed_plane(
                    op,
                    scratch_region,
                    dest,
                    scratch_desc,
                    desc,
                    tex,
                    destination,
                );
            }
            self.gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
            self.restore_pixel_store(UnpackDirection::Unpack, saved);
            self.gl.bind_texture(glow::TEXTURE_2D, None);
            if let Some(input) = input {
                self.gl.delete_buffer(input);
            }
            if let Some(pbo) = pbo {
                self.gl.delete_buffer(pbo);
            }
            self.gl.delete_texture(tex);
            result.and_then(|()| self.defer_driver_error(op))
        }
    }
}

/// The native buffer-copy API uses signed 32-bit offsets. The end is
/// exclusive; validating it once bounds every row in the entire upload.
fn pbo_range_fits(offset: u64, end: u64) -> bool {
    offset
        .checked_add(end)
        .is_some_and(|last| last <= i32::MAX as u64)
}

#[cfg(test)]
#[path = "../../tests/backend/native/exec_copy_packed_upload__tests.rs"]
mod tests;
