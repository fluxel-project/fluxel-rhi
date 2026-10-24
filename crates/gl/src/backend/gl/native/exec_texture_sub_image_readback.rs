//! Direct desktop texture subimage readback into GPU pixel-pack buffers.
use super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    GlBufferRange, GlError, GlExtensionSet, GlFamilyApi as _, GlFamilyProfile, GlFormat,
    GlPixelLayout, GlTextureDesc, GlTextureDimension, GlTextureRegion,
};
use glow::HasContext as _;

pub(super) type GetTexImageFn =
    unsafe extern "system" fn(u32, i32, u32, u32, *mut core::ffi::c_void);

pub(super) type GetTextureSubImageFn = unsafe extern "system" fn(
    u32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    u32,
    u32,
    i32,
    *mut core::ffi::c_void,
);

pub(super) fn direct_read_supported(profile: GlFamilyProfile, extensions: &GlExtensionSet) -> bool {
    matches!(profile, GlFamilyProfile::Desktop { major, minor } if major > 4 || (major == 4 && minor >= 5))
        || (matches!(profile, GlFamilyProfile::Desktop { .. })
            && extensions
                .raw_reported_names()
                .any(|name| name == "GL_ARB_get_texture_sub_image"))
}

pub(super) fn direct_color_format(format: GlFormat) -> bool {
    !matches!(
        format,
        GlFormat::Rgb9e5Ufloat
            | GlFormat::R8Snorm
            | GlFormat::Rg8Snorm
            | GlFormat::Rgba8Snorm
            | GlFormat::R16Snorm
            | GlFormat::Rg16Snorm
            | GlFormat::Rgba16Snorm
            | GlFormat::Depth16Unorm
            | GlFormat::Depth32Float
            | GlFormat::Stencil8
            | GlFormat::Depth24PlusStencil8
            | GlFormat::Depth32FloatStencil8
    ) && super::exec_copy::upload_encoding(format).is_some()
        && super::exec_copy::upload_pixel_format(format).is_some()
}

#[cfg(test)]
std::thread_local! {
    static DIRECT_READ_COUNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static LEGACY_READ_COUNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static FORCE_LEGACY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) fn allow_direct_read() -> bool {
    #[cfg(test)]
    {
        return !FORCE_LEGACY.with(|flag| flag.get());
    }
    #[cfg(not(test))]
    {
        true
    }
}

impl NativeGlProvider {
    #[cfg(test)]
    pub(crate) fn force_legacy_color_readback_test(&self, enabled: bool) {
        FORCE_LEGACY.with(|flag| flag.set(enabled));
    }
    #[cfg(test)]
    pub(crate) fn legacy_color_readback_test_state(&self) -> (bool, u32) {
        (
            self.get_tex_image.is_some(),
            LEGACY_READ_COUNT.with(|count| count.get()),
        )
    }
    #[cfg(test)]
    pub(crate) fn direct_color_readback_test_state(&self) -> (bool, u32) {
        (
            self.get_texture_sub_image.is_some()
                && direct_read_supported(
                    self.discovery.context().profile(),
                    self.discovery.extensions(),
                ),
            DIRECT_READ_COUNT.with(|count| count.get()),
        )
    }

    pub(super) fn copy_texture_sub_image_to_buffer(
        &mut self,
        texture: glow::NativeTexture,
        desc: GlTextureDesc,
        source: GlTextureRegion,
        buffer: glow::NativeBuffer,
        destination: GlBufferRange,
        layout: GlPixelLayout,
    ) -> Result<(), GlError> {
        const OP: &str = "texture-sub-image-to-buffer";
        let get = self.get_texture_sub_image.ok_or(GlError::Unsupported {
            operation: OP,
            reason: "glGetTextureSubImage unavailable",
        })?;
        let (format, kind) = super::exec_copy::upload_encoding(desc.format)
            .ok_or_else(|| Self::validation(OP, "unsupported color format"))?;
        if desc.sample_count != 1
            || super::exec_copy::upload_pixel_format(desc.format) != Some(layout.format)
            || !matches!(
                desc.dimension,
                GlTextureDimension::D1
                    | GlTextureDimension::D2
                    | GlTextureDimension::D2Array
                    | GlTextureDimension::D3
            )
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "texture readback format, sample count or dimension",
            });
        }
        if desc.dimension == GlTextureDimension::D1
            && (source.subresource.base_layer != 0
                || source.origin[1] != 0
                || source.origin[2] != 0
                || source.extent.height != 1
                || source.extent.depth_or_layers != 1)
        {
            return Err(Self::validation(OP, "invalid 1D readback"));
        }
        let end = layout
            .required_bytes(source)
            .map_err(|_| Self::validation(OP, "invalid pixel layout"))?;
        if end > destination.size {
            return Err(Self::validation(OP, "destination too short"));
        }
        let _total_end = destination
            .offset
            .checked_add(end)
            .and_then(|v| isize::try_from(v).ok())
            .ok_or_else(|| Self::validation(OP, "destination exceeds native GL range"))?;
        let level = i32::try_from(source.subresource.mip_level)
            .map_err(|_| Self::validation(OP, "mip exceeds GL range"))?;
        let x = i32::try_from(source.origin[0])
            .map_err(|_| Self::validation(OP, "x exceeds GL range"))?;
        let y_end = source.origin[1]
            .checked_add(source.extent.height)
            .and_then(|x| i32::try_from(x).ok())
            .ok_or_else(|| Self::validation(OP, "y exceeds GL range"))?;
        let _ = y_end;
        let z = source
            .subresource
            .base_layer
            .checked_add(source.origin[2])
            .ok_or_else(|| Self::validation(OP, "layer overflow"))?;
        z.checked_add(source.extent.depth_or_layers)
            .and_then(|x| i32::try_from(x).ok())
            .ok_or_else(|| Self::validation(OP, "layer exceeds GL range"))?;
        let width = i32::try_from(source.extent.width)
            .map_err(|_| Self::validation(OP, "width exceeds GL range"))?;
        let bpp = u64::from(layout.format.bytes_per_pixel());
        let row_bytes = u64::from(source.extent.width)
            .checked_mul(bpp)
            .and_then(|x| i32::try_from(x).ok())
            .ok_or_else(|| Self::validation(OP, "row exceeds GL range"))?;
        let stride = u64::from(layout.bytes_per_row) * u64::from(layout.rows_per_image);
        let scratch_needed = (destination.offset + layout.offset) % bpp != 0
            || u64::from(layout.bytes_per_row) % bpp != 0;
        #[cfg(test)]
        DIRECT_READ_COUNT.with(|count| count.set(count.get() + 1));
        // SAFETY: all objects belong to the current context, destination ranges and GL integer bounds were validated. The scratch PBO handles unaligned byte offsets without CPU readback.
        unsafe {
            let pack = self.gl.get_parameter_i32(glow::PIXEL_PACK_BUFFER_BINDING) as u32;
            let read = self.gl.get_parameter_i32(glow::COPY_READ_BUFFER_BINDING) as u32;
            let write = self.gl.get_parameter_i32(glow::COPY_WRITE_BUFFER_BINDING) as u32;
            let scratch = if scratch_needed {
                let scratch = self
                    .gl
                    .create_buffer()
                    .map_err(|_| GlError::OutOfMemory { operation: OP })?;
                self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(scratch));
                self.gl
                    .buffer_data_size(glow::PIXEL_PACK_BUFFER, row_bytes, glow::STREAM_READ);
                Some(scratch)
            } else {
                None
            };
            self.gl
                .bind_buffer(glow::PIXEL_PACK_BUFFER, Some(scratch.unwrap_or(buffer)));
            let saved = self.pixel_store;
            for (key, value) in [
                (glow::PACK_ALIGNMENT, 1),
                (glow::PACK_ROW_LENGTH, 0),
                (glow::PACK_IMAGE_HEIGHT, 0),
                (glow::PACK_SKIP_PIXELS, 0),
                (glow::PACK_SKIP_ROWS, 0),
                (glow::PACK_SKIP_IMAGES, 0),
            ] {
                self.gl.pixel_store_i32(key, value);
            }
            for layer in 0..source.extent.depth_or_layers {
                for row in 0..source.extent.height {
                    let offset = destination.offset
                        + layout.offset
                        + stride * u64::from(layer)
                        + u64::from(layout.bytes_per_row) * u64::from(row);
                    get(
                        texture.0.get(),
                        level,
                        x,
                        (source.origin[1] + row) as i32,
                        (z + layer) as i32,
                        width,
                        1,
                        1,
                        format,
                        kind,
                        row_bytes,
                        if scratch.is_some() {
                            core::ptr::null_mut()
                        } else {
                            offset as usize as *mut core::ffi::c_void
                        },
                    );
                    if let Some(scratch) = scratch {
                        self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(scratch));
                        self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
                        self.copy_buffer_sub_data_wide(
                            glow::COPY_READ_BUFFER,
                            glow::COPY_WRITE_BUFFER,
                            0,
                            offset,
                            u64::try_from(row_bytes).expect("validated row size"),
                            OP,
                        )?;
                    }
                }
            }
            let result = self.driver_error(OP);
            for (key, value) in [
                (glow::PACK_ALIGNMENT, saved.pack_alignment as i32),
                (glow::PACK_ROW_LENGTH, saved.pack_row_length as i32),
                (glow::PACK_IMAGE_HEIGHT, saved.pack_image_height as i32),
                (glow::PACK_SKIP_PIXELS, saved.pack_skip_pixels as i32),
                (glow::PACK_SKIP_ROWS, saved.pack_skip_rows as i32),
                (glow::PACK_SKIP_IMAGES, saved.pack_skip_images as i32),
            ] {
                self.gl.pixel_store_i32(key, value);
            }
            self.gl.bind_buffer(
                glow::PIXEL_PACK_BUFFER,
                std::num::NonZeroU32::new(pack).map(glow::NativeBuffer),
            );
            self.gl.bind_buffer(
                glow::COPY_READ_BUFFER,
                std::num::NonZeroU32::new(read).map(glow::NativeBuffer),
            );
            self.gl.bind_buffer(
                glow::COPY_WRITE_BUFFER,
                std::num::NonZeroU32::new(write).map(glow::NativeBuffer),
            );
            if let Some(scratch) = scratch {
                self.gl.delete_buffer(scratch);
            }
            result
        }
    }
}

impl NativeGlProvider {
    /// Copies a complete mip to temporary GPU memory and selects rows on GPU.
    pub(super) fn copy_texture_image_to_buffer(
        &mut self,
        texture: glow::NativeTexture,
        desc: GlTextureDesc,
        source: GlTextureRegion,
        buffer: glow::NativeBuffer,
        destination: GlBufferRange,
        layout: GlPixelLayout,
    ) -> Result<(), GlError> {
        const OP: &str = "texture-image-to-buffer";
        let get = self.get_tex_image.ok_or(GlError::Unsupported {
            operation: OP,
            reason: "glGetTexImage unavailable",
        })?;
        let (format, kind) = super::exec_copy::upload_encoding(desc.format)
            .ok_or_else(|| Self::validation(OP, "unsupported format"))?;
        if desc.sample_count != 1
            || super::exec_copy::upload_pixel_format(desc.format) != Some(layout.format)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "texture readback format or sample count",
            });
        }
        let (target, binding) = match desc.dimension {
            GlTextureDimension::D1 => (glow::TEXTURE_1D, glow::TEXTURE_BINDING_1D),
            GlTextureDimension::D2 => (glow::TEXTURE_2D, glow::TEXTURE_BINDING_2D),
            GlTextureDimension::D2Array => (glow::TEXTURE_2D_ARRAY, glow::TEXTURE_BINDING_2D_ARRAY),
            GlTextureDimension::D3 => (glow::TEXTURE_3D, glow::TEXTURE_BINDING_3D),
            _ => {
                return Err(GlError::Unsupported {
                    operation: OP,
                    reason: "texture readback dimension",
                });
            }
        };
        let mip = desc
            .mip_extent(source.subresource.mip_level)
            .ok_or_else(|| Self::validation(OP, "invalid mip"))?;
        let end = layout
            .required_bytes(source)
            .map_err(|_| Self::validation(OP, "invalid pixel layout"))?;
        if end > destination.size {
            return Err(Self::validation(OP, "destination too short"));
        }
        destination
            .offset
            .checked_add(end)
            .and_then(|v| i32::try_from(v).ok())
            .ok_or_else(|| Self::validation(OP, "destination exceeds GL range"))?;
        let bpp = u64::from(layout.format.bytes_per_pixel());
        let mip_row = u64::from(mip.width)
            .checked_mul(bpp)
            .ok_or_else(|| Self::validation(OP, "mip row overflow"))?;
        let mip_layer = mip_row
            .checked_mul(u64::from(mip.height))
            .ok_or_else(|| Self::validation(OP, "mip layer overflow"))?;
        let scratch_size = mip_layer
            .checked_mul(u64::from(mip.depth_or_layers))
            .and_then(|v| i32::try_from(v).ok())
            .ok_or_else(|| Self::validation(OP, "mip exceeds GL buffer range"))?;
        let row_bytes = u64::from(source.extent.width)
            .checked_mul(bpp)
            .and_then(|v| i32::try_from(v).ok())
            .ok_or_else(|| Self::validation(OP, "row exceeds GL buffer range"))?;
        let level = i32::try_from(source.subresource.mip_level)
            .map_err(|_| Self::validation(OP, "mip exceeds GL range"))?;
        let z = source
            .subresource
            .base_layer
            .checked_add(source.origin[2])
            .ok_or_else(|| Self::validation(OP, "layer overflow"))?;
        let output_stride = u64::from(layout.bytes_per_row) * u64::from(layout.rows_per_image);
        #[cfg(test)]
        LEGACY_READ_COUNT.with(|count| count.set(count.get() + 1));
        // SAFETY: the mip and destination ranges fit the validated GL integer bounds. The temporary PBO receives the complete mip; all selected rows are copied on the GPU.
        unsafe {
            let prior_texture = self.gl.get_parameter_i32(binding) as u32;
            let pack = self.gl.get_parameter_i32(glow::PIXEL_PACK_BUFFER_BINDING) as u32;
            let read = self.gl.get_parameter_i32(glow::COPY_READ_BUFFER_BINDING) as u32;
            let write = self.gl.get_parameter_i32(glow::COPY_WRITE_BUFFER_BINDING) as u32;
            let scratch = self
                .gl
                .create_buffer()
                .map_err(|_| GlError::OutOfMemory { operation: OP })?;
            self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(scratch));
            self.gl
                .buffer_data_size(glow::PIXEL_PACK_BUFFER, scratch_size, glow::STREAM_READ);
            self.gl.bind_texture(target, Some(texture));
            let saved = self.pixel_store;
            for (key, value) in [
                (glow::PACK_ALIGNMENT, 1),
                (glow::PACK_ROW_LENGTH, 0),
                (glow::PACK_IMAGE_HEIGHT, 0),
                (glow::PACK_SKIP_PIXELS, 0),
                (glow::PACK_SKIP_ROWS, 0),
                (glow::PACK_SKIP_IMAGES, 0),
            ] {
                self.gl.pixel_store_i32(key, value);
            }
            get(target, level, format, kind, core::ptr::null_mut());
            let mut result = self.driver_error(OP);
            if result.is_ok() {
                self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(scratch));
                self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
                for layer in 0..source.extent.depth_or_layers {
                    for row in 0..source.extent.height {
                        let src = u64::from(z + layer) * mip_layer
                            + u64::from(source.origin[1] + row) * mip_row
                            + u64::from(source.origin[0]) * bpp;
                        let dst = destination.offset
                            + layout.offset
                            + output_stride * u64::from(layer)
                            + u64::from(layout.bytes_per_row) * u64::from(row);
                        self.copy_buffer_sub_data_wide(
                            glow::COPY_READ_BUFFER,
                            glow::COPY_WRITE_BUFFER,
                            src,
                            dst,
                            u64::try_from(row_bytes).expect("validated row size"),
                            OP,
                        )?;
                    }
                }
                result = self.driver_error(OP);
            }
            for (key, value) in [
                (glow::PACK_ALIGNMENT, saved.pack_alignment as i32),
                (glow::PACK_ROW_LENGTH, saved.pack_row_length as i32),
                (glow::PACK_IMAGE_HEIGHT, saved.pack_image_height as i32),
                (glow::PACK_SKIP_PIXELS, saved.pack_skip_pixels as i32),
                (glow::PACK_SKIP_ROWS, saved.pack_skip_rows as i32),
                (glow::PACK_SKIP_IMAGES, saved.pack_skip_images as i32),
            ] {
                self.gl.pixel_store_i32(key, value);
            }
            self.gl.bind_texture(
                target,
                std::num::NonZeroU32::new(prior_texture).map(glow::NativeTexture),
            );
            self.gl.bind_buffer(
                glow::PIXEL_PACK_BUFFER,
                std::num::NonZeroU32::new(pack).map(glow::NativeBuffer),
            );
            self.gl.bind_buffer(
                glow::COPY_READ_BUFFER,
                std::num::NonZeroU32::new(read).map(glow::NativeBuffer),
            );
            self.gl.bind_buffer(
                glow::COPY_WRITE_BUFFER,
                std::num::NonZeroU32::new(write).map(glow::NativeBuffer),
            );
            self.gl.delete_buffer(scratch);
            result
        }
    }
}
