//! Native copy, upload, and readback execution.
//!
//! Texture copy has two honest routes: `glCopyImageSubData` when the profile's
//! core version supplies it (GLES 3.2+ and desktop GL 4.3+ context), and
//! otherwise the framebuffer route shared with the browser backend (a scratch
//! framebuffer served by `glCopyTexSubImage2D` / `glReadPixels`). The route
//! decision is a pure function of the discovered profile fact, recorded in
//! [`texture_copy_route`]. Provider entry points keep Core's WebGPU-style
//! texture row order; logical row zero is native texel row zero.

use super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    GlBufferRange, GlCopyDomainApi, GlError, GlFamilyApi as _, GlFormatResourceKind, GlPixelFormat,
    GlPixelLayout, GlPixelStoreState, GlReadback, GlTextureDesc, GlTextureRegion, GlTextureUsage,
    validate_texture_copy,
};
use crate::backend::gl::native::buffer_abi::pointer_sized;

impl GlCopyDomainApi for NativeGlProvider {
    fn buffer_to_texture(
        &mut self,
        source: GlBufferRange,
        destination: GlTextureRegion,
        layout: GlPixelLayout,
    ) -> Result<(), GlError> {
        self.upload_from_pixel_buffer(source, destination, layout)
    }
    fn copy_buffer_range(
        &mut self,
        source: GlBufferRange,
        destination: GlBufferRange,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "copy-buffer";
        self.assert_ready(OP)?;
        let (source_name, source_desc) = self.buffer(OP, source.buffer)?;
        let (destination_name, destination_desc) = self.buffer(OP, destination.buffer)?;
        source
            .validate_for(source_desc)
            .and_then(|_| destination.validate_for(destination_desc))
            .map_err(|_| Self::validation(OP, "invalid buffer range"))?;
        if source.size != destination.size {
            return Err(Self::validation(OP, "copy sizes differ"));
        }
        let read_offset = pointer_sized(source.offset)
            .ok_or_else(|| Self::validation(OP, "source offset exceeds native GLintptr"))?;
        let write_offset = pointer_sized(destination.offset)
            .ok_or_else(|| Self::validation(OP, "destination offset exceeds native GLintptr"))?;
        let size = pointer_sized(source.size)
            .ok_or_else(|| Self::validation(OP, "copy size exceeds native GLsizeiptr"))?;
        // The copy targets are Layer 1-private scratch: bind immediately before
        // use, no restore on return (`GlCopyDomainApi` documents why).
        // SAFETY: current-context contract; both live resources and all ranges
        // were validated before bindings or the copy command are changed.
        unsafe {
            self.gl
                .bind_buffer(glow::COPY_READ_BUFFER, Some(source_name));
            self.gl
                .bind_buffer(glow::COPY_WRITE_BUFFER, Some(destination_name));
            if let Some(functions) = self.buffer_functions {
                (functions.copy_buffer_sub_data)(
                    glow::COPY_READ_BUFFER,
                    glow::COPY_WRITE_BUFFER,
                    read_offset,
                    write_offset,
                    size,
                );
            } else {
                let read_offset = i32::try_from(read_offset)
                    .map_err(|_| Self::validation(OP, "GL buffer ABI entry points unavailable"))?;
                let write_offset = i32::try_from(write_offset)
                    .map_err(|_| Self::validation(OP, "GL buffer ABI entry points unavailable"))?;
                let size = i32::try_from(size)
                    .map_err(|_| Self::validation(OP, "GL buffer ABI entry points unavailable"))?;
                self.gl.copy_buffer_sub_data(
                    glow::COPY_READ_BUFFER,
                    glow::COPY_WRITE_BUFFER,
                    read_offset,
                    write_offset,
                    size,
                );
            }
        }
        self.driver_error(OP)
    }

    fn copy_texture_region(
        &mut self,
        source: GlTextureRegion,
        destination: GlTextureRegion,
    ) -> Result<(), GlError> {
        const OP: &str = "copy-texture";
        self.assert_ready(OP)?;
        let (source_name, source_desc) = self.texture(OP, source.subresource.texture)?;
        let (destination_name, destination_desc) =
            self.texture(OP, destination.subresource.texture)?;
        validate_texture_copy(source, source_desc, destination, destination_desc)
            .map_err(|_| Self::validation(OP, "invalid texture copy"))?;
        if !source_desc
            .format
            .supports_blit_aspect(source.subresource.aspect)
            && !(source.subresource.aspect == crate::backend::gl::api::GlTextureAspect::Color
                && source_desc.format.compressed_info().is_some())
        {
            return Err(Self::validation(
                OP,
                "copy aspect does not match texture format",
            ));
        }
        // Validate each side against its own mip without reversing row order.
        let source = native_texture_region(OP, source_desc, source)?;
        let destination = native_texture_region(OP, destination_desc, destination)?;
        if source.subresource.texture == destination.subresource.texture
            && source.subresource.mip_level == destination.subresource.mip_level
        {
            // Reading and writing one mip is a driver-dependent feedback loop;
            // reject deterministically instead of trusting the driver check.
            return Err(Self::validation(
                OP,
                "copy source and destination name the same mip",
            ));
        }
        let copy_kind = |desc: GlTextureDesc| {
            if desc.sample_count > 1
                && desc.dimension == crate::backend::gl::api::GlTextureDimension::D2Array
            {
                GlFormatResourceKind::MultisampleArray
            } else {
                GlFormatResourceKind::Texture
            }
        };
        if !source_desc.usage.contains(GlTextureUsage::COPY_SOURCE)
            || !destination_desc
                .usage
                .contains(GlTextureUsage::COPY_DESTINATION)
        {
            return Err(Self::validation(
                OP,
                "copy source or destination usage is missing",
            ));
        }
        let facts = self.discovery.formats();
        let source_copy = facts
            .get_for(
                copy_kind(source_desc),
                source_desc.format,
                source_desc.sample_count,
            )
            .map(|fact| fact.copy_source)
            .unwrap_or(false);
        let destination_copy = facts
            .get_for(
                copy_kind(destination_desc),
                destination_desc.format,
                destination_desc.sample_count,
            )
            .map(|fact| fact.copy_destination)
            .unwrap_or(false);
        if !source_copy || !destination_copy {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "format lacks a proved copy fact on this context",
            });
        }
        // A selected depth/stencil plane uses an attachment blit. This also
        // supplies the GLES fallback where copy-image is not available.
        if source.subresource.aspect != crate::backend::gl::api::GlTextureAspect::Color {
            return self.copy_packed_plane(
                OP,
                source,
                destination,
                source_desc,
                destination_desc,
                source_name,
                destination_name,
            );
        }
        // Multisample color images are admitted only after discovery proved
        // the direct image-copy entry point. They must never fall through to
        // the FBO route: that route has distinct API-family semantics; WebGL
        // therefore keeps its multisample-copy facts false.
        match if source_desc.sample_count > 1 || self.copy_image_extension.is_some() {
            CopyRoute::DirectImage
        } else {
            texture_copy_route(self.discovery.context().profile())
        } {
            CopyRoute::DirectImage => self.copy_texture_direct(
                OP,
                source,
                destination,
                source_desc,
                destination_desc,
                source_name,
                destination_name,
            ),
            CopyRoute::Framebuffer => self.copy_texture_framebuffer(
                OP,
                source,
                destination,
                source_desc,
                destination_desc,
                source_name,
                destination_name,
            ),
        }
    }

    fn upload_buffer(&mut self, destination: GlBufferRange, bytes: &[u8]) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "upload-buffer";
        self.assert_ready(OP)?;
        let (name, desc) = self.buffer(OP, destination.buffer)?;
        destination
            .validate_for(desc)
            .map_err(|_| Self::validation(OP, "invalid destination buffer range"))?;
        if u64::try_from(bytes.len()).ok() != Some(destination.size) {
            return Err(Self::validation(OP, "byte length differs from the range"));
        }
        let offset = pointer_sized(destination.offset)
            .ok_or_else(|| Self::validation(OP, "offset exceeds native GLintptr"))?;
        // COPY_WRITE_BUFFER keeps ARRAY_BUFFER and ELEMENT_ARRAY_BUFFER
        // vertex-state bindings untouched by transfer work.  The copy targets
        // themselves are Layer 1-private scratch: bound immediately before use,
        // not restored on return (`GlCopyDomainApi` documents why).
        // SAFETY: current-context contract; the exact byte slice was validated.
        unsafe {
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(name));
            if let Some(functions) = self.buffer_functions {
                (functions.buffer_sub_data)(
                    glow::COPY_WRITE_BUFFER,
                    offset,
                    isize::try_from(bytes.len()).map_err(|_| {
                        Self::validation(OP, "upload size exceeds native GLsizeiptr")
                    })?,
                    bytes.as_ptr().cast(),
                );
            } else {
                let offset = i32::try_from(offset)
                    .map_err(|_| Self::validation(OP, "GL buffer ABI entry points unavailable"))?;
                self.gl
                    .buffer_sub_data_u8_slice(glow::COPY_WRITE_BUFFER, offset, bytes);
            }
        }
        self.driver_error(OP)
    }

    fn read_buffer(&mut self, source: GlBufferRange) -> Result<Vec<u8>, GlError> {
        self.read_buffer_now(source)
    }

    fn upload_texture(
        &mut self,
        destination: GlTextureRegion,
        layout: GlPixelLayout,
        bytes: &[u8],
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "upload-texture";
        self.assert_ready(OP)?;
        let (name, desc) = self.texture(OP, destination.subresource.texture)?;
        // MSAA copy facts admit only matching-sample image copies. Pixel
        // transfer APIs cannot address multisample texture storage.
        if desc.sample_count != 1 {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "multisample pixel upload is unavailable",
            });
        }
        destination
            .validate_for(desc)
            .map_err(|_| Self::validation(OP, "invalid texture region"))?;
        if matches!(
            self.discovery.context().profile(),
            crate::backend::gl::api::GlFamilyProfile::Embedded { .. }
        ) && matches!(
            desc.format,
            crate::backend::gl::api::GlFormat::Depth16Unorm
                | crate::backend::gl::api::GlFormat::Depth32Float
        ) {
            return self.upload_packed_plane(
                OP,
                name,
                desc,
                destination,
                layout,
                packed_upload::PackedSource::Cpu(bytes),
            );
        }
        if matches!(
            desc.format,
            crate::backend::gl::api::GlFormat::Depth24PlusStencil8
                | crate::backend::gl::api::GlFormat::Depth32FloatStencil8
        ) {
            return self.upload_packed_plane(
                OP,
                name,
                desc,
                destination,
                layout,
                packed_upload::PackedSource::Cpu(bytes),
            );
        }
        if desc.format == crate::backend::gl::api::GlFormat::Stencil8
            && !self
                .discovery
                .formats()
                .get_for(GlFormatResourceKind::Texture, desc.format, 1)
                .is_some_and(|facts| facts.copy_destination)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "stencil texture lacks copy destination evidence",
            });
        }
        if desc.format.compressed_info().is_some() {
            return self.upload_compressed_blocks(
                name,
                desc,
                destination,
                layout,
                super::exec_compressed_upload::CompressedSource::Cpu(bytes),
            );
        }
        let Some((gl_format, gl_type)) = upload_encoding(desc.format) else {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "format accepts no CPU pixel upload in this shared semantic",
            });
        };
        if texture::upload_pixel_format(desc.format) != Some(layout.format) {
            return Err(Self::validation(
                OP,
                "pixel layout does not match texture format",
            ));
        }
        transfer_format(layout.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "pixel encoding has no native transfer route",
        })?;
        ensure_upload_2d_region(OP, desc, destination)?;
        if desc.dimension == crate::backend::gl::api::GlTextureDimension::D1
            && self.texture_sub_image_1d.is_none()
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "glTexSubImage1D unavailable",
            });
        }
        let needed = layout
            .required_bytes(destination)
            .map_err(|_| Self::validation(OP, "invalid pixel layout"))?;
        if u64::try_from(bytes.len())
            .ok()
            .is_none_or(|length| length < needed)
        {
            return Err(Self::validation(OP, "upload source is shorter than layout"));
        }
        let native_destination = native_texture_region(OP, desc, destination)?;
        let (level, x, y, width, height) = rect_coords(OP, native_destination)?;
        // Texture texel y=0 stores Core row zero; pack only layout padding.
        let payload = packed_unpack_rows(OP, layout, destination, bytes)?;

        let saved = self.pixel_store;
        // SAFETY: current-context contract; the layout was validated and the
        // exact tracked pixel-store state is restored on every return path.
        let prior_unpack_buffer =
            unsafe { self.gl.get_parameter_i32(glow::PIXEL_UNPACK_BUFFER_BINDING) as u32 };
        let result = unsafe {
            (|| {
                self.gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
                self.apply_tight_unpack();
                match desc.dimension {
                    crate::backend::gl::api::GlTextureDimension::D1 => {
                        let entry = self
                            .texture_sub_image_1d
                            .expect("1D upload entry point was checked before mutation");
                        self.gl.bind_texture(glow::TEXTURE_1D, Some(name));
                        entry(
                            glow::TEXTURE_1D,
                            level,
                            x,
                            width,
                            gl_format,
                            gl_type,
                            payload.as_ptr().cast(),
                        );
                    }
                    crate::backend::gl::api::GlTextureDimension::D2 => {
                        self.gl.bind_texture(glow::TEXTURE_2D, Some(name));
                        self.gl.tex_sub_image_2d(
                            glow::TEXTURE_2D,
                            level,
                            x,
                            y,
                            width,
                            height,
                            gl_format,
                            gl_type,
                            glow::PixelUnpackData::Slice(Some(payload.as_ref())),
                        );
                    }
                    crate::backend::gl::api::GlTextureDimension::D2Array
                    | crate::backend::gl::api::GlTextureDimension::D3 => {
                        let z = destination
                            .subresource
                            .base_layer
                            .checked_add(destination.origin[2])
                            .and_then(|value| i32::try_from(value).ok())
                            .ok_or_else(|| Self::validation(OP, "array layer exceeds GLint"))?;
                        let depth =
                            i32::try_from(destination.extent.depth_or_layers).map_err(|_| {
                                Self::validation(OP, "array layer count exceeds GLsizei")
                            })?;
                        let target =
                            if desc.dimension == crate::backend::gl::api::GlTextureDimension::D3 {
                                glow::TEXTURE_3D
                            } else {
                                glow::TEXTURE_2D_ARRAY
                            };
                        self.gl.bind_texture(target, Some(name));
                        self.gl.tex_sub_image_3d(
                            target,
                            level,
                            x,
                            y,
                            z,
                            width,
                            height,
                            depth,
                            gl_format,
                            gl_type,
                            glow::PixelUnpackData::Slice(Some(payload.as_ref())),
                        );
                    }
                    _ => unreachable!("upload preflight admitted an unsupported texture dimension"),
                }
                self.driver_error(OP)
            })()
        };
        // SAFETY: current-context contract; exact snapshot restoration.
        unsafe {
            self.gl.bind_buffer(
                glow::PIXEL_UNPACK_BUFFER,
                std::num::NonZeroU32::new(prior_unpack_buffer).map(glow::NativeBuffer),
            );
            self.restore_pixel_store(UnpackDirection::Unpack, saved);
        }
        result
    }

    fn read_texture(
        &mut self,
        source: GlTextureRegion,
        layout: GlPixelLayout,
    ) -> Result<GlReadback, GlError> {
        self.read_texture_now(source, layout)
    }

    fn pixel_store(&self) -> GlPixelStoreState {
        self.pixel_store
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum UnpackDirection {
    Unpack,
    Pack,
}

#[path = "exec_copy_texture.rs"]
mod texture;
use texture::{CopyRoute, texture_copy_route};
pub(super) use texture::{upload_encoding, upload_pixel_format};

#[path = "exec_readback.rs"]
mod readback;

/// The readback encoding this backend returns for a discovered texture format.
///
/// `RGBA`/`UNSIGNED_BYTE` is the only combination every accepted profile must
/// support through the framebuffer route.
fn readback_encoding(format: crate::backend::gl::api::GlFormat) -> Option<(u32, u32)> {
    match format {
        crate::backend::gl::api::GlFormat::Rgba8Unorm
        | crate::backend::gl::api::GlFormat::Rgba8Srgb => Some((glow::RGBA, glow::UNSIGNED_BYTE)),
        _ => None,
    }
}

/// The client pixel format a transfer must use, if this backend accepts it.
fn transfer_format(format: GlPixelFormat) -> Option<()> {
    match format {
        GlPixelFormat::Rgba8
        | GlPixelFormat::Red8
        | GlPixelFormat::Rg8
        | GlPixelFormat::Stencil8
        | GlPixelFormat::Depth16
        | GlPixelFormat::Depth32Float
        | GlPixelFormat::TexelBytes(2 | 4 | 8 | 16) => Some(()),
        // Native GLES has no BGRA upload and the discovered RGBA8 textures
        // accept no RGB/RED encoding; every one of those rejects before
        // touching pixel state.
        _ => None,
    }
}

/// Rejects regions that are not one axis-aligned rectangle of one slice.
fn ensure_single_2d_rect(operation: &'static str, region: GlTextureRegion) -> Result<(), GlError> {
    if region.subresource.base_layer != 0
        || region.subresource.layer_count != 1
        || region.origin[2] != 0
        || region.extent.depth_or_layers != 1
    {
        return Err(GlError::Unsupported {
            operation,
            reason: "this copy slice transfers one 2D rectangle only",
        });
    }
    Ok(())
}

/// Admits CPU uploads into a 2D image, contiguous array layers, or volume slices.
///
/// The public region has a layer-relative Z coordinate, while GL's 2D-array
/// upload receives the absolute layer selected by `base_layer + origin.z`.
/// Region validation already checked that this span stays inside the selected
/// subresource; this gate only refuses dimensions with different native routes.
fn ensure_upload_2d_region(
    operation: &'static str,
    desc: crate::backend::gl::api::GlTextureDesc,
    region: GlTextureRegion,
) -> Result<(), GlError> {
    match desc.dimension {
        crate::backend::gl::api::GlTextureDimension::D1
            if region.subresource.base_layer == 0
                && region.subresource.layer_count == 1
                && region.origin[1] == 0
                && region.origin[2] == 0
                && region.extent.height == 1
                && region.extent.depth_or_layers == 1 =>
        {
            Ok(())
        }
        crate::backend::gl::api::GlTextureDimension::D2
            if region.subresource.base_layer == 0
                && region.subresource.layer_count == 1
                && region.origin[2] == 0
                && region.extent.depth_or_layers == 1 =>
        {
            Ok(())
        }
        crate::backend::gl::api::GlTextureDimension::D2Array
        | crate::backend::gl::api::GlTextureDimension::D3 => Ok(()),
        _ => Err(GlError::Unsupported {
            operation,
            reason: "this upload route supports 2D, 2D-array and 3D textures only",
        }),
    }
}

fn validation(operation: &'static str, message: &'static str) -> GlError {
    GlError::Validation {
        operation,
        message: message.into(),
    }
}

/// Texture storage uses Core row zero at GL texel y=0.
fn native_texture_region(
    operation: &'static str,
    desc: GlTextureDesc,
    region: GlTextureRegion,
) -> Result<GlTextureRegion, GlError> {
    region
        .validate_for(desc)
        .map_err(|_| validation(operation, "invalid texture region"))?;
    Ok(region)
}

/// Converts one 2D rectangle's geometry into GLint command coordinates.
fn rect_coords(
    operation: &'static str,
    region: GlTextureRegion,
) -> Result<(i32, i32, i32, i32, i32), GlError> {
    let level = i32::try_from(region.subresource.mip_level)
        .map_err(|_| validation(operation, "mip level exceeds GLint"))?;
    let x =
        i32::try_from(region.origin[0]).map_err(|_| validation(operation, "x exceeds GLint"))?;
    let y =
        i32::try_from(region.origin[1]).map_err(|_| validation(operation, "y exceeds GLint"))?;
    let width = i32::try_from(region.extent.width)
        .map_err(|_| validation(operation, "width exceeds GLint"))?;
    let height = i32::try_from(region.extent.height)
        .map_err(|_| validation(operation, "height exceeds GLint"))?;
    Ok((level, x, y, width, height))
}

/// Packs positive-stride Core rows into tight native texture storage order.
fn packed_unpack_rows(
    operation: &'static str,
    layout: GlPixelLayout,
    region: GlTextureRegion,
    bytes: &[u8],
) -> Result<Vec<u8>, GlError> {
    let bpp = usize::try_from(layout.format.bytes_per_pixel())
        .map_err(|_| validation(operation, "pixel size exceeds address space"))?;
    let width_bytes = usize::try_from(region.extent.width)
        .ok()
        .and_then(|width| width.checked_mul(bpp))
        .ok_or_else(|| validation(operation, "row width exceeds address space"))?;
    let height = usize::try_from(region.extent.height)
        .map_err(|_| validation(operation, "height exceeds address space"))?;
    let offset = usize::try_from(layout.offset)
        .map_err(|_| validation(operation, "layout offset exceeds addressable range"))?;
    let row_stride = usize::try_from(layout.bytes_per_row)
        .map_err(|_| validation(operation, "row stride exceeds address space"))?;
    let image_stride = row_stride
        .checked_mul(layout.rows_per_image as usize)
        .ok_or_else(|| validation(operation, "image stride exceeds address space"))?;
    let image_bytes = height
        .checked_mul(width_bytes)
        .ok_or_else(|| validation(operation, "staging size exceeds address space"))?;
    let layers = usize::try_from(region.extent.depth_or_layers)
        .map_err(|_| validation(operation, "layer count exceeds address space"))?;
    let total = layers
        .checked_mul(image_bytes)
        .ok_or_else(|| validation(operation, "staging size exceeds address space"))?;
    let mut staged = Vec::new();
    staged
        .try_reserve_exact(total)
        .map_err(|_| GlError::OutOfMemory { operation })?;
    for image in 0..layers {
        let image_base = offset
            .checked_add(
                image
                    .checked_mul(image_stride)
                    .ok_or_else(|| validation(operation, "image address overflows"))?,
            )
            .ok_or_else(|| validation(operation, "image address overflows"))?;
        for gl_row in 0..height {
            let start = image_base
                .checked_add(
                    gl_row
                        .checked_mul(row_stride)
                        .ok_or_else(|| validation(operation, "row address overflows"))?,
                )
                .ok_or_else(|| validation(operation, "row address overflows"))?;
            let end = start
                .checked_add(width_bytes)
                .ok_or_else(|| validation(operation, "row address overflows"))?;
            staged.extend_from_slice(
                bytes
                    .get(start..end)
                    .ok_or_else(|| validation(operation, "source ends inside a row"))?,
            );
        }
    }
    Ok(staged)
}

/// Places native texture rows in Core order while preserving output padding.
fn place_rows(layout: GlPixelLayout, region: GlTextureRegion, body: &[u8], bytes: &mut [u8]) {
    place_rows_for_extent(layout, region.extent, body, bytes, false);
}

/// Scatters rows into the public layout. Window-system default framebuffer
/// rows retain their native bottom-to-top orientation and require reversal.
fn place_rows_for_extent(
    layout: GlPixelLayout,
    extent: crate::backend::gl::api::GlExtent3d,
    body: &[u8],
    bytes: &mut [u8],
    reverse: bool,
) {
    let bpp = layout.format.bytes_per_pixel() as usize;
    let width_bytes = (extent.width as usize) * bpp;
    let row = layout.bytes_per_row as usize;
    let mappable = layout.stride_is_mappable();
    let offset = layout.offset as usize;
    for image in 0..extent.depth_or_layers as usize {
        let image_base = offset + image * row * (layout.rows_per_image as usize);
        let read_base = if mappable {
            image * row * (layout.rows_per_image as usize)
        } else {
            image * (extent.height as usize) * width_bytes
        };
        for row_index in 0..extent.height as usize {
            let dst = image_base + row_index * row;
            let gl_row = if reverse {
                extent.height as usize - 1 - row_index
            } else {
                row_index
            };
            let src = read_base + gl_row * if mappable { row } else { width_bytes };
            if let (Some(dst), Some(src)) = (
                bytes.get_mut(dst..dst + width_bytes),
                body.get(src..src + width_bytes),
            ) {
                dst.copy_from_slice(src);
            }
        }
    }
}

#[path = "exec_copy_texture_planes.rs"]
mod planes;

#[path = "exec_copy_packed_upload.rs"]
pub(super) mod packed_upload;

#[path = "exec_copy_stencil_pack.rs"]
mod stencil_pack;

#[cfg(test)]
#[path = "exec_copy_depth_staging_tests.rs"]
mod depth_staging_tests;
