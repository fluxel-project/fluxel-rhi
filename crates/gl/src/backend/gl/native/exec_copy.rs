//! Native copy, upload, and readback execution.
//!
//! Texture copy has two honest routes: `glCopyImageSubData` when the profile's
//! core version supplies it (GLES 3.2+ and desktop GL 4.3+ context), and
//! otherwise the framebuffer route shared with the browser backend (a scratch
//! framebuffer served by `glCopyTexSubImage2D` / `glReadPixels`). The route
//! decision is a pure function of the discovered profile fact, recorded in
//! [`texture_copy_route`]. Provider entry points keep Core's WebGPU-style
//! upper-left texture coordinates; this module converts each validated region
//! exactly once before issuing the lower-left GL command.

use super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    GlBufferRange, GlCopyDomainApi, GlError, GlFamilyApi as _, GlFormatResourceKind, GlPixelFormat,
    GlPixelLayout, GlPixelStoreState, GlReadback, GlTextureDesc, GlTextureRegion, GlTextureUsage,
    validate_texture_copy,
};

impl GlCopyDomainApi for NativeGlProvider {
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
        let read_offset = i32::try_from(source.offset)
            .map_err(|_| Self::validation(OP, "source offset exceeds GLintptr"))?;
        let write_offset = i32::try_from(destination.offset)
            .map_err(|_| Self::validation(OP, "destination offset exceeds GLintptr"))?;
        let size = i32::try_from(source.size)
            .map_err(|_| Self::validation(OP, "copy size exceeds GLsizeiptr"))?;
        // The copy targets are Layer 1-private scratch: bind immediately before
        // use, no restore on return (`GlCopyDomainApi` documents why).
        // SAFETY: current-context contract; both live resources and all ranges
        // were validated before bindings or the copy command are changed.
        unsafe {
            self.gl
                .bind_buffer(glow::COPY_READ_BUFFER, Some(source_name));
            self.gl
                .bind_buffer(glow::COPY_WRITE_BUFFER, Some(destination_name));
            self.gl.copy_buffer_sub_data(
                glow::COPY_READ_BUFFER,
                glow::COPY_WRITE_BUFFER,
                read_offset,
                write_offset,
                size,
            );
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
        // Core texture coordinates are logical upper-left coordinates. Convert
        // each side against its own mip: differing mip heights make a shared
        // subtraction wrong even when the copied extent is equal.
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
        if source_desc.sample_count != 1 || destination_desc.sample_count != 1 {
            // The shared copy semantic is single-sample; multisample transfer
            // is a resolve word (`blit_framebuffer`), not a texture copy.
            return Err(Self::validation(
                OP,
                "copy operates on single-sample textures",
            ));
        }
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
            .get_for(GlFormatResourceKind::Texture, source_desc.format, 1)
            .map(|fact| fact.copy_source)
            .unwrap_or(false);
        let destination_copy = facts
            .get_for(GlFormatResourceKind::Texture, destination_desc.format, 1)
            .map(|fact| fact.copy_destination)
            .unwrap_or(false);
        if !source_copy || !destination_copy {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "format lacks a proved copy fact on this context",
            });
        }
        ensure_single_2d_rect(OP, source)?;
        ensure_single_2d_rect(OP, destination)?;
        match texture_copy_route(self.discovery.context().profile()) {
            CopyRoute::DirectImage => {
                self.copy_texture_direct(OP, source, destination, source_name, destination_name)
            }
            CopyRoute::Framebuffer => self.copy_texture_framebuffer(
                OP,
                source,
                destination,
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
        let offset = i32::try_from(destination.offset)
            .map_err(|_| Self::validation(OP, "offset exceeds GLintptr"))?;
        // COPY_WRITE_BUFFER keeps ARRAY_BUFFER and ELEMENT_ARRAY_BUFFER
        // vertex-state bindings untouched by transfer work.  The copy targets
        // themselves are Layer 1-private scratch: bound immediately before use,
        // not restored on return (`GlCopyDomainApi` documents why).
        // SAFETY: current-context contract; the exact byte slice was validated.
        unsafe {
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(name));
            self.gl
                .buffer_sub_data_u8_slice(glow::COPY_WRITE_BUFFER, offset, bytes);
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
        destination
            .validate_for(desc)
            .map_err(|_| Self::validation(OP, "invalid texture region"))?;
        if desc.format.compressed_info().is_some() {
            // Reversing a compressed block-row would still leave pixels inside
            // every block upside down. Until a format-aware codec is present,
            // reject rather than publish a texture that violates public row 0.
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "compressed upload cannot preserve upper-left texel rows",
            });
        }
        // Depth storage is filled through clears and FBO work in GL, never
        // from client pixels in this shared semantic; every other unmapped
        // format fails closed.
        let Some((gl_format, gl_type)) = upload_encoding(desc.format) else {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "format accepts no CPU pixel upload in this shared semantic",
            });
        };
        transfer_format(layout.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "pixel encoding has no native transfer route",
        })?;
        ensure_upload_2d_region(OP, desc, destination)?;
        let needed = layout
            .required_bytes(destination)
            .map_err(|_| Self::validation(OP, "invalid pixel layout"))?;
        if u64::try_from(bytes.len()).ok() != Some(needed) {
            return Err(Self::validation(
                OP,
                "upload source length differs from layout",
            ));
        }
        let native_destination = native_texture_region(OP, desc, destination)?;
        let (level, x, y, width, height) = rect_coords(OP, native_destination)?;
        // GL consumes the first client row at the lower edge of the native
        // rectangle. A WebGPU row zero is its upper edge, so pack a reversed
        // row sequence instead of relying on an unavailable negative stride.
        let payload = flipped_unpack_rows(OP, layout, destination, bytes)?;

        let saved = self.pixel_store;
        // SAFETY: current-context contract; the layout was validated and the
        // exact tracked pixel-store state is restored on every return path.
        let result = unsafe {
            self.apply_tight_unpack();
            match desc.dimension {
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
                crate::backend::gl::api::GlTextureDimension::D2Array => {
                    let z = destination
                        .subresource
                        .base_layer
                        .checked_add(destination.origin[2])
                        .and_then(|value| i32::try_from(value).ok())
                        .ok_or_else(|| Self::validation(OP, "array layer exceeds GLint"))?;
                    let depth = i32::try_from(destination.extent.depth_or_layers)
                        .map_err(|_| Self::validation(OP, "array layer count exceeds GLsizei"))?;
                    self.gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(name));
                    self.gl.tex_sub_image_3d(
                        glow::TEXTURE_2D_ARRAY,
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
        };
        // SAFETY: current-context contract; exact snapshot restoration.
        unsafe { self.restore_pixel_store(UnpackDirection::Unpack, saved) };
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
use texture::{CopyRoute, texture_copy_route, upload_encoding};

#[path = "exec_readback.rs"]
mod readback;

/// The readback encoding this backend returns for a discovered texture format.
///
/// `RGBA`/`UNSIGNED_BYTE` is the only combination every accepted profile must
/// support through the framebuffer route.
fn readback_encoding(format: crate::backend::gl::api::GlFormat) -> Option<(u32, u32)> {
    upload_encoding(format)
}

/// The client pixel format a transfer must use, if this backend accepts it.
fn transfer_format(format: GlPixelFormat) -> Option<()> {
    match format {
        GlPixelFormat::Rgba8 => Some(()),
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

/// Admits CPU uploads into either a 2D image or contiguous array layers.
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
        crate::backend::gl::api::GlTextureDimension::D2
            if region.subresource.base_layer == 0
                && region.subresource.layer_count == 1
                && region.origin[2] == 0
                && region.extent.depth_or_layers == 1 =>
        {
            Ok(())
        }
        crate::backend::gl::api::GlTextureDimension::D2Array => Ok(()),
        _ => Err(GlError::Unsupported {
            operation,
            reason: "this upload route supports two-dimensional images and array layers only",
        }),
    }
}

fn validation(operation: &'static str, message: &'static str) -> GlError {
    GlError::Validation {
        operation,
        message: message.into(),
    }
}

/// Converts a validated Core texture rectangle into the lower-left coordinate
/// system consumed by native GL texture commands.
///
/// `GlTextureRegion` deliberately reaches this boundary in Core's WebGPU
/// convention. Each mip has a distinct height, so conversion must happen only
/// after resolving the live descriptor and must not reuse a base-level height.
fn native_texture_region(
    operation: &'static str,
    desc: GlTextureDesc,
    mut region: GlTextureRegion,
) -> Result<GlTextureRegion, GlError> {
    let mip = desc
        .mip_extent(region.subresource.mip_level)
        .ok_or_else(|| validation(operation, "mip level is not live"))?;
    let upper_edge = region.origin[1]
        .checked_add(region.extent.height)
        .ok_or_else(|| validation(operation, "texture y range overflows"))?;
    region.origin[1] = mip
        .height
        .checked_sub(upper_edge)
        .ok_or_else(|| validation(operation, "texture y range exceeds its mip"))?;
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

/// Produces tight lower-left rows for a native GL texture upload.
///
/// GL has no negative pixel-row stride. The staging copy is therefore required
/// even when the caller's positive stride was otherwise mappable: the first
/// WebGPU row becomes the last native row. Array layers retain their order;
/// each image's rows are reversed independently.
fn flipped_unpack_rows(
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
        for gl_row in (0..height).rev() {
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

/// Places readback rows into the caller-declared output layout.
///
/// GL rows arrive from the lower edge first. This helper preserves the caller's
/// declared pitch and padding while scattering them in upper-left row order;
/// mappable layouts still require that reversal.
fn place_rows(layout: GlPixelLayout, region: GlTextureRegion, body: &[u8], bytes: &mut [u8]) {
    place_rows_for_extent(layout, region.extent, body, bytes);
}

/// Scatters lower-left GL rows into one public upper-left output extent.
///
/// Default-framebuffer readback has no owned `GlTextureRegion`, but must obey
/// the same public row-zero convention as texture readback.
fn place_rows_for_extent(
    layout: GlPixelLayout,
    extent: crate::backend::gl::api::GlExtent3d,
    body: &[u8],
    bytes: &mut [u8],
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
            let gl_row = extent.height as usize - 1 - row_index;
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
