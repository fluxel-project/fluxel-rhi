//! Pixel-store, packing, and rectangle helpers for WebGL2 transfers.
//!
//! Helpers stay separate from copy issue verbs so each transfer can restore its
//! temporary pixel-store state before another browser command observes it.

//! Browser copy, upload, and readback execution.
//!
//! WebGL2 exposes no direct texture-to-texture copy, so `copy_texture_region`
//! and `read_texture` honestly take the framebuffer route: the source region
//! is attached to a per-operation scratch framebuffer and served with
//! `copyTexSubImage2D` / `readPixels`. This is the backend-provided route for
//! the common copy semantic (plan functional order 3), not a claim that a
//! `glCopyImageSubData`-style command exists. Region coordinates follow the
//! the public upper-left convention. Texture row zero is stored at GL row zero.

use std::borrow::Cow;
use wasm_bindgen::JsValue;
use web_sys::{WebGl2RenderingContext as Gl, WebGlFramebuffer};

use super::super::api::{
    GlBufferRange, GlCopyDomainApi, GlError, GlFormatResourceKind, GlPixelLayout,
    GlPixelStoreState, GlReadback, GlRepackPolicy, GlTextureDesc, GlTextureDimension,
    GlTextureRegion, GlTextureUsage, validate_texture_copy,
};
use super::discovery::WebGl2BrowserDiscovery;

/// Scoped pixel-store half to restore.
#[derive(Clone, Copy)]
pub(super) enum PixelDirection {
    Unpack,
    Pack,
}

impl WebGl2BrowserDiscovery {
    /// Unbinds and deletes a per-operation scratch framebuffer.
    pub(super) fn cleanup_scratch(&self, scratch: &WebGlFramebuffer) {
        self.raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
        self.raw.delete_framebuffer(Some(scratch));
    }

    /// Applies the unpack half of one transfer layout before `texSubImage2D`.
    ///
    /// The client offset is applied by slicing the payload, so GL skip
    /// parameters stay zero and the tracked snapshot resumes exactly.
    pub(super) fn apply_unpack(&mut self, layout: GlPixelLayout) {
        // WebGL2 only permits UNPACK_FLIP_Y_WEBGL for client ArrayBufferView
        // uploads. This backend's 2D upload route deliberately has no PBO
        // source, so clear a stale private transfer binding before enabling
        // the flip rather than relying on ambient context state.
        self.raw.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, None);
        // The same pair the staging branches on, so WebGL2 is told the pitch of
        // the payload it is actually given rather than of the caller's buffer.
        // Reading the caller's pitch over tight staging is an out-of-bounds
        // read on the native provider and a structured error here, which is how
        // the two providers came to disagree about the same input.
        let (alignment, row_length) = layout.staged_row_parameters();
        let mappable = layout.stride_is_mappable();
        self.raw
            .pixel_storei(Gl::UNPACK_ALIGNMENT, alignment as i32);
        self.raw
            .pixel_storei(Gl::UNPACK_ROW_LENGTH, row_length as i32);
        self.raw.pixel_storei(
            Gl::UNPACK_IMAGE_HEIGHT,
            if mappable {
                layout.rows_per_image as i32
            } else {
                0
            },
        );
        self.raw.pixel_storei(Gl::UNPACK_SKIP_PIXELS, 0);
        self.raw.pixel_storei(Gl::UNPACK_SKIP_ROWS, 0);
        self.raw.pixel_storei(Gl::UNPACK_SKIP_IMAGES, 0);
        self.raw.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, 0);
    }

    /// Applies the pack half of one transfer layout before `readPixels`.
    pub(super) fn apply_pack(&mut self, layout: GlPixelLayout) {
        let (alignment, row_length) = layout.staged_row_parameters();
        self.raw.pixel_storei(Gl::PACK_ALIGNMENT, alignment as i32);
        self.raw
            .pixel_storei(Gl::PACK_ROW_LENGTH, row_length as i32);
    }

    /// Restores the exact snapshot so tracked pixel-store state survives
    /// every return path, including driver and browser failures.
    fn restore_pixel_store(&mut self, saved: &GlPixelStoreState, direction: PixelDirection) {
        match direction {
            PixelDirection::Unpack => {
                self.raw
                    .pixel_storei(Gl::UNPACK_ALIGNMENT, i32::from(saved.unpack_alignment));
                self.raw
                    .pixel_storei(Gl::UNPACK_ROW_LENGTH, saved.unpack_row_length as i32);
                self.raw
                    .pixel_storei(Gl::UNPACK_IMAGE_HEIGHT, saved.unpack_image_height as i32);
                self.raw
                    .pixel_storei(Gl::UNPACK_SKIP_PIXELS, saved.unpack_skip_pixels as i32);
                self.raw
                    .pixel_storei(Gl::UNPACK_SKIP_ROWS, saved.unpack_skip_rows as i32);
                self.raw
                    .pixel_storei(Gl::UNPACK_SKIP_IMAGES, saved.unpack_skip_images as i32);
                self.raw.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, 0);
            }
            PixelDirection::Pack => {
                self.raw
                    .pixel_storei(Gl::PACK_ALIGNMENT, i32::from(saved.pack_alignment));
                self.raw
                    .pixel_storei(Gl::PACK_ROW_LENGTH, saved.pack_row_length as i32);
                self.raw
                    .pixel_storei(Gl::PACK_SKIP_PIXELS, saved.pack_skip_pixels as i32);
                self.raw
                    .pixel_storei(Gl::PACK_SKIP_ROWS, saved.pack_skip_rows as i32);
            }
        }
        self.pixel_store = *saved;
    }

    /// Compressed mips stay on the existing route: storage is undefined until
    /// the first `compressedTexImage2D` defines it (audit P2-7).
    pub(super) fn upload_compressed_mip(
        &mut self,
        op: &'static str,
        raw: &web_sys::WebGlTexture,
        desc: GlTextureDesc,
        destination: GlTextureRegion,
        bytes: &[u8],
    ) -> Result<(), GlError> {
        let Some(info) = desc.format.compressed_info() else {
            return Err(validation(op, "compressed info disappeared"));
        };
        if desc.dimension != GlTextureDimension::D2
            || destination.subresource.base_layer != 0
            || destination.subresource.layer_count != 1
            || destination.origin != [0; 3]
            || destination.extent.depth_or_layers != 1
            || desc.mip_extent(destination.subresource.mip_level) != Some(destination.extent)
        {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "compressed upload must define one complete 2D mip",
            });
        }
        let exact = info
            .checked_encoded_size(destination.extent.width, destination.extent.height)
            .map_err(|_| validation(op, "compressed encoded size overflow"))?;
        if u64::try_from(bytes.len()).ok() != Some(exact) {
            return Err(validation(
                op,
                "compressed bytes do not match exact block layout",
            ));
        }
        let internal =
            super::format_map::internal_format(desc.format).ok_or(GlError::Unsupported {
                operation: op,
                reason: "compressed format has no WebGL2 internal format",
            })?;
        let level = i32::try_from(destination.subresource.mip_level)
            .map_err(|_| validation(op, "mip level exceeds GLint"))?;
        let width = i32::try_from(destination.extent.width)
            .map_err(|_| validation(op, "width exceeds GLint"))?;
        let height = i32::try_from(destination.extent.height)
            .map_err(|_| validation(op, "height exceeds GLint"))?;
        self.raw.bind_texture(Gl::TEXTURE_2D, Some(raw));
        self.raw.compressed_tex_image_2d_with_u8_array(
            Gl::TEXTURE_2D,
            level,
            internal,
            width,
            height,
            0,
            bytes,
        );
        self.driver_error(op)
    }

    /// Restores one pixel-store half and immediately consumes any browser
    /// error caused by restoration.  Pixel-store state is shared GL context
    /// state, so letting an error escape into the following action would make
    /// both the error attribution and cache knowledge false.
    pub(super) fn restore_pixel_store_checked(
        &mut self,
        saved: &GlPixelStoreState,
        direction: PixelDirection,
    ) -> Result<(), GlError> {
        self.restore_pixel_store(saved, direction);
        self.driver_error("restore-pixel-store")
    }
}

pub(super) fn js_failure(operation: &'static str, value: JsValue) -> GlError {
    GlError::Driver {
        operation,
        message: format!("browser exception: {value:?}"),
    }
}

pub(super) fn validation(operation: &'static str, message: &'static str) -> GlError {
    GlError::Validation {
        operation,
        message: message.into(),
    }
}

/// Rejects regions that are not one axis-aligned rectangle of one slice.
pub(super) fn ensure_single_2d_rect(
    operation: &'static str,
    region: GlTextureRegion,
) -> Result<(), GlError> {
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

/// Converts a public upper-left rectangle into one GL lower-left rectangle.
pub(super) fn rect_coords(
    operation: &'static str,
    region: GlTextureRegion,
    descriptor: GlTextureDesc,
) -> Result<(i32, i32, i32, i32, i32), GlError> {
    let level = i32::try_from(region.subresource.mip_level)
        .map_err(|_| validation(operation, "mip level exceeds GLint"))?;
    let x =
        i32::try_from(region.origin[0]).map_err(|_| validation(operation, "x exceeds GLint"))?;
    let mip = descriptor
        .mip_extent(region.subresource.mip_level)
        .ok_or_else(|| validation(operation, "mip level is outside texture storage"))?;
    if region.origin[1]
        .checked_add(region.extent.height)
        .is_none_or(|end| end > mip.height)
    {
        return Err(validation(operation, "rectangle leaves texture height"));
    }
    let y =
        i32::try_from(region.origin[1]).map_err(|_| validation(operation, "y exceeds GLint"))?;
    let width = i32::try_from(region.extent.width)
        .map_err(|_| validation(operation, "width exceeds GLint"))?;
    let height = i32::try_from(region.extent.height)
        .map_err(|_| validation(operation, "height exceeds GLint"))?;
    Ok((level, x, y, width, height))
}

/// Produces the exact client bytes one `texSubImage2D` call reads.
///
/// GL expresses any layout whose row pitch is a whole number of pixels wide
/// and divisible by the requested alignment; otherwise a staging copy is
/// packed tightly, exactly as [`GlRepackPolicy::Bounded`] permits. The
/// returned bytes always start at the layout offset so GL skip parameters
/// remain zero.
pub(super) fn staged_unpack<'a>(
    operation: &'static str,
    layout: GlPixelLayout,
    region: GlTextureRegion,
    bytes: &'a [u8],
) -> Result<Cow<'a, [u8]>, GlError> {
    let bpp = layout.format.bytes_per_pixel();
    let mappable = layout.stride_is_mappable();
    let offset = usize::try_from(layout.offset)
        .map_err(|_| validation(operation, "layout offset exceeds addressable range"))?;
    if offset > bytes.len() {
        return Err(validation(operation, "layout offset leaves the source"));
    }
    if mappable {
        return Ok(Cow::Borrowed(&bytes[offset..]));
    }
    match layout.repack {
        GlRepackPolicy::Disallow => Err(validation(
            operation,
            "row stride is not expressible and repack is disallowed",
        )),
        GlRepackPolicy::Bounded { max_bytes } => {
            let total = layout
                .required_bytes(region)
                .map_err(|_| validation(operation, "invalid pixel layout"))?;
            if total > max_bytes {
                return Err(validation(
                    operation,
                    "repack staging exceeds the declared bound",
                ));
            }
            let width_bytes = (region.extent.width as usize) * (bpp as usize);
            let row = layout.bytes_per_row as usize;
            let mut staged = Vec::with_capacity((region.extent.height as usize) * width_bytes);
            for image in 0..region.extent.depth_or_layers as usize {
                let image_base = offset + image * row * (layout.rows_per_image as usize);
                for row_index in 0..region.extent.height as usize {
                    let start = image_base + row_index * row;
                    staged.extend_from_slice(
                        bytes
                            .get(start..start + width_bytes)
                            .ok_or_else(|| validation(operation, "source ends inside a row"))?,
                    );
                }
            }
            Ok(Cow::Owned(staged))
        }
    }
}

/// Places readback rows into the caller-declared output layout.
///
/// Place native texture rows into the public layout, including padding.
pub(super) fn place_rows(
    layout: GlPixelLayout,
    region: GlTextureRegion,
    body: &[u8],
    bytes: &mut [u8],
) {
    let bpp = layout.format.bytes_per_pixel() as usize;
    let width_bytes = (region.extent.width as usize) * bpp;
    let row = layout.bytes_per_row as usize;
    let mappable = layout.stride_is_mappable();
    let offset = layout.offset as usize;
    let source_row = if mappable { row } else { width_bytes };
    let source_image_rows = if mappable {
        layout.rows_per_image as usize
    } else {
        region.extent.height as usize
    };
    for image in 0..region.extent.depth_or_layers as usize {
        let image_base = offset + image * row * (layout.rows_per_image as usize);
        let read_base = image * source_image_rows * source_row;
        for row_index in 0..region.extent.height as usize {
            let dst = image_base + row_index * row;
            let src = read_base + row_index * source_row;
            if let (Some(dst), Some(src)) = (
                bytes.get_mut(dst..dst + width_bytes),
                body.get(src..src + width_bytes),
            ) {
                dst.copy_from_slice(src);
            }
        }
    }
}
