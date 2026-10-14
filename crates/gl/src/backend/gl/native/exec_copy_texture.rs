//! Native texture-copy routes and pixel-store state operations.
//!
//! This private child stays below the Core bridge. It issues a selected GL
//! copy or pixel-store operation immediately after the caller has validated
//! the provider-owned resource identities and profile capability.

use super::super::provider::NativeGlProvider;
use super::{UnpackDirection, rect_coords};
use crate::backend::gl::api::{
    GlError, GlFamilyProfile, GlPixelLayout, GlPixelStoreState, GlTextureDesc, GlTextureDimension,
    GlTextureRegion,
};

/// The texture-copy route the discovered profile fact selects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CopyRoute {
    /// `glCopyImageSubData` is core: GLES from 3.2, desktop from 4.3.
    DirectImage,
    /// Older desktop cores use the scratch-framebuffer route.
    Framebuffer,
}

pub(super) fn texture_copy_route(profile: GlFamilyProfile) -> CopyRoute {
    match profile {
        crate::backend::gl::api::GlFamilyProfile::Embedded { major, minor } => {
            if major > 3 || (major == 3 && minor >= 2) {
                CopyRoute::DirectImage
            } else {
                CopyRoute::Framebuffer
            }
        }
        crate::backend::gl::api::GlFamilyProfile::Desktop { major, minor } => {
            if major > 4 || (major == 4 && minor >= 3) {
                CopyRoute::DirectImage
            } else {
                CopyRoute::Framebuffer
            }
        }
        crate::backend::gl::api::GlFamilyProfile::WebGl2 => CopyRoute::Framebuffer,
    }
}

impl NativeGlProvider {
    pub(super) fn copy_texture_direct(
        &mut self,
        op: &'static str,
        source: GlTextureRegion,
        destination: GlTextureRegion,
        source_name: glow::NativeTexture,
        destination_name: glow::NativeTexture,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        let src_level = i32::try_from(source.subresource.mip_level)
            .map_err(|_| Self::validation(op, "mip level exceeds GLint"))?;
        let dst_level = i32::try_from(destination.subresource.mip_level)
            .map_err(|_| Self::validation(op, "mip level exceeds GLint"))?;
        let (_, dx, dy, width, height) = rect_coords(op, destination)?;
        let sx =
            i32::try_from(source.origin[0]).map_err(|_| Self::validation(op, "x exceeds GLint"))?;
        let sy =
            i32::try_from(source.origin[1]).map_err(|_| Self::validation(op, "y exceeds GLint"))?;
        // SAFETY: current-context contract; both live textures and the exact
        // rectangle were validated, and identical-format copies need no
        // format-conversion precondition.
        unsafe {
            self.gl.copy_image_sub_data(
                source_name,
                glow::TEXTURE_2D,
                src_level,
                sx,
                sy,
                0,
                destination_name,
                glow::TEXTURE_2D,
                dst_level,
                dx,
                dy,
                0,
                width,
                height,
                1,
            );
        }
        self.driver_error(op)
    }

    /// Framebuffer route for desktop cores before 4.3, mirroring the browser
    /// backend: a per-operation scratch framebuffer feeds `glCopyTexSubImage2D`.
    pub(super) fn copy_texture_framebuffer(
        &mut self,
        op: &'static str,
        source: GlTextureRegion,
        destination: GlTextureRegion,
        source_name: glow::NativeTexture,
        destination_name: glow::NativeTexture,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        let src_level = i32::try_from(source.subresource.mip_level)
            .map_err(|_| Self::validation(op, "mip level exceeds GLint"))?;
        let (level, dx, dy, width, height) = rect_coords(op, destination)?;
        let sx =
            i32::try_from(source.origin[0]).map_err(|_| Self::validation(op, "x exceeds GLint"))?;
        let sy =
            i32::try_from(source.origin[1]).map_err(|_| Self::validation(op, "y exceeds GLint"))?;
        // SAFETY: current-context contract; the scratch framebuffer exists only
        // inside this block and is unbound/deleted before every return.
        unsafe {
            let Ok(scratch) = self.gl.create_framebuffer() else {
                return Err(GlError::OutOfMemory { operation: op });
            };
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(scratch));
            self.gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(source_name),
                src_level,
            );
            let mut result = match self.require_complete(op) {
                Ok(()) => None,
                Err(error) => Some(Err(error)),
            };
            if result.is_none() {
                self.gl
                    .bind_texture(glow::TEXTURE_2D, Some(destination_name));
                self.gl.copy_tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    level,
                    dx,
                    dy,
                    sx,
                    sy,
                    width,
                    height,
                );
                result = Some(self.driver_error(op));
            }
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.delete_framebuffer(scratch);
            let _ = self.gl.get_error();
            result.expect("copy outcome was computed on every path")
        }
    }

    /// Compressed mips stay on the whole-mip route: storage is undefined until
    /// the first `compressedTexImage2D` defines it (audit P2-7).
    pub(super) fn upload_compressed_mip(
        &mut self,
        op: &'static str,
        name: glow::NativeTexture,
        desc: GlTextureDesc,
        destination: GlTextureRegion,
        bytes: &[u8],
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        let Some(info) = desc.format.compressed_info() else {
            return Err(Self::validation(op, "compressed info disappeared"));
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
            .map_err(|_| Self::validation(op, "compressed encoded size overflow"))?;
        if u64::try_from(bytes.len()).ok() != Some(exact) {
            return Err(Self::validation(
                op,
                "compressed bytes do not match exact block layout",
            ));
        }
        let internal = super::super::resource::native_texture_format(desc.format).ok_or(
            GlError::Unsupported {
                operation: op,
                reason: "compressed format has no native storage token",
            },
        )?;
        let level = i32::try_from(destination.subresource.mip_level)
            .map_err(|_| Self::validation(op, "mip level exceeds GLint"))?;
        let width = i32::try_from(destination.extent.width)
            .map_err(|_| Self::validation(op, "width exceeds GLsizei"))?;
        let height = i32::try_from(destination.extent.height)
            .map_err(|_| Self::validation(op, "height exceeds GLsizei"))?;
        let size = i32::try_from(exact)
            .map_err(|_| Self::validation(op, "compressed upload exceeds GLsizei"))?;
        // SAFETY: current-context contract; complete mip and exact block bytes were validated.
        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(name));
            self.gl.compressed_tex_image_2d(
                glow::TEXTURE_2D,
                level,
                internal as i32,
                width,
                height,
                0,
                size,
                bytes,
            );
        }
        self.driver_error(op)
    }

    /// Applies the tight lower-left staging layout before `texSubImage2D`.
    ///
    /// Upload staging reverses public rows, so it is always tight and starts at
    /// its first byte. Keeping caller layout values here would make GL skip or
    /// pad rows that no longer exist in the staging buffer.
    ///
    /// # Safety
    ///
    /// The caller owns the current provider context and restores the saved
    /// pixel-store state on every transfer exit path.
    pub(super) unsafe fn apply_tight_unpack(&self) {
        use glow::HasContext as _;
        unsafe {
            self.gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            self.gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 0);
            self.gl.pixel_store_i32(glow::UNPACK_IMAGE_HEIGHT, 0);
            self.gl.pixel_store_i32(glow::UNPACK_SKIP_PIXELS, 0);
            self.gl.pixel_store_i32(glow::UNPACK_SKIP_ROWS, 0);
            self.gl.pixel_store_i32(glow::UNPACK_SKIP_IMAGES, 0);
        }
    }

    /// Applies the pack half of one transfer layout before `readPixels`.
    ///
    /// # Safety
    ///
    /// The caller owns the current provider context; `layout` has already
    /// passed the transfer-shape checks that bound GLsizei conversions.
    pub(super) unsafe fn apply_pack(&self, layout: GlPixelLayout) {
        use glow::HasContext as _;
        let (alignment, row_length) = layout.staged_row_parameters();
        unsafe {
            self.gl
                .pixel_store_i32(glow::PACK_ALIGNMENT, alignment as i32);
            self.gl
                .pixel_store_i32(glow::PACK_ROW_LENGTH, row_length as i32);
        }
    }

    /// Restores the exact snapshot so tracked pixel-store state survives
    /// every return path, including driver failures.
    ///
    /// # Safety
    ///
    /// The caller owns the current provider context and passes the snapshot
    /// taken before it changed this direction's pixel-store fields.
    pub(super) unsafe fn restore_pixel_store(
        &mut self,
        direction: UnpackDirection,
        saved: GlPixelStoreState,
    ) {
        use glow::HasContext as _;
        unsafe {
            match direction {
                UnpackDirection::Unpack => {
                    self.gl
                        .pixel_store_i32(glow::UNPACK_ALIGNMENT, i32::from(saved.unpack_alignment));
                    self.gl
                        .pixel_store_i32(glow::UNPACK_ROW_LENGTH, saved.unpack_row_length as i32);
                    self.gl.pixel_store_i32(
                        glow::UNPACK_IMAGE_HEIGHT,
                        saved.unpack_image_height as i32,
                    );
                    self.gl
                        .pixel_store_i32(glow::UNPACK_SKIP_PIXELS, saved.unpack_skip_pixels as i32);
                    self.gl
                        .pixel_store_i32(glow::UNPACK_SKIP_ROWS, saved.unpack_skip_rows as i32);
                    self.gl
                        .pixel_store_i32(glow::UNPACK_SKIP_IMAGES, saved.unpack_skip_images as i32);
                }
                UnpackDirection::Pack => {
                    self.gl
                        .pixel_store_i32(glow::PACK_ALIGNMENT, i32::from(saved.pack_alignment));
                    self.gl
                        .pixel_store_i32(glow::PACK_ROW_LENGTH, saved.pack_row_length as i32);
                    self.gl
                        .pixel_store_i32(glow::PACK_SKIP_PIXELS, saved.pack_skip_pixels as i32);
                    self.gl
                        .pixel_store_i32(glow::PACK_SKIP_ROWS, saved.pack_skip_rows as i32);
                }
            }
            self.pixel_store = saved;
        }
    }
}

/// The `(format, type)` pair a native upload accepts for one texture format.
///
/// ES 3.0 and desktop GL 4.x accept exactly one `(format, type)` pair per
/// sized internal format, so unsupported pairs return `None` instead of a
/// driver-dependent guess.
pub(super) fn upload_encoding(format: crate::backend::gl::api::GlFormat) -> Option<(u32, u32)> {
    match format {
        crate::backend::gl::api::GlFormat::Rgba8Unorm
        | crate::backend::gl::api::GlFormat::Rgba8Srgb => Some((glow::RGBA, glow::UNSIGNED_BYTE)),
        _ => None,
    }
}
