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
        source_desc: GlTextureDesc,
        destination_desc: GlTextureDesc,
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
        let (src_target, src_z) = texture_copy_target_and_z(op, source_desc, source)?;
        let (dst_target, dst_z) = texture_copy_target_and_z(op, destination_desc, destination)?;
        let depth = i32::try_from(source.extent.depth_or_layers)
            .map_err(|_| Self::validation(op, "depth exceeds GLsizei"))?;
        if source_desc.dimension == GlTextureDimension::D1
            && (source.origin[1] != 0
                || source.origin[2] != 0
                || source.extent.height != 1
                || source.extent.depth_or_layers != 1
                || destination.origin[1] != 0
                || destination.origin[2] != 0
                || destination.extent.height != 1
                || destination.extent.depth_or_layers != 1)
        {
            return Err(Self::validation(op, "invalid 1D texture copy region"));
        }
        // SAFETY: current-context contract; both live textures and the exact
        // rectangle were validated, and identical-format copies need no
        // format-conversion precondition.
        unsafe {
            if let Some(copy) = self.copy_image_extension {
                copy(
                    source_name.0.get(),
                    src_target,
                    src_level,
                    sx,
                    sy,
                    src_z,
                    destination_name.0.get(),
                    dst_target,
                    dst_level,
                    dx,
                    dy,
                    dst_z,
                    width,
                    height,
                    depth,
                );
            } else {
                self.gl.copy_image_sub_data(
                    source_name,
                    src_target,
                    src_level,
                    sx,
                    sy,
                    src_z,
                    destination_name,
                    dst_target,
                    dst_level,
                    dx,
                    dy,
                    dst_z,
                    width,
                    height,
                    depth,
                );
            }
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
        source_desc: GlTextureDesc,
        destination_desc: GlTextureDesc,
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
        let (src_target, src_z) = texture_copy_target_and_z(op, source_desc, source)?;
        let (dst_target, dst_z) = texture_copy_target_and_z(op, destination_desc, destination)?;
        let depth = i32::try_from(source.extent.depth_or_layers)
            .map_err(|_| Self::validation(op, "depth exceeds GLsizei"))?;
        if source_desc.dimension == GlTextureDimension::D1
            && (source.origin[1] != 0
                || source.origin[2] != 0
                || source.extent.height != 1
                || source.extent.depth_or_layers != 1
                || destination.origin[1] != 0
                || destination.origin[2] != 0
                || destination.extent.height != 1
                || destination.extent.depth_or_layers != 1)
        {
            return Err(Self::validation(op, "invalid 1D texture copy region"));
        }
        if source_desc.dimension == GlTextureDimension::D1
            && (self.framebuffer_texture_1d.is_none() || self.copy_tex_sub_image_1d.is_none())
        {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "1D framebuffer copy entry points unavailable",
            });
        }
        // SAFETY: current-context contract; the scratch framebuffer exists only
        // inside this block and is unbound/deleted before every return.
        unsafe {
            let Ok(scratch) = self.gl.create_framebuffer() else {
                return Err(GlError::OutOfMemory { operation: op });
            };
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(scratch));
            let mut result = Ok(());
            for slice in 0..depth {
                if src_target == glow::TEXTURE_1D {
                    self.framebuffer_texture_1d
                        .expect("1D framebuffer entry point was checked before mutation")(
                        glow::FRAMEBUFFER,
                        glow::COLOR_ATTACHMENT0,
                        src_target,
                        source_name.0.get(),
                        src_level,
                    );
                } else if src_target == glow::TEXTURE_2D {
                    self.gl.framebuffer_texture_2d(
                        glow::FRAMEBUFFER,
                        glow::COLOR_ATTACHMENT0,
                        src_target,
                        Some(source_name),
                        src_level,
                    );
                } else {
                    self.gl.framebuffer_texture_layer(
                        glow::FRAMEBUFFER,
                        glow::COLOR_ATTACHMENT0,
                        Some(source_name),
                        src_level,
                        src_z + slice,
                    );
                }
                if let Err(error) = self.require_complete(op) {
                    result = Err(error);
                    break;
                }
                self.gl.bind_texture(dst_target, Some(destination_name));
                if dst_target == glow::TEXTURE_1D {
                    self.copy_tex_sub_image_1d
                        .expect("1D copy entry point was checked before mutation")(
                        dst_target, level, dx, sx, sy, width,
                    );
                } else if dst_target == glow::TEXTURE_2D {
                    self.gl
                        .copy_tex_sub_image_2d(dst_target, level, dx, dy, sx, sy, width, height);
                } else {
                    self.gl.copy_tex_sub_image_3d(
                        dst_target,
                        level,
                        dx,
                        dy,
                        dst_z + slice,
                        sx,
                        sy,
                        width,
                        height,
                    );
                }
                if let Err(error) = self.driver_error(op) {
                    result = Err(error);
                    break;
                }
            }
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.delete_framebuffer(scratch);
            let _ = self.gl.get_error();
            result
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

    /// Applies the tight staging layout before `texSubImage2D`.
    ///
    /// Upload staging removes row padding and starts at its first byte.
    /// Keeping caller layout values would make GL skip rows in that buffer.
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

fn texture_copy_target_and_z(
    op: &'static str,
    desc: GlTextureDesc,
    region: GlTextureRegion,
) -> Result<(u32, i32), GlError> {
    let (target, z) = match desc.dimension {
        GlTextureDimension::D1 => (glow::TEXTURE_1D, 0),
        GlTextureDimension::D2 => (glow::TEXTURE_2D, 0),
        GlTextureDimension::D2Array => (
            glow::TEXTURE_2D_ARRAY,
            region
                .subresource
                .base_layer
                .checked_add(region.origin[2])
                .ok_or_else(|| NativeGlProvider::validation(op, "layer overflows"))?,
        ),
        GlTextureDimension::D3 => (glow::TEXTURE_3D, region.origin[2]),
        _ => {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "texture copy target",
            });
        }
    };
    Ok((
        target,
        i32::try_from(z).map_err(|_| NativeGlProvider::validation(op, "layer exceeds GLint"))?,
    ))
}

/// The `(format, type)` pair a native upload accepts for one texture format.
///
/// Each pair preserves the RHI's encoded bytes; the native APIs may accept
/// additional conversion pairs which are deliberately not used here.
pub(in crate::backend::gl::native) fn upload_encoding(
    format: crate::backend::gl::api::GlFormat,
) -> Option<(u32, u32)> {
    use crate::backend::gl::api::GlFormat as F;
    match format {
        F::Depth16Unorm => Some((glow::DEPTH_COMPONENT, glow::UNSIGNED_SHORT)),
        F::Depth32Float => Some((glow::DEPTH_COMPONENT, glow::FLOAT)),
        F::R16Uint => Some((glow::RED_INTEGER, glow::UNSIGNED_SHORT)),
        F::R16Sint => Some((glow::RED_INTEGER, glow::SHORT)),
        F::R16Float => Some((glow::RED, glow::HALF_FLOAT)),
        F::R16Unorm => Some((glow::RED, glow::UNSIGNED_SHORT)),
        F::R16Snorm => Some((glow::RED, glow::SHORT)),
        F::Rg16Uint => Some((glow::RG_INTEGER, glow::UNSIGNED_SHORT)),
        F::Rg16Sint => Some((glow::RG_INTEGER, glow::SHORT)),
        F::Rg16Float => Some((glow::RG, glow::HALF_FLOAT)),
        F::Rg16Unorm => Some((glow::RG, glow::UNSIGNED_SHORT)),
        F::Rg16Snorm => Some((glow::RG, glow::SHORT)),
        F::Rgba16Uint => Some((glow::RGBA_INTEGER, glow::UNSIGNED_SHORT)),
        F::Rgba16Sint => Some((glow::RGBA_INTEGER, glow::SHORT)),
        F::Rgba16Float => Some((glow::RGBA, glow::HALF_FLOAT)),
        F::Rgba16Unorm => Some((glow::RGBA, glow::UNSIGNED_SHORT)),
        F::Rgba16Snorm => Some((glow::RGBA, glow::SHORT)),
        F::R32Uint => Some((glow::RED_INTEGER, glow::UNSIGNED_INT)),
        F::R32Sint => Some((glow::RED_INTEGER, glow::INT)),
        F::R32Float => Some((glow::RED, glow::FLOAT)),
        F::Rg32Uint => Some((glow::RG_INTEGER, glow::UNSIGNED_INT)),
        F::Rg32Sint => Some((glow::RG_INTEGER, glow::INT)),
        F::Rg32Float => Some((glow::RG, glow::FLOAT)),
        F::Rgba32Uint => Some((glow::RGBA_INTEGER, glow::UNSIGNED_INT)),
        F::Rgba32Sint => Some((glow::RGBA_INTEGER, glow::INT)),
        F::Rgba32Float => Some((glow::RGBA, glow::FLOAT)),
        F::Rgb9e5Ufloat => Some((glow::RGB, glow::UNSIGNED_INT_5_9_9_9_REV)),
        F::Rgb10a2Uint => Some((glow::RGBA_INTEGER, glow::UNSIGNED_INT_2_10_10_10_REV)),
        F::Rgb10a2Unorm => Some((glow::RGBA, glow::UNSIGNED_INT_2_10_10_10_REV)),
        F::Rg11b10Ufloat => Some((glow::RGB, glow::UNSIGNED_INT_10F_11F_11F_REV)),
        crate::backend::gl::api::GlFormat::R8Unorm => Some((glow::RED, glow::UNSIGNED_BYTE)),
        crate::backend::gl::api::GlFormat::R8Snorm => Some((glow::RED, glow::BYTE)),
        crate::backend::gl::api::GlFormat::R8Uint => Some((glow::RED_INTEGER, glow::UNSIGNED_BYTE)),
        crate::backend::gl::api::GlFormat::R8Sint => Some((glow::RED_INTEGER, glow::BYTE)),
        crate::backend::gl::api::GlFormat::Rg8Unorm => Some((glow::RG, glow::UNSIGNED_BYTE)),
        crate::backend::gl::api::GlFormat::Rg8Snorm => Some((glow::RG, glow::BYTE)),
        crate::backend::gl::api::GlFormat::Rg8Uint => Some((glow::RG_INTEGER, glow::UNSIGNED_BYTE)),
        crate::backend::gl::api::GlFormat::Rg8Sint => Some((glow::RG_INTEGER, glow::BYTE)),
        crate::backend::gl::api::GlFormat::Rgba8Snorm => Some((glow::RGBA, glow::BYTE)),
        crate::backend::gl::api::GlFormat::Rgba8Unorm
        | crate::backend::gl::api::GlFormat::Rgba8Srgb => Some((glow::RGBA, glow::UNSIGNED_BYTE)),
        crate::backend::gl::api::GlFormat::Rgba8Uint => {
            Some((glow::RGBA_INTEGER, glow::UNSIGNED_BYTE))
        }
        crate::backend::gl::api::GlFormat::Rgba8Sint => Some((glow::RGBA_INTEGER, glow::BYTE)),
        _ => None,
    }
}

pub(in crate::backend::gl::native) fn upload_pixel_format(
    format: crate::backend::gl::api::GlFormat,
) -> Option<crate::backend::gl::api::GlPixelFormat> {
    use crate::backend::gl::api::{GlFormat as F, GlPixelFormat as P};
    Some(match format {
        F::Depth16Unorm => P::Depth16,
        F::Depth32Float => P::Depth32Float,
        F::R8Unorm | F::R8Snorm | F::R8Uint | F::R8Sint => P::Red8,
        F::Rg8Unorm | F::Rg8Snorm | F::Rg8Uint | F::Rg8Sint => P::Rg8,
        F::Rgba8Unorm | F::Rgba8Srgb | F::Rgba8Snorm | F::Rgba8Uint | F::Rgba8Sint => P::Rgba8,
        F::R16Uint | F::R16Sint | F::R16Float | F::R16Unorm | F::R16Snorm => P::TexelBytes(2),
        F::Rg16Uint
        | F::Rg16Sint
        | F::Rg16Float
        | F::Rg16Unorm
        | F::Rg16Snorm
        | F::R32Uint
        | F::R32Sint
        | F::R32Float
        | F::Rgb9e5Ufloat
        | F::Rgb10a2Uint
        | F::Rgb10a2Unorm
        | F::Rg11b10Ufloat => P::TexelBytes(4),
        F::Rgba16Uint
        | F::Rgba16Sint
        | F::Rgba16Float
        | F::Rgba16Unorm
        | F::Rgba16Snorm
        | F::Rg32Uint
        | F::Rg32Sint
        | F::Rg32Float => P::TexelBytes(8),
        F::Rgba32Uint | F::Rgba32Sint | F::Rgba32Float => P::TexelBytes(16),
        _ => return None,
    })
}
