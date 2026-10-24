//! Native GL resource allocation and sampler ownership.
//!
//! This module owns the provider operations that allocate or retire GL object
//! names. `provider` retains the context-wide tables and lifecycle facts; each
//! operation here runs only after that provider has established currentness.
//! Allocation validates the immutable private descriptor before mutating GL,
//! and every failure path deletes a name that was not published into its table.

use super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    self, BufferId, GlBufferDesc, GlError, GlFamilyApi as _, GlRenderBufferDesc,
    GlResourceApi as _, GlSamplerApi as _, GlSamplerDesc, GlTextureDesc, RenderbufferId, SamplerId,
    TextureId,
};
#[path = "resource/buffer.rs"]
mod buffer;
#[path = "resource/mapping.rs"]
mod mapping;
pub(crate) use mapping::native_texture_format_supported;
pub(super) use mapping::{depth_attachment_point, has_stencil_plane, native_texture_format};
use mapping::{native_compare, native_mag, native_min, native_wrap};

#[cfg(test)]
impl NativeGlProvider {
    pub(crate) fn disable_texture_storage_for_test(&mut self) {
        self.texture_storage_entries = false;
    }
}

impl crate::backend::gl::api::GlResourceApi for NativeGlProvider {
    fn create_buffer_resource(&mut self, desc: GlBufferDesc) -> Result<BufferId, GlError> {
        self.create_native_buffer(desc)
    }
    fn create_texture_resource(&mut self, desc: GlTextureDesc) -> Result<TextureId, GlError> {
        use glow::HasContext as _;
        const OP: &str = "create-texture";
        self.assert_ready(OP)?;
        desc.validate()
            .map_err(|_| Self::validation(OP, "invalid texture descriptor"))?;
        let internal = native_texture_format(desc.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "format has no proven native texture storage mapping",
        })?;
        let limits = self.discovery.limits();
        validate_texture_limits(&limits, desc).map_err(|reason| GlError::Unsupported {
            operation: OP,
            reason,
        })?;
        let class = texture_storage_class(
            self.discovery.context().profile(),
            &limits,
            self.discovery.formats(),
            desc,
        )
        .map_err(|reason| GlError::Unsupported {
            operation: OP,
            reason,
        })?;
        if desc.sample_count > 1
            && desc.dimension == api::GlTextureDimension::D2Array
            && self.multisample_array.is_none()
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "multisample array storage entry unavailable",
            });
        }
        if desc.format.compressed_info().is_some()
            && !matches!(
                desc.dimension,
                crate::backend::gl::api::GlTextureDimension::D2
                    | crate::backend::gl::api::GlTextureDimension::D2Array
            )
            && !(desc.dimension == api::GlTextureDimension::D3
                && desc.format.supports_compressed_3d(
                    self.discovery.context().profile(),
                    self.discovery.extensions(),
                ))
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "compressed native texture dimension",
            });
        }
        let storage_available = self.texture_storage_entries
            && match self.discovery.context().profile() {
                api::GlFamilyProfile::Desktop { major, minor } => {
                    major > 4
                        || (major == 4 && minor >= 2)
                        || self
                            .discovery
                            .extensions()
                            .raw_reported_names()
                            .any(|name| name == "GL_ARB_texture_storage")
                }
                api::GlFamilyProfile::Embedded { major, .. } => major >= 3,
                _ => false,
            };
        if !storage_available
            && desc.format.compressed_info().is_none()
            && mutable_texture_format(desc.format).is_none()
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "mutable texture format unavailable",
            });
        }
        let mut compressed_mips = Vec::new();
        if let Some(info) = desc.format.compressed_info() {
            if !storage_available {
                if (desc.dimension == api::GlTextureDimension::D2
                    && self.compressed_image_2d.is_none())
                    || (matches!(
                        desc.dimension,
                        api::GlTextureDimension::D2Array | api::GlTextureDimension::D3
                    ) && self.compressed_image_3d.is_none())
                {
                    return Err(GlError::Unsupported {
                        operation: OP,
                        reason: "compressed allocation entry point unavailable",
                    });
                }
                for level in 0..desc.mip_level_count {
                    let extent = desc
                        .mip_extent(level)
                        .ok_or_else(|| Self::validation(OP, "invalid mip"))?;
                    let size = info
                        .checked_encoded_size(extent.width, extent.height)
                        .map_err(|_| Self::validation(OP, "compressed size overflow"))?
                        .checked_mul(u64::from(extent.depth_or_layers))
                        .and_then(|v| i32::try_from(v).ok())
                        .ok_or_else(|| Self::validation(OP, "compressed image exceeds GLsizei"))?;
                    compressed_mips.push((
                        level as i32,
                        extent.width as i32,
                        extent.height as i32,
                        extent.depth_or_layers as i32,
                        size,
                    ));
                }
            }
        }
        let width = i32::try_from(desc.extent.width)
            .map_err(|_| Self::validation(OP, "width exceeds GLsizei"))?;
        let height = i32::try_from(desc.extent.height)
            .map_err(|_| Self::validation(OP, "height exceeds GLsizei"))?;
        let layers = i32::try_from(desc.extent.depth_or_layers)
            .map_err(|_| Self::validation(OP, "layer count exceeds GLsizei"))?;
        let levels = i32::try_from(desc.mip_level_count)
            .map_err(|_| Self::validation(OP, "mip count exceeds GLsizei"))?;
        let samples = i32::try_from(desc.sample_count)
            .map_err(|_| Self::validation(OP, "sample count exceeds GLsizei"))?;
        // SAFETY: current-context contract; all profile/format/size validation preceded mutation.
        let name = unsafe { self.gl.create_texture() }.map_err(|message| GlError::Driver {
            operation: OP,
            message,
        })?;
        // SAFETY: `name` belongs to this current context and every target,
        // storage class, extent, and format was validated before allocation.
        // A driver error deletes this unpublished name below.
        unsafe {
            let (restore_target, restore_binding) = match class {
                TextureStorageClass::SingleSample => match desc.dimension {
                    api::GlTextureDimension::D1 => (glow::TEXTURE_1D, glow::TEXTURE_BINDING_1D),
                    api::GlTextureDimension::D2 => (glow::TEXTURE_2D, glow::TEXTURE_BINDING_2D),
                    api::GlTextureDimension::D2Array => {
                        (glow::TEXTURE_2D_ARRAY, glow::TEXTURE_BINDING_2D_ARRAY)
                    }
                    api::GlTextureDimension::D3 => (glow::TEXTURE_3D, glow::TEXTURE_BINDING_3D),
                    _ => unreachable!(),
                },
                TextureStorageClass::Multisample
                    if desc.dimension == api::GlTextureDimension::D2Array =>
                {
                    (
                        glow::TEXTURE_2D_MULTISAMPLE_ARRAY,
                        glow::TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY,
                    )
                }
                TextureStorageClass::Multisample => (
                    glow::TEXTURE_2D_MULTISAMPLE,
                    glow::TEXTURE_BINDING_2D_MULTISAMPLE,
                ),
            };
            let previous = self.gl.get_parameter_i32(restore_binding) as u32;
            match class {
                TextureStorageClass::SingleSample => {
                    let target = single_sample_texture_target(desc.dimension)
                        .expect("storage class already admitted this texture dimension");
                    self.gl.bind_texture(target, Some(name));
                    if !storage_available {
                        // Mutable GL 4.0 fallback defines exactly this mip range;
                        // sampler completeness must not require unallocated mips.
                        self.gl
                            .tex_parameter_i32(target, glow::TEXTURE_MAX_LEVEL, levels - 1);
                    }
                    // Samplerless texelFetch still needs a complete texture.
                    // Nearest is legal for integer and unfilterable float
                    // formats; sampler objects override these defaults.
                    self.gl.tex_parameter_i32(
                        target,
                        glow::TEXTURE_MIN_FILTER,
                        glow::NEAREST_MIPMAP_NEAREST as i32,
                    );
                    self.gl.tex_parameter_i32(
                        target,
                        glow::TEXTURE_MAG_FILTER,
                        glow::NEAREST as i32,
                    );
                    if storage_available {
                        match desc.dimension {
                            crate::backend::gl::api::GlTextureDimension::D1 => {
                                self.gl.tex_storage_1d(target, levels, internal, width);
                            }
                            crate::backend::gl::api::GlTextureDimension::D2 => {
                                self.gl
                                    .tex_storage_2d(target, levels, internal, width, height);
                            }
                            crate::backend::gl::api::GlTextureDimension::D2Array
                            | crate::backend::gl::api::GlTextureDimension::D3 => {
                                self.gl.tex_storage_3d(
                                    target, levels, internal, width, height, layers,
                                );
                            }
                            _ => unreachable!(
                                "texture_storage_class admitted an unsupported dimension"
                            ),
                        }
                    } else if desc.format.compressed_info().is_none() {
                        let old_unpack =
                            self.gl.get_parameter_i32(glow::PIXEL_UNPACK_BUFFER_BINDING) as u32;
                        self.gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
                        let (format, ty) = mutable_texture_format(desc.format)
                            .expect("validated mutable texture format");
                        for level in 0..desc.mip_level_count {
                            let e = desc.mip_extent(level).expect("validated mip");
                            if desc.dimension == api::GlTextureDimension::D1 {
                                self.gl.tex_image_1d(
                                    target,
                                    level as i32,
                                    internal as i32,
                                    e.width as i32,
                                    0,
                                    format,
                                    ty,
                                    glow::PixelUnpackData::Slice(None),
                                );
                            } else if desc.dimension == api::GlTextureDimension::D2 {
                                self.gl.tex_image_2d(
                                    target,
                                    level as i32,
                                    internal as i32,
                                    e.width as i32,
                                    e.height as i32,
                                    0,
                                    format,
                                    ty,
                                    glow::PixelUnpackData::Slice(None),
                                );
                            } else {
                                self.gl.tex_image_3d(
                                    target,
                                    level as i32,
                                    internal as i32,
                                    e.width as i32,
                                    e.height as i32,
                                    e.depth_or_layers as i32,
                                    0,
                                    format,
                                    ty,
                                    glow::PixelUnpackData::Slice(None),
                                );
                            }
                        }
                        self.gl.bind_buffer(
                            glow::PIXEL_UNPACK_BUFFER,
                            std::num::NonZeroU32::new(old_unpack).map(glow::NativeBuffer),
                        );
                    } else {
                        let old_unpack =
                            self.gl.get_parameter_i32(glow::PIXEL_UNPACK_BUFFER_BINDING) as u32;
                        self.gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
                        for &(level, width, height, depth, size) in &compressed_mips {
                            if desc.dimension == api::GlTextureDimension::D2 {
                                (self.compressed_image_2d.unwrap())(
                                    target,
                                    level,
                                    internal,
                                    width,
                                    height,
                                    0,
                                    size,
                                    core::ptr::null(),
                                );
                            } else {
                                (self.compressed_image_3d.unwrap())(
                                    target,
                                    level,
                                    internal,
                                    width,
                                    height,
                                    depth,
                                    0,
                                    size,
                                    core::ptr::null(),
                                );
                            }
                        }
                        self.gl.bind_buffer(
                            glow::PIXEL_UNPACK_BUFFER,
                            std::num::NonZeroU32::new(old_unpack).map(glow::NativeBuffer),
                        );
                    }
                }
                TextureStorageClass::Multisample => {
                    // Sample locations are fixed: a resolve reads every texel's
                    // samples as one block, and the per-texel locations that the
                    // alternative would expose are not expressible in this
                    // layer's vocabulary.
                    self.gl.bind_texture(restore_target, Some(name));
                    if desc.dimension == api::GlTextureDimension::D2Array {
                        self.multisample_array.unwrap()(
                            restore_target,
                            samples,
                            internal,
                            width,
                            height,
                            layers,
                            1,
                        );
                    } else if self.discovery.context().profile().meets(
                        Some(api::GlVersion::new(4, 3)),
                        Some(api::GlVersion::new(3, 1)),
                    ) {
                        self.gl.tex_storage_2d_multisample(
                            glow::TEXTURE_2D_MULTISAMPLE,
                            samples,
                            internal,
                            width,
                            height,
                            true,
                        );
                    } else {
                        // Desktop GL 4.0 has mutable multisample texture images;
                        // immutable TexStorage2DMultisample only arrived in 4.3.
                        self.gl.tex_image_2d_multisample(
                            glow::TEXTURE_2D_MULTISAMPLE,
                            samples,
                            internal as i32,
                            width,
                            height,
                            true,
                        );
                    }
                }
            }
            self.gl.bind_texture(
                restore_target,
                std::num::NonZeroU32::new(previous).map(glow::NativeTexture),
            );
        }
        if let Err(error) = self.driver_error(OP) {
            // SAFETY: rollback deletes the unpublished texture created by this
            // same current context, regardless of which allocation call failed.
            unsafe { self.gl.delete_texture(name) };
            return Err(error);
        }
        let id = TextureId::new(self.context_stamp(), self.slot(OP)?, 0);
        self.textures.insert(id, (name, desc));
        Ok(id)
    }
    fn create_render_buffer(
        &mut self,
        desc: GlRenderBufferDesc,
    ) -> Result<RenderbufferId, GlError> {
        use glow::HasContext as _;
        const OP: &str = "create-render-buffer";
        self.assert_ready(OP)?;
        desc.validate()
            .map_err(|_| Self::validation(OP, "invalid renderbuffer descriptor"))?;
        let limits = self.discovery.limits();
        if desc.width > limits.max_renderbuffer_size || desc.height > limits.max_renderbuffer_size {
            return Err(Self::validation(
                OP,
                "renderbuffer extent exceeds the discovered limit",
            ));
        }
        if desc.samples > limits.max_samples {
            return Err(Self::validation(
                OP,
                "renderbuffer sample count exceeds the discovered limit",
            ));
        }
        let facts = self
            .discovery
            .formats()
            .get_for(
                crate::backend::gl::api::GlFormatResourceKind::Renderbuffer,
                desc.format,
                desc.samples,
            )
            .ok_or(GlError::Unsupported {
                operation: OP,
                reason: "no exact renderbuffer format fact for this context",
            })?;
        if !facts.renderable {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "format lacks renderable evidence at this sample count",
            });
        }
        let internal = native_texture_format(desc.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "format has no proven native renderbuffer mapping",
        })?;
        let width =
            i32::try_from(desc.width).map_err(|_| Self::validation(OP, "width exceeds GLsizei"))?;
        let height = i32::try_from(desc.height)
            .map_err(|_| Self::validation(OP, "height exceeds GLsizei"))?;
        // SAFETY: current-context contract; limits and facts were checked first.
        let name = unsafe { self.gl.create_renderbuffer() }.map_err(|message| GlError::Driver {
            operation: OP,
            message,
        })?;
        // SAFETY: `name` belongs to this current context; the validated format,
        // extent, and sample count were converted before binding or storage
        // allocation. The rollback below owns the unpublished object on error.
        unsafe {
            self.gl.bind_renderbuffer(glow::RENDERBUFFER, Some(name));
            if desc.samples > 1 {
                self.gl.renderbuffer_storage_multisample(
                    glow::RENDERBUFFER,
                    desc.samples as i32,
                    internal,
                    width,
                    height,
                );
            } else {
                self.gl
                    .renderbuffer_storage(glow::RENDERBUFFER, internal, width, height);
            }
        }
        if let Err(error) = self.driver_error(OP) {
            // SAFETY: rollback deletes only this current-context renderbuffer;
            // no public identity was inserted before the error check.
            unsafe { self.gl.delete_renderbuffer(name) };
            return Err(error);
        }
        let id = RenderbufferId::new(self.context_stamp(), self.slot(OP)?, 0);
        self.renderbuffers.insert(id, (name, desc));
        Ok(id)
    }
    fn destroy_buffer_resource(&mut self, id: BufferId) -> Result<(), GlError> {
        self.destroy_native_buffer(id)
    }
    fn destroy_texture_resource(&mut self, id: TextureId) -> Result<(), GlError> {
        use glow::HasContext as _;
        self.assert_ready("destroy-texture")?;
        let (name, _) = self.texture("destroy-texture", id)?;
        // SAFETY: current-context contract; liveness was checked before mutation.
        unsafe { self.gl.delete_texture(name) };
        self.driver_error("destroy-texture")?;
        self.textures.remove(&id);
        Ok(())
    }
    fn destroy_render_buffer(&mut self, id: RenderbufferId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "destroy-render-buffer";
        self.assert_ready(OP)?;
        let (name, _) = self.renderbuffer(OP, id)?;
        // SAFETY: current-context contract; liveness was checked before mutation.
        unsafe { self.gl.delete_renderbuffer(name) };
        self.driver_error(OP)?;
        self.renderbuffers.remove(&id);
        Ok(())
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl crate::backend::gl::api::GlSamplerApi for NativeGlProvider {
    fn create_sampler(
        &mut self,
        desc: crate::backend::gl::api::GlSamplerDesc,
    ) -> Result<SamplerId, GlError> {
        use glow::HasContext as _;
        self.assert_ready("create-sampler")?;
        desc.validate_for(&self.discovery)
            .map_err(|_| Self::validation("create-sampler", "invalid sampler descriptor"))?;
        // SAFETY: current-context contract; descriptor was fully preflighted.
        let name = unsafe { self.gl.create_sampler() }.map_err(|message| GlError::Driver {
            operation: "create-sampler",
            message,
        })?;
        // SAFETY: `name` belongs to this current context. Every parameter comes
        // from a validated closed enum/value, and the unpublished name is
        // deleted on the driver-error path below.
        unsafe {
            self.gl
                .sampler_parameter_i32(name, 0x2802, native_wrap(desc.address_mode_u));
            self.gl
                .sampler_parameter_i32(name, 0x2803, native_wrap(desc.address_mode_v));
            self.gl
                .sampler_parameter_i32(name, 0x8072, native_wrap(desc.address_mode_w));
            self.gl
                .sampler_parameter_i32(name, 0x2800, native_mag(desc.mag_filter));
            self.gl.sampler_parameter_i32(
                name,
                0x2801,
                native_min(desc.min_filter, desc.mipmap_filter),
            );
            self.gl
                .sampler_parameter_f32(name, 0x813A, f32::from_bits(desc.lod_min_bits));
            self.gl
                .sampler_parameter_f32(name, 0x813B, f32::from_bits(desc.lod_max_bits));
            if let Some(compare) = desc.compare {
                self.gl.sampler_parameter_i32(name, 0x884C, 0x884E);
                self.gl
                    .sampler_parameter_i32(name, 0x884D, native_compare(compare));
            }
            if let Some(anisotropy) = desc.max_anisotropy_bits {
                self.gl
                    .sampler_parameter_f32(name, 0x84FE, f32::from_bits(anisotropy));
            }
        }
        if let Err(error) = self.driver_error("create-sampler") {
            // SAFETY: rollback deletes the unpublished sampler allocated by
            // this same current context.
            unsafe { self.gl.delete_sampler(name) };
            return Err(error);
        }
        let id = SamplerId::new(self.context_stamp(), self.slot("create-sampler")?, 0);
        self.samplers.insert(
            id,
            super::sampler_border::NativeSamplerSet {
                float: name,
                sint: None,
                uint: None,
            },
        );
        Ok(id)
    }
    fn destroy_sampler(&mut self, id: SamplerId) -> Result<(), GlError> {
        use glow::HasContext as _;
        self.assert_ready("destroy-sampler")?;
        self.sampler("destroy-sampler", id)?;
        self.texture_unit_samplers.retain(|_, bound| *bound != id);
        let names = self.samplers.remove(&id).expect("validated sampler");
        // SAFETY: current-context contract; liveness was checked before mutation.
        unsafe {
            self.gl.delete_sampler(names.float);
            if let Some(name) = names.sint {
                self.gl.delete_sampler(name);
            }
            if let Some(name) = names.uint {
                self.gl.delete_sampler(name);
            }
        };
        self.driver_error("destroy-sampler")?;
        Ok(())
    }
}

/// The storage class one texture allocation must be created through.
///
/// The two classes are distinct GL targets with distinct lifetime rules, not
/// two settings of one target: multisample storage has a single level, cannot
/// be sampled, and cannot be allocated immutably through the single-sample
/// entry point.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TextureStorageClass {
    /// Immutable single-sample storage, mip chain included.
    SingleSample,
    /// Immutable multisample storage: one level, no sampler access.
    Multisample,
}

/// Decides how one texture descriptor may be allocated on this context.
///
/// The decision is one function rather than a check inside each allocation
/// branch because the same recorded facts decide both branches: whether an
/// exact fact exists at the descriptor's own sample count, and whether the
/// descriptor's declared usage is one the fact actually supports. Keeping the
/// usage check here is also what keeps this provider from accepting storage the
/// recorder already refuses -- a descriptor that asks for sampled usage on
/// storage no sampler can read is rejected before an object exists, instead of
/// at the first bind with a GL error that names a different operation.
///
/// The failure is a reason string rather than a `GlError` so the same decision
/// can be exercised without a live context; the caller attaches the operation.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(super) fn texture_storage_class(
    profile: crate::backend::gl::api::GlFamilyProfile,
    _limits: &crate::backend::gl::api::GlLimits,
    formats: &crate::backend::gl::api::GlFormatTable,
    desc: GlTextureDesc,
) -> Result<TextureStorageClass, &'static str> {
    // Array and volume textures share format facts with D2, but use distinct
    // targets and immutable 3D storage. Desktop GL's core 1D target is also
    // admitted, while embedded profiles keep their 2D/array/3D contract.
    let desktop_1d = desc.dimension == crate::backend::gl::api::GlTextureDimension::D1
        && matches!(
            profile,
            crate::backend::gl::api::GlFamilyProfile::Desktop { .. }
        );
    if !desktop_1d
        && !matches!(
            desc.dimension,
            crate::backend::gl::api::GlTextureDimension::D2
                | crate::backend::gl::api::GlTextureDimension::D2Array
                | crate::backend::gl::api::GlTextureDimension::D3
        )
    {
        return Err("native texture storage covers 2D, 2D-array and 3D targets only");
    }
    if desktop_1d && (desc.extent.height != 1 || desc.extent.depth_or_layers != 1) {
        return Err("native 1D texture extent must have height and depth one");
    }
    if desc.dimension == api::GlTextureDimension::D3
        && (desc
            .format
            .supports_blit_aspect(api::GlTextureAspect::DepthOnly)
            || desc
                .format
                .supports_blit_aspect(api::GlTextureAspect::StencilOnly))
    {
        return Err("GL depth/stencil formats do not permit 3D texture storage");
    }
    let kind = if desc.sample_count > 1
        && desc.dimension == crate::backend::gl::api::GlTextureDimension::D2Array
    {
        crate::backend::gl::api::GlFormatResourceKind::MultisampleArray
    } else {
        crate::backend::gl::api::GlFormatResourceKind::Texture
    };
    let facts = formats
        .get_for(kind, desc.format, desc.sample_count)
        .ok_or("format has no discovery evidence at this sample count")?;
    if desc
        .usage
        .contains(crate::backend::gl::api::GlTextureUsage::SAMPLED)
        && !facts.sampled
    {
        return Err("format is not sampled at this sample count");
    }
    if desc
        .usage
        .contains(crate::backend::gl::api::GlTextureUsage::RENDER_ATTACHMENT)
        && !facts.renderable
    {
        return Err("format is not renderable at this sample count");
    }
    if desc
        .usage
        .contains(crate::backend::gl::api::GlTextureUsage::COPY_SOURCE)
        && !facts.copy_source
        // Exact native facts can admit a matching-sample direct image copy;
        // they never imply pixel readback. A renderable multisample
        // color attachment is also admitted as a narrow resolve source, which
        // immediately reduces it to one sample before regular readback.
        && !(desc.sample_count > 1
            && desc
                .usage
                .contains(crate::backend::gl::api::GlTextureUsage::RENDER_ATTACHMENT)
            && desc
                .format
                .supports_blit_aspect(crate::backend::gl::api::GlTextureAspect::Color)
            && facts.renderable)
    {
        return Err("format is not a copy source at this sample count");
    }
    if desc
        .usage
        .contains(crate::backend::gl::api::GlTextureUsage::COPY_DESTINATION)
        && !facts.copy_destination
    {
        return Err("format is not a copy destination at this sample count");
    }
    if desc
        .usage
        .contains(crate::backend::gl::api::GlTextureUsage::STORAGE_BINDING)
        && !facts.storage_read
        && !facts.storage_write
    {
        return Err("format has no discovered image access at this sample count");
    }
    if desc.sample_count <= 1 {
        return Ok(TextureStorageClass::SingleSample);
    }
    if !matches!(
        desc.dimension,
        crate::backend::gl::api::GlTextureDimension::D2
            | crate::backend::gl::api::GlTextureDimension::D2Array
    ) {
        return Err("multisample texture storage requires a 2D or 2D-array target");
    }
    if !super::discovery::supports_multisample_texture_storage(profile) {
        return Err("this context has no multisample texture storage");
    }
    // The exact per-format texture fact above governs multisample texture
    // allocation. GL_MAX_SAMPLES describes renderbuffers and can differ from
    // the texture-specific sample limits.
    Ok(TextureStorageClass::Multisample)
}

/// Validates the allocation dimensions against the immutable discovery
/// snapshot before a native entry point sees them.
///
/// `texture_support` applies the same ceiling when it publishes Core support.
/// Creation repeats it because callers may create directly after a prior
/// query, and a changed or malformed descriptor must not turn that preflight
/// contract into a driver-dependent allocation failure.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
fn validate_texture_limits(
    limits: &crate::backend::gl::api::GlLimits,
    desc: GlTextureDesc,
) -> Result<(), &'static str> {
    let is_1d = desc.dimension == crate::backend::gl::api::GlTextureDimension::D1;
    let is_3d = desc.dimension == crate::backend::gl::api::GlTextureDimension::D3;
    let xy_limit = if is_3d {
        limits.max_3d_texture_size
    } else {
        limits.max_texture_size
    };
    let z_limit = if is_3d {
        limits.max_3d_texture_size
    } else {
        limits.max_array_texture_layers
    };
    if desc.extent.width > xy_limit
        || (!is_1d && desc.extent.height > xy_limit)
        || desc.extent.depth_or_layers > z_limit
        || desc.extent.width > i32::MAX as u32
        || desc.extent.height > i32::MAX as u32
        || desc.extent.depth_or_layers > i32::MAX as u32
        || desc.mip_level_count > i32::MAX as u32
    {
        return Err("texture size or sample count exceeds discovery limits");
    }
    let largest = desc
        .extent
        .width
        .max(if is_1d { 1 } else { desc.extent.height })
        .max(if is_3d {
            desc.extent.depth_or_layers
        } else {
            1
        });
    let max_mips = u32::BITS - largest.leading_zeros();
    if desc.mip_level_count > max_mips || (desc.sample_count > 1 && desc.mip_level_count != 1) {
        return Err("texture mip count exceeds discovery limits");
    }
    Ok(())
}

/// Returns the texture target admitted by the single-sample allocation path.
///
/// The caller has already passed [`texture_storage_class`], which is why this
/// helper returns an option rather than silently mapping other dimensions.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
const fn single_sample_texture_target(
    dimension: crate::backend::gl::api::GlTextureDimension,
) -> Option<u32> {
    match dimension {
        crate::backend::gl::api::GlTextureDimension::D1 => Some(glow::TEXTURE_1D),
        crate::backend::gl::api::GlTextureDimension::D2 => Some(glow::TEXTURE_2D),
        crate::backend::gl::api::GlTextureDimension::D2Array => Some(glow::TEXTURE_2D_ARRAY),
        crate::backend::gl::api::GlTextureDimension::D3 => Some(glow::TEXTURE_3D),
        _ => None,
    }
}

/// Allocation usage applied to every native buffer (audit P2-11).
///
/// The policy is one explicit, auditable constant per family: `STATIC_DRAW`
/// matches the native desktop residency model where the 0.14 cache re-uploads
/// through `bufferSubData` while the driver is free to place the store in
/// device-local memory. A `DYNAMIC_DRAW` fast path may only be introduced
/// after profiling attributes a benefit (plan "Private: upload-ring/orphaning
/// strategy").
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]

/// Transfer tuple used only to define mutable storage on pre-4.2 desktop GL.
fn mutable_texture_format(format: api::GlFormat) -> Option<(u32, u32)> {
    use api::GlFormat as F;
    if let Some(encoding) = super::exec_copy::upload_encoding(format) {
        return Some(encoding);
    }
    Some(match format {
        F::Rgba8Unorm | F::Rgba8Srgb => (glow::RGBA, glow::UNSIGNED_BYTE),
        F::Rgba8Uint => (glow::RGBA_INTEGER, glow::UNSIGNED_BYTE),
        F::Rgba8Sint => (glow::RGBA_INTEGER, glow::BYTE),
        F::Rgba16Float | F::Rgba32Float => (glow::RGBA, glow::FLOAT),
        F::Depth16Unorm | F::Depth24Unorm | F::Depth32Float => (glow::DEPTH_COMPONENT, glow::FLOAT),
        F::Depth24PlusStencil8 => (glow::DEPTH_STENCIL, glow::UNSIGNED_INT_24_8),
        F::Depth32FloatStencil8 => (glow::DEPTH_STENCIL, glow::FLOAT_32_UNSIGNED_INT_24_8_REV),
        F::Stencil8 => (glow::STENCIL_INDEX, glow::UNSIGNED_BYTE),
        _ => return None,
    })
}
