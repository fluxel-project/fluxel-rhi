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
#[path = "resource/mapping.rs"]
mod mapping;
pub(crate) use mapping::native_texture_format_supported;
pub(super) use mapping::{depth_attachment_point, has_stencil_plane, native_texture_format};
use mapping::{native_compare, native_mag, native_min, native_wrap};

impl crate::backend::gl::api::GlResourceApi for NativeGlProvider {
    fn create_buffer_resource(&mut self, desc: GlBufferDesc) -> Result<BufferId, GlError> {
        use glow::HasContext as _;
        self.assert_ready("create-buffer")?;
        desc.validate().map_err(|error| GlError::Validation {
            operation: "create-buffer",
            message: error.message(),
        })?;
        let size = i32::try_from(desc.size)
            .map_err(|_| Self::validation("create-buffer", "buffer exceeds GLsizei"))?;
        // SAFETY: current-context contract; all validation completed before GL mutation.
        let name = unsafe { self.gl.create_buffer() }.map_err(|message| GlError::Driver {
            operation: "create-buffer",
            message,
        })?;
        // COPY_WRITE_BUFFER is Layer 1-private scratch: bound immediately before
        // the allocation, not restored on return (`GlCopyDomainApi` documents
        // why).
        // SAFETY: see above.
        unsafe {
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(name));
            self.gl
                .buffer_data_size(glow::COPY_WRITE_BUFFER, size, BUFFER_ALLOCATION_USAGE);
        }
        if let Err(error) = self.driver_error("create-buffer") {
            // SAFETY: rollback deletes only the name allocated by this current
            // context above; it was never published to the resource table.
            unsafe { self.gl.delete_buffer(name) };
            return Err(error);
        }
        let id = BufferId::new(self.context_stamp(), self.slot("create-buffer")?, 0);
        self.buffers.insert(id, (name, desc));
        Ok(id)
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
        if desc.format.compressed_info().is_some()
            && desc.dimension != crate::backend::gl::api::GlTextureDimension::D2
        {
            // Compressed definitions use the legacy complete-mip upload route
            // below this allocator. It has no array-layer spelling yet, so
            // accepting array storage here would publish an allocation no
            // command can define correctly.
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "compressed native texture arrays have no upload route",
            });
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
            match class {
                TextureStorageClass::SingleSample => {
                    let target = single_sample_texture_target(desc.dimension)
                        .expect("storage class already admitted this texture dimension");
                    self.gl.bind_texture(target, Some(name));
                    if desc.format.compressed_info().is_none() {
                        match desc.dimension {
                            crate::backend::gl::api::GlTextureDimension::D2 => {
                                self.gl
                                    .tex_storage_2d(target, levels, internal, width, height);
                            }
                            crate::backend::gl::api::GlTextureDimension::D2Array => {
                                self.gl.tex_storage_3d(
                                    target, levels, internal, width, height, layers,
                                );
                            }
                            _ => unreachable!(
                                "texture_storage_class admitted an unsupported dimension"
                            ),
                        }
                    }
                }
                TextureStorageClass::Multisample => {
                    // Sample locations are fixed: a resolve reads every texel's
                    // samples as one block, and the per-texel locations that the
                    // alternative would expose are not expressible in this
                    // layer's vocabulary.
                    self.gl
                        .bind_texture(glow::TEXTURE_2D_MULTISAMPLE, Some(name));
                    self.gl.tex_storage_2d_multisample(
                        glow::TEXTURE_2D_MULTISAMPLE,
                        samples,
                        internal,
                        width,
                        height,
                        true,
                    );
                }
            }
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
        use glow::HasContext as _;
        self.assert_ready("destroy-buffer")?;
        let (name, _) = self.buffer("destroy-buffer", id)?;
        // SAFETY: current-context contract; liveness was checked before GL mutation.
        unsafe { self.gl.delete_buffer(name) };
        self.driver_error("destroy-buffer")?;
        self.buffers.remove(&id);
        Ok(())
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
        self.samplers.insert(id, name);
        Ok(id)
    }
    fn destroy_sampler(&mut self, id: SamplerId) -> Result<(), GlError> {
        use glow::HasContext as _;
        self.assert_ready("destroy-sampler")?;
        let name = self.sampler("destroy-sampler", id)?;
        // SAFETY: current-context contract; liveness was checked before mutation.
        unsafe { self.gl.delete_sampler(name) };
        self.driver_error("destroy-sampler")?;
        self.samplers.remove(&id);
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
    limits: &crate::backend::gl::api::GlLimits,
    formats: &crate::backend::gl::api::GlFormatTable,
    desc: GlTextureDesc,
) -> Result<TextureStorageClass, &'static str> {
    // D2Array uses the same format facts as D2 but a different target and 3D
    // storage entry point. Keep that distinction here so no allocation path
    // can accidentally bind a layered image as TEXTURE_2D. Other dimensions
    // need their own target and view contracts and remain unsupported.
    if !matches!(
        desc.dimension,
        crate::backend::gl::api::GlTextureDimension::D2
            | crate::backend::gl::api::GlTextureDimension::D2Array
    ) {
        return Err("native texture storage covers 2D and 2D-array targets only");
    }
    let facts = formats
        .get_for(
            crate::backend::gl::api::GlFormatResourceKind::Texture,
            desc.format,
            desc.sample_count,
        )
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
    if desc.dimension != crate::backend::gl::api::GlTextureDimension::D2 {
        return Err("multisample texture storage requires a two-dimensional target");
    }
    if !super::discovery::supports_multisample_texture_storage(profile) {
        return Err("this context has no multisample texture storage");
    }
    // The per-class ceiling is not re-applied here: it is what bounded the fact
    // this decision just required, and a snapshot whose fact table exceeded its
    // own recorded ceilings is already rejected when the snapshot is built, so
    // repeating the comparison could never change the outcome. The multisample
    // ceiling is a different bound and is checked: it governs every multisample
    // allocation regardless of format class, and the fact table does not
    // constrain a texture fact by it.
    if desc.sample_count > limits.max_samples {
        return Err("sample count exceeds the recorded multisample ceiling");
    }
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
    if desc.extent.width > limits.max_texture_size
        || desc.extent.height > limits.max_texture_size
        || desc.extent.depth_or_layers > limits.max_array_texture_layers
        || desc.extent.width > i32::MAX as u32
        || desc.extent.height > i32::MAX as u32
        || desc.extent.depth_or_layers > i32::MAX as u32
        || desc.mip_level_count > i32::MAX as u32
        || desc.sample_count > limits.max_samples.max(1)
    {
        return Err("texture size or sample count exceeds discovery limits");
    }
    let max_mips = u32::BITS - desc.extent.width.max(desc.extent.height).leading_zeros();
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
        crate::backend::gl::api::GlTextureDimension::D2 => Some(glow::TEXTURE_2D),
        crate::backend::gl::api::GlTextureDimension::D2Array => Some(glow::TEXTURE_2D_ARRAY),
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
pub(super) const BUFFER_ALLOCATION_USAGE: u32 = glow::STATIC_DRAW;
