//! Device-owned resource and layout creation.
//!
//! Every successful constructor stamps its result with this `DeviceId` so
//! later recording rejects cross-device objects. Backend allocation remains
//! entirely behind the static SPI; this module performs no lifetime policy.
#![allow(missing_docs, dead_code, unused_variables, unused_imports)]
use super::*;
use std::{
    future::Future,
    marker::PhantomData,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

mod inference;
mod volume_attachment;

/// Data for one complete texture mip and aspect, including each layer or
/// volume slice in that mip.
pub struct TextureInitData<'a> {
    pub layout: TexelCopyBufferLayout,
    pub bytes: &'a [u8],
}

impl<B: ResourceBackend + CommandBackend> Device<B> {
    /// Creates a device buffer and records a full initialization upload at
    /// the current encoder position. The descriptor must include `COPY_DST`
    /// and its size must match the data. The caller submits the encoder and
    /// orders every use of the new buffer after the recorded upload.
    pub fn create_buffer_initialized(
        &self,
        encoder: &mut CommandEncoder<B>,
        desc: &BufferDesc,
        bytes: &[u8],
    ) -> Result<Buffer<B>> {
        if encoder.device != self.id {
            return Err(Error::WrongDevice);
        }
        if desc.memory != MemoryClass::Device || !desc.usage.contains(BufferUsage::COPY_DST) {
            return Err(Error::Unsupported(
                "staged initialization requires a device COPY_DST buffer",
            ));
        }
        if desc.size == 0 || u64::try_from(bytes.len()) != Ok(desc.size) {
            return Err(Error::InvalidRange);
        }
        encoder.ready()?;
        let buffer = self.create_buffer(desc)?;
        encoder.write_buffer(&buffer, 0, bytes)?;
        Ok(buffer)
    }

    /// Creates a texture and records initialization for every mip and aspect.
    /// Data is ordered by increasing mip, with depth before stencil for a
    /// combined format. Each item includes all array layers or volume slices.
    /// The descriptor must include `COPY_DST` and use one sample. The selected
    /// backend must support CPU uploads for the texture and each supplied aspect.
    /// Submit and order texture access after the recorded uploads.
    pub fn create_texture_initialized(
        &self,
        encoder: &mut CommandEncoder<B>,
        desc: &TextureDesc,
        initial_data: &[TextureInitData<'_>],
    ) -> Result<Texture<B>> {
        if encoder.device != self.id {
            return Err(Error::WrongDevice);
        }
        if desc.samples != 1 || !desc.usage.contains(TextureUsage::COPY_DST) {
            return Err(Error::Unsupported(
                "staged initialization requires a single-sample COPY_DST texture",
            ));
        }
        use TextureAspects as A;
        use TextureFormat as F;
        let aspects: &[TextureAspects] = match desc.format {
            F::Depth16Unorm | F::Depth24Plus | F::Depth32Float => &[A::DEPTH],
            F::Depth24PlusStencil8 | F::Depth32FloatStencil8 => &[A::DEPTH, A::STENCIL],
            F::Stencil8 => &[A::STENCIL],
            _ => &[A::COLOR],
        };
        let expected = (desc.mip_levels as usize).checked_mul(aspects.len());
        if desc.mip_levels == 0 || expected != Some(initial_data.len()) {
            return Err(Error::InvalidRange);
        }
        encoder.ready()?;
        let texture = self.create_texture(desc)?;
        for (mip, mip_data) in (0..desc.mip_levels).zip(initial_data.chunks_exact(aspects.len())) {
            let mip_extent = |axis: usize| desc.extent[axis].checked_shr(mip).unwrap_or(0).max(1);
            let depth_or_layers = if desc.dimension == TextureDimension::D3 {
                mip_extent(2)
            } else {
                desc.array_layers
            };
            for (&aspect, data) in aspects.iter().zip(mip_data) {
                encoder.write_texture(
                    TextureRegion {
                        texture: &texture,
                        mip,
                        origin: [0; 3],
                        extent: [mip_extent(0), mip_extent(1), depth_or_layers],
                        aspect,
                    },
                    data.layout,
                    data.bytes,
                )?;
            }
        }
        Ok(texture)
    }
}

impl<B: ResourceBackend> Device<B> {
    /// Checks the index format and byte range of a device-owned buffer view.
    pub fn index_buffer_view_support(
        &self,
        buffer: &Buffer<B>,
        range: Range<u64>,
        format: IndexFormat,
    ) -> Result<()> {
        if buffer.device != self.id {
            return Err(Error::WrongDevice);
        }
        B::index_buffer_view_support(&buffer.native, range, format)
    }

    /// Checks whether this buffer can be viewed with the given storage ABI,
    /// access and byte range before creating or rewriting descriptors.
    pub fn storage_buffer_view_support(
        &self,
        buffer: &Buffer<B>,
        range: Range<u64>,
        view: StorageBufferView,
        access: StorageAccess,
    ) -> Result<()> {
        if buffer.device != self.id {
            return Err(Error::WrongDevice);
        }
        if !self.caps.storage_buffers {
            return Err(Error::Unsupported("storage buffers"));
        }
        if access == StorageAccess::Atomic {
            return Err(Error::InvalidState);
        }
        B::storage_buffer_view_support(&buffer.native, range, view, access)
    }

    pub fn buffer_support(&self, desc: &BufferDesc) -> Support<BufferSupport> {
        if desc.usage.contains(BufferUsage::STORAGE) && !self.caps.storage_buffers {
            return Support::Unsupported("storage buffers");
        }
        B::buffer_support(&self.native, desc)
    }

    /// Qualifies a host-writable upload or host-readable readback allocation
    /// for the requested placement.
    pub fn cpu_visible_buffer_support(
        &self,
        desc: &BufferDesc,
        placement: CpuVisibleBufferPlacement,
    ) -> Support<BufferSupport> {
        let read = desc.usage.contains(BufferUsage::MAP_READ);
        let write = desc.usage.contains(BufferUsage::MAP_WRITE);
        if !matches!(
            (desc.memory, read, write),
            (MemoryClass::Upload, false, true) | (MemoryClass::Readback, true, false)
        ) {
            return Support::Unsupported("CPU-visible buffer mapping usage");
        }
        if placement == CpuVisibleBufferPlacement::GpuUpload {
            if !self.caps.gpu_upload_buffers {
                return Support::Unsupported("GPU upload buffers");
            }
            if desc.memory != MemoryClass::Upload {
                return Support::Unsupported("GPU upload buffers require write mapping");
            }
        }
        if let CpuVisibleBufferPlacement::Custom { cache, pool } = placement {
            if let Support::Unsupported(reason) =
                B::custom_cpu_visible_buffer_support(&self.native, desc.memory, cache, pool)
            {
                return Support::Unsupported(reason);
            }
        }
        self.buffer_support(desc)
    }

    /// Allocates a buffer through the selected CPU-visible placement.
    pub fn create_cpu_visible_buffer(
        &self,
        desc: &BufferDesc,
        placement: CpuVisibleBufferPlacement,
    ) -> Result<Buffer<B>> {
        B::check_resource_device(&self.native)?;
        if let Support::Unsupported(reason) = self.cpu_visible_buffer_support(desc, placement) {
            return Err(Error::Unsupported(reason));
        }
        let native = match placement {
            CpuVisibleBufferPlacement::Standard => B::create_buffer(&self.native, desc)?,
            CpuVisibleBufferPlacement::GpuUpload => {
                B::create_gpu_upload_buffer(&self.native, desc)?
            }
            CpuVisibleBufferPlacement::Custom { cache, pool } => {
                B::create_custom_cpu_visible_buffer(&self.native, desc, cache, pool)?
            }
        };
        // SAFETY: the SPI allocated the buffer for this exact device and descriptor.
        Ok(unsafe { Buffer::from_native(self.id, native) })
    }
    pub fn texture_support(&self, desc: &TextureDesc) -> Support<TextureSupport> {
        if validate_transient_texture_desc(desc).is_err() {
            return Support::Unsupported("invalid transient attachment descriptor");
        }
        if desc.usage.contains(TextureUsage::STORAGE_BINDING) && !self.caps.storage_textures {
            return Support::Unsupported("storage textures");
        }
        let limits = self.caps.limits;
        let (dimension_limit, axes) = match desc.dimension {
            TextureDimension::D1 => (limits.max_texture_dimension_1d, 1),
            TextureDimension::D2 => (limits.max_texture_dimension_2d, 2),
            TextureDimension::D3 => (limits.max_texture_dimension_3d, 3),
        };
        if dimension_limit != 0
            && desc.extent[..axes]
                .iter()
                .any(|&axis| axis > dimension_limit)
        {
            return Support::Unsupported("texture exceeds the device dimension limit");
        }
        if desc.dimension != TextureDimension::D3
            && limits.max_texture_array_layers != 0
            && desc.array_layers > limits.max_texture_array_layers
        {
            return Support::Unsupported("texture exceeds the device array layer limit");
        }
        B::texture_support(&self.native, desc)
    }
    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<Buffer<B>> {
        B::check_resource_device(&self.native)?;
        if let Support::Unsupported(reason) = self.buffer_support(desc) {
            return Err(Error::Unsupported(reason));
        }
        // SAFETY: the SPI created the buffer for this device after support validation.
        Ok(unsafe { Buffer::from_native(self.id, B::create_buffer(&self.native, desc)?) })
    }
    pub fn create_texture(&self, desc: &TextureDesc) -> Result<Texture<B>> {
        validate_transient_texture_desc(desc)?;
        B::check_resource_device(&self.native)?;
        if let Support::Unsupported(reason) = self.texture_support(desc) {
            return Err(Error::Unsupported(reason));
        }
        // SAFETY: the SPI created the texture for this device after support validation.
        let mut texture =
            unsafe { Texture::from_native(self.id, B::create_texture(&self.native, desc)?) };
        texture.desc = Some(desc.clone());
        texture.native_transient_usage =
            B::texture_uses_native_transient_attachment(&texture.native);
        Ok(texture)
    }
    fn validate_view_request(&self, texture: &Texture<B>, desc: &TextureViewDesc) -> Result<()> {
        if texture.device != self.id {
            return Err(Error::WrongDevice);
        }
        if let Some(texture_desc) = texture
            .desc
            .as_ref()
            .filter(|texture| texture.usage.contains(TextureUsage::TRANSIENT_ATTACHMENT))
        {
            if desc.dimension != TextureViewDimension::D2
                || desc.format != texture_desc.format
                || desc.mips != (0..1)
                || desc.layers != (0..1)
            {
                return Err(Error::InvalidState);
            }
        }
        Ok(())
    }
    /// Checks whether the view can be created on the given texture without
    /// allocating descriptors. The backend may omit the query.
    pub fn texture_view_support(&self, texture: &Texture<B>, desc: &TextureViewDesc) -> Result<()> {
        self.validate_view_request(texture, desc)?;
        B::texture_view_support(&self.native, &texture.native, desc)
    }
    pub fn create_view(
        &self,
        texture: &Texture<B>,
        desc: &TextureViewDesc,
    ) -> Result<TextureView<B>> {
        self.validate_view_request(texture, desc)?;
        let native = B::create_view(&self.native, &texture.native, desc)?;
        Ok(self.wrap_view(texture, native))
    }
    /// Checks admission for a view restricted to the selected descriptor usage.
    pub fn texture_view_usage_support(
        &self,
        texture: &Texture<B>,
        desc: &TextureViewDesc,
        usage: TextureUsage,
    ) -> Result<()> {
        self.validate_explicit_view_usage(texture, desc, usage)?;
        B::texture_view_usage_support(&self.native, &texture.native, desc, usage)
    }
    /// Creates only the selected sampled, storage, or attachment descriptors.
    pub fn create_texture_view_with_usage(
        &self,
        texture: &Texture<B>,
        desc: &TextureViewDesc,
        usage: TextureUsage,
    ) -> Result<TextureView<B>> {
        self.validate_explicit_view_usage(texture, desc, usage)?;
        let native = B::create_texture_view_with_usage(&self.native, &texture.native, desc, usage)?;
        let mut view = self.wrap_view(texture, native);
        view.usage = Some(usage);
        Ok(view)
    }
    fn validate_explicit_view_usage(
        &self,
        texture: &Texture<B>,
        desc: &TextureViewDesc,
        usage: TextureUsage,
    ) -> Result<()> {
        self.validate_view_request(texture, desc)?;
        if !self.caps.texture_view_usage {
            return Err(Error::Unsupported("explicit texture view usage"));
        }
        let descriptor_uses = TextureUsage::TEXTURE_BINDING
            | TextureUsage::STORAGE_BINDING
            | TextureUsage::RENDER_ATTACHMENT;
        if usage.0 == 0
            || !descriptor_uses.contains(usage)
            || texture
                .desc
                .as_ref()
                .is_some_and(|desc| !desc.usage.contains(usage))
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }
    /// Checks admission for a sampled-only color view with a component mapping.
    pub fn sampled_texture_view_support(
        &self,
        texture: &Texture<B>,
        desc: &TextureViewDesc,
        mapping: TextureComponentMapping,
    ) -> Result<()> {
        self.validate_sampled_texture_view_request(texture, desc)?;
        B::sampled_texture_view_support(&self.native, &texture.native, desc, mapping)
    }
    /// Creates a color view usable for sampled bindings only. Storage and
    /// attachment uses require a separate view, including with identity mapping.
    pub fn create_sampled_texture_view(
        &self,
        texture: &Texture<B>,
        desc: &TextureViewDesc,
        mapping: TextureComponentMapping,
    ) -> Result<TextureView<B>> {
        self.validate_sampled_texture_view_request(texture, desc)?;
        let native = B::create_sampled_texture_view(&self.native, &texture.native, desc, mapping)?;
        let mut view = self.wrap_view(texture, native);
        view.usage = Some(TextureUsage::TEXTURE_BINDING);
        Ok(view)
    }
    fn validate_sampled_texture_view_request(
        &self,
        texture: &Texture<B>,
        desc: &TextureViewDesc,
    ) -> Result<()> {
        self.validate_view_request(texture, desc)?;
        if !self.caps.sampled_texture_component_mapping {
            return Err(Error::Unsupported("sampled texture component mapping"));
        }
        if desc.aspects != TextureAspects::COLOR {
            return Err(Error::Unsupported(
                "sampled component mapping requires a color view",
            ));
        }
        Ok(())
    }
    fn wrap_view(&self, texture: &Texture<B>, native: B::NativeTextureView) -> TextureView<B> {
        // SAFETY: `texture` was checked to belong to this device and SPI creates its view.
        let mut view = unsafe { TextureView::from_native(self.id, native) };
        view.texture_desc = texture.desc.clone();
        view
    }
    /// Checks descriptor admission without creating a sampler. Backends may
    /// support creation even when they do not expose a separate support query.
    pub fn sampler_support(&self, desc: &SamplerDesc) -> Result<()> {
        B::sampler_support(&self.native, desc)
    }
    pub fn create_sampler(&self, desc: &SamplerDesc) -> Result<Sampler<B>> {
        // SAFETY: the SPI created the sampler for this device.
        Ok(unsafe { Sampler::from_native(self.id, B::create_sampler(&self.native, desc)?) })
    }
    fn validate_bind_group_layout(&self, desc: &BindGroupLayoutDesc) -> Result<()> {
        if desc.entries.len() > self.caps.limits.max_bindings_per_group as usize {
            return Err(Error::InvalidRange);
        }
        let mut last = None;
        for (index, entry) in desc.entries.iter().enumerate() {
            if entry.binding >= self.caps.limits.max_bindings_per_group {
                return Err(Error::InvalidRange);
            }
            if last.is_some_and(|previous| entry.binding <= previous) {
                return Err(Error::InvalidState);
            }
            last = Some(entry.binding);
            if let BindingType::UniformBuffer { min_size, .. } = entry.ty {
                let limit = self.caps.limits.max_uniform_buffer_binding_size;
                if limit != 0 && min_size > limit {
                    return Err(Error::InvalidRange);
                }
            }
            if matches!(entry.ty, BindingType::VideoFrame)
                && (entry.count != BindingCount::One
                    || entry.partially_bound
                    || entry.nonuniform_indexing
                    || entry.visibility.as_slice() != [ShaderStage::Fragment])
            {
                return Err(Error::InvalidState);
            }
            match entry.count {
                BindingCount::One => {}
                BindingCount::Fixed(0) => return Err(Error::InvalidRange),
                BindingCount::Fixed(_) if !self.caps.binding_arrays => {
                    return Err(Error::Unsupported("binding arrays"));
                }
                BindingCount::Fixed(_) => {}
                BindingCount::RuntimeSized { max_count: 0 } => return Err(Error::InvalidRange),
                BindingCount::RuntimeSized { .. } if !self.caps.runtime_sized_binding_arrays => {
                    return Err(Error::Unsupported("runtime-sized binding arrays"));
                }
                BindingCount::RuntimeSized { .. } if index + 1 != desc.entries.len() => {
                    return Err(Error::InvalidState);
                }
                BindingCount::RuntimeSized { .. } => {}
            }
            if matches!(&entry.ty, BindingType::AccelerationStructure) {
                if !self.caps.acceleration_structure {
                    return Err(Error::Unsupported("acceleration structure binding"));
                }
                match entry.count {
                    BindingCount::One => {}
                    BindingCount::Fixed(_) if !self.caps.acceleration_structure_binding_arrays => {
                        return Err(Error::Unsupported("acceleration structure binding arrays"));
                    }
                    BindingCount::RuntimeSized { .. }
                        if !self
                            .caps
                            .runtime_sized_acceleration_structure_binding_arrays =>
                    {
                        return Err(Error::Unsupported(
                            "runtime-sized acceleration structure arrays",
                        ));
                    }
                    _ => {}
                }
                if entry.partially_bound
                    && !self
                        .caps
                        .partially_bound_acceleration_structure_binding_arrays
                {
                    return Err(Error::Unsupported(
                        "partially-bound acceleration structure arrays",
                    ));
                }
            }
            match &entry.ty {
                BindingType::StorageBuffer { .. } if !self.caps.storage_buffers => {
                    return Err(Error::Unsupported("storage buffers"));
                }
                BindingType::StorageTexture { .. } if !self.caps.storage_textures => {
                    return Err(Error::Unsupported("storage textures"));
                }
                BindingType::StorageBuffer {
                    access: StorageAccess::Atomic,
                    ..
                } => return Err(Error::InvalidState),
                BindingType::StorageTexture {
                    format,
                    access: StorageAccess::Atomic,
                    ..
                } if *format == TextureFormat::R32Uint
                    && (!self.caps.texture_atomic_r32uint_compute
                        && !self.caps.texture_atomic_r32uint_min_compute
                        && !self.caps.texture_atomic_r32uint_max_compute
                        && !self.caps.texture_atomic_r32uint_xor_compute
                        && !self.caps.texture_atomic_r32uint_and_compute
                        && !self.caps.texture_atomic_r32uint_or_compute
                        || entry.visibility.as_slice() != [ShaderStage::Compute]) =>
                {
                    return Err(Error::Unsupported("R32Uint compute texture atomic binding"));
                }
                BindingType::StorageTexture {
                    format: TextureFormat::R32Uint,
                    access: StorageAccess::Atomic,
                    ..
                } => {}
                BindingType::StorageTexture {
                    format,
                    access: StorageAccess::Atomic,
                    ..
                } if *format == TextureFormat::R32Sint
                    && (!self.caps.texture_atomic_r32sint_compute
                        && !self.caps.texture_atomic_r32sint_min_compute
                        && !self.caps.texture_atomic_r32sint_max_compute
                        || entry.visibility.as_slice() != [ShaderStage::Compute]) =>
                {
                    return Err(Error::Unsupported("R32Sint compute texture atomic binding"));
                }
                BindingType::StorageTexture {
                    format: TextureFormat::R32Sint,
                    access: StorageAccess::Atomic,
                    ..
                } => {}
                BindingType::StorageTexture {
                    format: TextureFormat::R64Uint,
                    dimension,
                    access: StorageAccess::Atomic,
                    ..
                } if !self.caps.shader_int64_atomic_image_r64uint_add
                    && !self.caps.shader_int64_atomic_image_r64uint_umin
                    && !self.caps.shader_int64_atomic_image_r64uint_umax
                    || *dimension != TextureViewDimension::D2
                    || entry.visibility.as_slice() != [ShaderStage::Compute] =>
                {
                    return Err(Error::Unsupported("R64Uint compute image atomic binding"));
                }
                BindingType::StorageTexture {
                    format: TextureFormat::R64Uint,
                    dimension: TextureViewDimension::D2,
                    access: StorageAccess::Atomic,
                    ..
                } => {}
                BindingType::StorageTexture {
                    access: StorageAccess::Atomic,
                    ..
                } => return Err(Error::Unsupported("texture atomic binding")),
                _ => {}
            }
            let nonuniform_supported = match &entry.ty {
                BindingType::UniformBuffer { .. } => self.caps.nonuniform_uniform_buffer_indexing,
                BindingType::StorageBuffer { .. } => self.caps.nonuniform_storage_buffer_indexing,
                BindingType::SampledTexture { .. } => self.caps.nonuniform_sampled_texture_indexing,
                BindingType::StorageTexture { .. } => self.caps.nonuniform_storage_texture_indexing,
                BindingType::Sampler { .. } => self.caps.nonuniform_sampler_indexing,
                BindingType::AccelerationStructure => {
                    self.caps.nonuniform_acceleration_structure_indexing
                }
                BindingType::VideoFrame => false,
            };
            if entry.nonuniform_indexing && !nonuniform_supported {
                return Err(Error::Unsupported("nonuniform binding indexing"));
            }
        }
        Ok(())
    }
    /// Checks layout admission without creating a backend layout object.
    pub fn bind_group_layout_support(&self, desc: &BindGroupLayoutDesc) -> Result<()> {
        self.validate_bind_group_layout(desc)?;
        B::bind_group_layout_support(&self.native, desc)
    }
    pub fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDesc,
    ) -> Result<BindGroupLayout<B>> {
        self.validate_bind_group_layout(desc)?;
        // SAFETY: the SPI creates the validated layout for this device.
        Ok(unsafe {
            BindGroupLayout::from_native(
                self.id,
                B::create_bind_group_layout(&self.native, desc)?,
                Arc::from(desc.entries.clone()),
            )
        })
    }
    pub fn create_bind_group(&self, desc: &BindGroupDesc<'_, B>) -> Result<BindGroup<B>> {
        if desc.layout.device != self.id {
            return Err(Error::WrongDevice);
        }
        if desc.entries.len() != desc.layout.entries.len() {
            return Err(Error::InvalidState);
        }
        for (layout, entry) in desc.layout.entries.iter().zip(desc.entries) {
            if layout.binding != entry.binding || !binding_matches(&layout.ty, &entry.resource) {
                return Err(Error::InvalidState);
            }
            // Check shape first, so provenance never traverses an arbitrarily
            // nested descriptor tree. All checks precede native allocation.
            if !binding_resource_device(&entry.resource, self.id) {
                return Err(Error::WrongDevice);
            }
            if binding_resource_transient(&entry.resource) {
                return Err(Error::InvalidState);
            }
            let uniform_limit = self.caps.limits.max_uniform_buffer_binding_size;
            if uniform_limit != 0 && matches!(layout.ty, BindingType::UniformBuffer { .. }) {
                let oversized = |resource: &BindingResource<'_, B>| {
                    matches!(resource, BindingResource::Buffer { range, .. }
                        if range.end.saturating_sub(range.start) > uniform_limit)
                };
                let oversized = match &entry.resource {
                    BindingResource::Array(values) => values.iter().flatten().any(oversized),
                    resource => oversized(resource),
                };
                if oversized {
                    return Err(Error::InvalidRange);
                }
            }
            let array = match &entry.resource {
                BindingResource::Array(values) => Some(values),
                _ => None,
            };
            match layout.count {
                BindingCount::One if array.is_some() => return Err(Error::InvalidState),
                BindingCount::Fixed(count) => {
                    let values = array.ok_or(Error::InvalidState)?;
                    if values.len() != count as usize
                        || (!layout.partially_bound && values.iter().any(Option::is_none))
                    {
                        return Err(Error::InvalidState);
                    }
                }
                BindingCount::RuntimeSized { max_count } => {
                    let values = array.ok_or(Error::InvalidState)?;
                    if values.len() > max_count as usize {
                        return Err(Error::InvalidRange);
                    }
                    if values.is_empty()
                        || (!layout.partially_bound && values.iter().any(Option::is_none))
                    {
                        return Err(Error::InvalidState);
                    }
                }
                BindingCount::One => {}
            }
        }
        // SAFETY: descriptor validation established one-device provenance before SPI creation.
        Ok(unsafe { BindGroup::from_native(self.id, B::create_bind_group(&self.native, desc)?) })
    }
    fn validate_pipeline_layout(&self, desc: &PipelineLayoutDesc<'_, B>) -> Result<()> {
        if desc.groups.len() > self.caps.limits.max_bind_groups as usize {
            return Err(Error::InvalidRange);
        }
        if desc.groups.iter().any(|v| v.device != self.id) {
            return Err(Error::WrongDevice);
        }
        let uniform_limit = self
            .caps
            .limits
            .max_dynamic_uniform_buffers_per_pipeline_layout;
        let storage_limit = self
            .caps
            .limits
            .max_dynamic_storage_buffers_per_pipeline_layout;
        if uniform_limit != 0 || storage_limit != 0 {
            let mut uniforms = 0usize;
            let mut storage = 0usize;
            for entry in desc.groups.iter().flat_map(|group| group.entries.iter()) {
                match entry.ty {
                    BindingType::UniformBuffer {
                        dynamic_offset: true,
                        ..
                    } => uniforms += 1,
                    BindingType::StorageBuffer {
                        dynamic_offset: true,
                        ..
                    } => storage += 1,
                    _ => {}
                }
            }
            if (uniform_limit != 0 && uniforms > uniform_limit as usize)
                || (storage_limit != 0 && storage > storage_limit as usize)
            {
                return Err(Error::InvalidRange);
            }
        }
        if !desc.push_constants.is_empty() && !self.caps.push_constants {
            return Err(Error::Unsupported("push constants"));
        }
        let push_constant_limit = self.caps.limits.max_push_constant_bytes;
        if push_constant_limit != 0
            && desc
                .push_constants
                .iter()
                .any(|range| range.range.end > push_constant_limit)
        {
            return Err(Error::InvalidRange);
        }
        Ok(())
    }
    /// Checks the pipeline layout's combined binding and constant budgets
    /// without creating a native pipeline layout.
    pub fn pipeline_layout_support(&self, desc: &PipelineLayoutDesc<'_, B>) -> Result<()> {
        self.validate_pipeline_layout(desc)?;
        B::pipeline_layout_support(&self.native, desc)
    }
    pub fn create_pipeline_layout(
        &self,
        desc: &PipelineLayoutDesc<'_, B>,
    ) -> Result<PipelineLayout<B>> {
        self.validate_pipeline_layout(desc)?;
        // SAFETY: every referenced layout was checked to belong to this device.
        let mut layout = unsafe {
            PipelineLayout::from_native(self.id, B::create_pipeline_layout(&self.native, desc)?)
        };
        layout.description = Some(Arc::new(PipelineLayoutDescription {
            groups: desc.groups.iter().map(|&group| group.clone()).collect(),
            push_constants: desc.push_constants.to_vec(),
        }));
        Ok(layout)
    }
}

fn binding_resource_transient<B: Backend>(resource: &BindingResource<'_, B>) -> bool {
    match resource {
        BindingResource::TextureView(view) => view
            .texture_desc
            .as_ref()
            .is_some_and(|d| d.usage.contains(TextureUsage::TRANSIENT_ATTACHMENT)),
        BindingResource::VideoFrame(_) => false,
        BindingResource::Array(values) => values.iter().flatten().any(binding_resource_transient),
        _ => false,
    }
}
