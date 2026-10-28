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

impl<B: ResourceBackend> Device<B> {
    pub fn buffer_support(&self, desc: &BufferDesc) -> Support<BufferSupport> {
        B::buffer_support(&self.native, desc)
    }
    pub fn texture_support(&self, desc: &TextureDesc) -> Support<TextureSupport> {
        B::texture_support(&self.native, desc)
    }
    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<Buffer<B>> {
        if let Support::Unsupported(reason) = B::buffer_support(&self.native, desc) {
            return Err(Error::Unsupported(reason));
        }
        // SAFETY: the SPI created the buffer for this device after support validation.
        Ok(unsafe { Buffer::from_native(self.id, B::create_buffer(&self.native, desc)?) })
    }
    pub fn create_texture(&self, desc: &TextureDesc) -> Result<Texture<B>> {
        if let Support::Unsupported(reason) = B::texture_support(&self.native, desc) {
            return Err(Error::Unsupported(reason));
        }
        // SAFETY: the SPI created the texture for this device after support validation.
        Ok(unsafe { Texture::from_native(self.id, B::create_texture(&self.native, desc)?) })
    }
    pub fn create_view(
        &self,
        texture: &Texture<B>,
        desc: &TextureViewDesc,
    ) -> Result<TextureView<B>> {
        if texture.device != self.id {
            return Err(Error::WrongDevice);
        }
        // SAFETY: `texture` was checked to belong to this device and SPI creates its view.
        Ok(unsafe {
            TextureView::from_native(
                self.id,
                B::create_view(&self.native, &texture.native, desc)?,
            )
        })
    }
    pub fn create_sampler(&self, desc: &SamplerDesc) -> Result<Sampler<B>> {
        // SAFETY: the SPI created the sampler for this device.
        Ok(unsafe { Sampler::from_native(self.id, B::create_sampler(&self.native, desc)?) })
    }
    pub fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDesc,
    ) -> Result<BindGroupLayout<B>> {
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
                BindingType::ExternalTexture => false,
            };
            if entry.nonuniform_indexing && !nonuniform_supported {
                return Err(Error::Unsupported("nonuniform binding indexing"));
            }
        }
        // SAFETY: the SPI created the validated layout for this device.
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
    pub fn create_pipeline_layout(
        &self,
        desc: &PipelineLayoutDesc<'_, B>,
    ) -> Result<PipelineLayout<B>> {
        if desc.groups.len() > self.caps.limits.max_bind_groups as usize {
            return Err(Error::InvalidRange);
        }
        if desc.groups.iter().any(|v| v.device != self.id) {
            return Err(Error::WrongDevice);
        }
        if !desc.push_constants.is_empty() && !self.caps.push_constants {
            return Err(Error::Unsupported("push constants"));
        }
        // SAFETY: every referenced layout was checked to belong to this device.
        Ok(unsafe {
            PipelineLayout::from_native(self.id, B::create_pipeline_layout(&self.native, desc)?)
        })
    }
}
