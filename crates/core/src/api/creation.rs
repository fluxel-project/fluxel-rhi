//! Statically typed RHI contract component.
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

/// Backend-native constructors and per-device support facts. No hazard tracker.
pub trait ResourceBackend: Backend {
    fn buffer_support(device: &Self::NativeDevice, desc: &BufferDesc) -> Support<BufferSupport>;
    fn texture_support(device: &Self::NativeDevice, desc: &TextureDesc) -> Support<TextureSupport>;
    fn create_buffer(device: &Self::NativeDevice, desc: &BufferDesc) -> Result<Self::NativeBuffer>;
    fn create_texture(
        device: &Self::NativeDevice,
        desc: &TextureDesc,
    ) -> Result<Self::NativeTexture>;
    fn create_view(
        device: &Self::NativeDevice,
        texture: &Self::NativeTexture,
        desc: &TextureViewDesc,
    ) -> Result<Self::NativeTextureView>;
    fn create_sampler(
        device: &Self::NativeDevice,
        desc: &SamplerDesc,
    ) -> Result<Self::NativeSampler>;
    fn create_bind_group_layout(
        device: &Self::NativeDevice,
        desc: &BindGroupLayoutDesc,
    ) -> Result<Self::NativeBindGroupLayout>;
    fn create_bind_group(
        device: &Self::NativeDevice,
        desc: &BindGroupDesc<'_, Self>,
    ) -> Result<Self::NativeBindGroup>;
    fn create_pipeline_layout(
        device: &Self::NativeDevice,
        desc: &PipelineLayoutDesc<'_, Self>,
    ) -> Result<Self::NativePipelineLayout>;
}
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
        Ok(unsafe { Buffer::from_native(self.id, B::create_buffer(&self.native, desc)?) })
    }
    pub fn create_texture(&self, desc: &TextureDesc) -> Result<Texture<B>> {
        if let Support::Unsupported(reason) = B::texture_support(&self.native, desc) {
            return Err(Error::Unsupported(reason));
        }
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
        Ok(unsafe {
            TextureView::from_native(
                self.id,
                B::create_view(&self.native, &texture.native, desc)?,
            )
        })
    }
    pub fn create_sampler(&self, desc: &SamplerDesc) -> Result<Sampler<B>> {
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
                BindingCount::RuntimeSized if !self.caps.runtime_sized_binding_arrays => {
                    return Err(Error::Unsupported("runtime-sized binding arrays"));
                }
                BindingCount::RuntimeSized if index + 1 != desc.entries.len() => {
                    return Err(Error::InvalidState);
                }
                BindingCount::RuntimeSized => {}
            }
            if entry.nonuniform_indexing
                && !self
                    .caps
                    .nonuniform_sampled_texture_and_storage_buffer_indexing
            {
                return Err(Error::Unsupported("nonuniform binding indexing"));
            }
        }
        Ok(unsafe {
            BindGroupLayout::from_native(
                self.id,
                B::create_bind_group_layout(&self.native, desc)?,
                Arc::from(desc.entries.clone()),
            )
        })
    }
    pub fn create_bind_group(&self, desc: &BindGroupDesc<'_, B>) -> Result<BindGroup<B>> {
        if desc.layout.device != self.id
            || !desc
                .entries
                .iter()
                .all(|v| binding_resource_device(&v.resource, self.id))
        {
            return Err(Error::WrongDevice);
        }
        if desc.entries.len() != desc.layout.entries.len() {
            return Err(Error::InvalidState);
        }
        for (layout, entry) in desc.layout.entries.iter().zip(desc.entries) {
            if layout.binding != entry.binding || !binding_matches(&layout.ty, &entry.resource) {
                return Err(Error::InvalidState);
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
                BindingCount::RuntimeSized => {
                    let values = array.ok_or(Error::InvalidState)?;
                    if values.is_empty()
                        || (!layout.partially_bound && values.iter().any(Option::is_none))
                    {
                        return Err(Error::InvalidState);
                    }
                }
                BindingCount::One => {}
            }
        }
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
        Ok(unsafe {
            PipelineLayout::from_native(self.id, B::create_pipeline_layout(&self.native, desc)?)
        })
    }
}
