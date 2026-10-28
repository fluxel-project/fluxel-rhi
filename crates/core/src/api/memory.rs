//! Explicit memory allocation and placement contracts.
//!
//! Allocation ownership stays with one device. Suballocation strategy and
//! eviction are backend or renderer decisions, not Core policy.
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

pub struct HeapDesc {
    pub size: u64,
    /// Backend-specific memory class. For Vulkan this is a physical-device
    /// memory type index selected from the requirement's `native_type_bits`.
    pub native_memory_class: u32,
}
#[derive(Clone, Copy, Debug)]
pub struct MemoryRequirements {
    /// Required heap range size, including any backend placement padding.
    pub size: u64,
    pub alignment: u64,
    /// Supported backend memory classes, encoded as bit positions.
    pub native_type_bits: u64,
    /// The resource requires a dedicated native allocation and cannot be placed.
    pub dedicated_only: bool,
}
pub struct MemoryHeap<B: MemoryBackend> {
    device: DeviceId,
    native: B::NativeHeap,
}
pub struct MemoryDomain<'a, B: MemoryBackend> {
    device: &'a Device<B>,
}
impl<B: MemoryBackend> Device<B> {
    pub fn memory(&self) -> Option<MemoryDomain<'_, B>> {
        self.caps
            .placed_memory
            .then_some(MemoryDomain { device: self })
    }
}
impl<B: MemoryBackend> MemoryDomain<'_, B> {
    pub fn buffer_requirements(&self, desc: &BufferDesc) -> Result<MemoryRequirements> {
        B::buffer_requirements(&self.device.native, desc)
    }
    pub fn texture_requirements(&self, desc: &TextureDesc) -> Result<MemoryRequirements> {
        B::texture_requirements(&self.device.native, desc)
    }
    pub fn heap(&self, desc: &HeapDesc) -> Result<MemoryHeap<B>> {
        Ok(MemoryHeap {
            device: self.device.id,
            native: B::create_heap(&self.device.native, desc)?,
        })
    }
    pub fn place_buffer(
        &self,
        heap: &MemoryHeap<B>,
        offset: u64,
        desc: &BufferDesc,
    ) -> Result<Buffer<B>> {
        if heap.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        // SAFETY: SPI allocation was requested from this device for this validated descriptor.
        Ok(unsafe {
            Buffer::from_native(
                self.device.id,
                B::place_buffer(&self.device.native, &heap.native, offset, desc)?,
            )
        })
    }
    pub fn place_texture(
        &self,
        heap: &MemoryHeap<B>,
        offset: u64,
        desc: &TextureDesc,
    ) -> Result<Texture<B>> {
        if heap.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        if desc.usage.contains(TextureUsage::TRANSIENT_ATTACHMENT) {
            return Err(Error::InvalidState);
        }
        // SAFETY: the backing allocation was checked to belong to this device.
        Ok(unsafe {
            Texture::from_native(
                self.device.id,
                B::place_texture(&self.device.native, &heap.native, offset, desc)?,
            )
        })
    }
}
