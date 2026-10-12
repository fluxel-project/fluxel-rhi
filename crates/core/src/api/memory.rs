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

pub struct HeapDesc {
    pub size: u64,
    pub native_memory_class: u32,
}
#[derive(Clone, Copy, Debug)]
pub struct MemoryRequirements {
    pub size: u64,
    pub alignment: u64,
    pub native_type_bits: u64,
    pub dedicated_only: bool,
}
pub trait MemoryBackend: ResourceBackend {
    fn buffer_requirements(
        device: &Self::NativeDevice,
        desc: &BufferDesc,
    ) -> Result<MemoryRequirements>;
    fn texture_requirements(
        device: &Self::NativeDevice,
        desc: &TextureDesc,
    ) -> Result<MemoryRequirements>;
    type NativeHeap: Clone;
    fn create_heap(device: &Self::NativeDevice, desc: &HeapDesc) -> Result<Self::NativeHeap>;
    fn place_buffer(
        device: &Self::NativeDevice,
        heap: &Self::NativeHeap,
        offset: u64,
        desc: &BufferDesc,
    ) -> Result<Self::NativeBuffer>;
    fn place_texture(
        device: &Self::NativeDevice,
        heap: &Self::NativeHeap,
        offset: u64,
        desc: &TextureDesc,
    ) -> Result<Self::NativeTexture>;
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
        Ok(unsafe {
            Texture::from_native(
                self.device.id,
                B::place_texture(&self.device.native, &heap.native, offset, desc)?,
            )
        })
    }
}
