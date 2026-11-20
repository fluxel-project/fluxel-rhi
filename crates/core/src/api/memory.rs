//! Explicit memory allocation and placement contracts.
//!
//! Allocation ownership stays with one device. Suballocation strategy and
//! the scheduling of residency changes belong to the caller.
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
/// Budget behavior selected by the caller for native asynchronous residency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidencyBudget {
    AllowOverBudget,
    DenyOverBudget,
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
    /// Qualifies a heap class and size before allocation. The allocation may
    /// still fail due to runtime memory pressure.
    pub fn heap_support(&self, desc: &HeapDesc) -> Result<()> {
        B::heap_support(&self.device.native, desc)
    }

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
    /// Checks whether this heap admits explicit residency operations.
    pub fn heap_residency_support(&self, heap: &MemoryHeap<B>) -> Result<()> {
        if heap.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        if !self.device.caps.explicit_heap_residency {
            return Err(Error::Unsupported("explicit heap residency"));
        }
        B::heap_residency_support(&self.device.native, &heap.native)
    }

    /// Makes the heap resident synchronously and adds one residency reference.
    /// A newly created eligible heap starts with one residency reference.
    pub fn make_heap_resident(&self, heap: &MemoryHeap<B>) -> Result<()> {
        self.heap_residency_support(heap)?;
        B::make_heap_resident(&self.device.native, &heap.native)
    }

    /// Enqueues one residency reference. Observe the returned host completion
    /// before relying on that reference for subsequent GPU use or eviction.
    /// Dropping the completion point does not cancel the native operation.
    pub fn enqueue_heap_residency(
        &self,
        heap: &MemoryHeap<B>,
        budget: ResidencyBudget,
    ) -> Result<CompletionPoint<B>> {
        self.heap_residency_support(heap)?;
        if !self.device.caps.async_heap_residency {
            return Err(Error::Unsupported("asynchronous heap residency"));
        }
        let native = B::enqueue_heap_residency(&self.device.native, &heap.native, budget)?;
        // SAFETY: SPI returns native completion for this device and operation.
        Ok(unsafe { CompletionPoint::from_native(self.device.id, native) })
    }

    /// Releases one residency reference. The final release permits eviction.
    ///
    /// # Safety
    /// The caller must own a residency reference. Before releasing it, all GPU
    /// uses of the heap and resources backed by it must have completed. The
    /// caller must exclude new GPU uses until residency has been reestablished
    /// or another reference guarantees residency through those uses.
    pub unsafe fn evict_heap(&self, heap: &MemoryHeap<B>) -> Result<()> {
        self.heap_residency_support(heap)?;
        // SAFETY: device ownership was checked; the caller supplies ordering.
        unsafe { B::evict_heap(&self.device.native, &heap.native) }
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
        let mut texture = unsafe {
            Texture::from_native(
                self.device.id,
                B::place_texture(&self.device.native, &heap.native, offset, desc)?,
            )
        };
        texture.desc = Some(desc.clone());
        Ok(texture)
    }
}
