//! Host mapping and readback contracts.
//!
//! Mapping is valid only after the caller has obtained the required completion
//! evidence. This module exposes no implicit wait or resource-retirement rule.
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

// Optional native domains. None of these methods are required from a backend
// that does not expose the corresponding capability. Domain acquisition checks
// the actual opened device once; commands then use B's concrete native types.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapMode {
    Read,
    Write,
}
pub struct MapLease<B: MappingBackend> {
    device: DeviceId,
    mode: MapMode,
    native: Option<B::NativeMap>,
}
impl<B: MappingBackend> MapLease<B> {
    pub fn bytes(&self) -> &[u8] {
        B::bytes(self.native.as_ref().expect("active mapping"))
    }
    pub fn bytes_mut(&mut self) -> Result<&mut [u8]> {
        if self.mode != MapMode::Write {
            return Err(Error::InvalidState);
        }
        Ok(B::bytes_mut(
            self.native.as_mut().ok_or(Error::InvalidState)?,
        ))
    }
    /// Publishes host modifications in this lease-relative range. Callers must
    /// flush modified bytes before GPU consumers, even on devices whose native
    /// memory happens to be coherent. Dropping a lease does not promise an
    /// implicit flush, and a backend must not overwrite subsequent GPU updates
    /// merely because a previously flushed lease is released.
    pub fn flush(&mut self, bytes: Range<u64>) -> Result<()> {
        if self.mode != MapMode::Write {
            return Err(Error::InvalidState);
        }
        B::flush(self.native.as_mut().ok_or(Error::InvalidState)?, bytes)
    }
    pub fn invalidate(&mut self, bytes: Range<u64>) -> Result<()> {
        if self.mode != MapMode::Read {
            return Err(Error::InvalidState);
        }
        B::invalidate(self.native.as_mut().ok_or(Error::InvalidState)?, bytes)
    }
}
impl<B: MappingBackend> Drop for MapLease<B> {
    fn drop(&mut self) {
        if let Some(map) = self.native.take() {
            // Taking first makes Drop idempotent if an explicit release path
            // is added later; the native mapping still receives one unmap.
            B::unmap(map);
        }
    }
}

/// A newly created buffer held exclusively by its initialization mapping.
/// GPU commands can refer to the buffer once initialization is finished.
pub struct MappedBuffer<B: MappingBackend> {
    mapping: MapLease<B>,
    buffer: Buffer<B>,
}

impl<B: MappingBackend> MappedBuffer<B> {
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        self.mapping
            .bytes_mut()
            .expect("mapped buffer has an active write lease")
    }

    /// Publishes the initialized contents and releases the CPU mapping.
    pub fn finish(self) -> Result<Buffer<B>> {
        let Self {
            mut mapping,
            buffer,
        } = self;
        let size = mapping.bytes_mut()?.len() as u64;
        mapping.flush(0..size)?;
        drop(mapping);
        Ok(buffer)
    }
}

impl<B: MappingBackend + ResourceBackend> Device<B> {
    /// Creates a zero-initialized, CPU-writable mapping over a new buffer.
    /// The descriptor must include MAP_WRITE and use a memory class the
    /// backend supports for host writes. No queue operation is issued.
    pub async fn create_buffer_mapped(&self, desc: &BufferDesc) -> Result<MappedBuffer<B>> {
        if !desc.usage.contains(BufferUsage::MAP_WRITE) {
            return Err(Error::Unsupported("mapped buffer requires MAP_WRITE"));
        }
        let buffer = self.create_buffer(desc)?;
        // SAFETY: the new buffer has never been exposed to the caller and
        // no GPU submission can refer to it before the mapping is released.
        let mut mapping = unsafe {
            self.map_buffer(&buffer, 0..desc.size, MapMode::Write)
                .await?
        };
        mapping.bytes_mut()?.fill(0);
        Ok(MappedBuffer { mapping, buffer })
    }
}

impl<B: MappingBackend> Device<B> {
    /// # Safety
    /// Renderer proves host access does not overlap incompatible GPU access.
    /// Cancellation of a pending map must leave no unreachable native mapping.
    pub async unsafe fn map_buffer(
        &self,
        buffer: &Buffer<B>,
        range: Range<u64>,
        mode: MapMode,
    ) -> Result<MapLease<B>> {
        if buffer.device != self.id {
            return Err(Error::WrongDevice);
        }
        Ok(MapLease {
            device: self.id,
            mode,
            native: Some(B::map(&self.native, &buffer.native, range, mode).await?),
        })
    }
}
pub struct ReadbackData {
    pub bytes: Vec<u8>,
    pub row_pitch: u64,
    pub slice_pitch: u64,
}
pub struct ReadbackTicket<B: ReadbackBackend> {
    device: DeviceId,
    native: B::NativeReadback,
}
impl<B: ReadbackBackend> CommandEncoder<B> {
    /// Stages a copy from a device-local source, or retains a host-readable
    /// source for mapping after completion. A copied source requires COPY_SRC
    /// usage; the caller establishes COPY/COPY_READ before this operation on
    /// backends with explicit resource states. The copy leaves that state in
    /// place. The caller still supplies completion covering this ticket.
    pub fn readback_buffer(
        &mut self,
        src: &Buffer<B>,
        range: Range<u64>,
    ) -> Result<ReadbackTicket<B>> {
        if src.device != self.device {
            return Err(Error::WrongDevice);
        }
        let result = B::encode_readback(self.ready()?, &src.native, range);
        if result.is_err() {
            // Readback encoding may allocate or record a partial native copy,
            // so it follows the encoder's sticky failure discipline.
            self.poisoned = true;
        }
        Ok(ReadbackTicket {
            device: self.device,
            native: result?,
        })
    }
}
impl<B: ReadbackBackend> Device<B> {
    /// # Safety: completion must actually cover the recorded readback copy;
    /// an unrelated CompletionPoint is not sufficient evidence.
    pub async unsafe fn read_after(
        &self,
        ticket: ReadbackTicket<B>,
        completed: &CompletionPoint<B>,
    ) -> Result<ReadbackData> {
        if ticket.device != self.id || completed.device != self.id {
            return Err(Error::WrongDevice);
        }
        B::map_readback(&self.native, ticket.native).await
    }
}
