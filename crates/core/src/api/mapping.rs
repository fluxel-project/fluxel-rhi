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

// Optional native domains. None of these methods are required from a backend
// that does not expose the corresponding capability. Domain acquisition checks
// the actual opened device once; commands then use B's concrete native types.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapMode {
    Read,
    Write,
}
pub trait MappingBackend: Backend {
    type NativeMap;
    type MapFuture<'a>: Future<Output = Result<Self::NativeMap>> + 'a
    where
        Self: 'a;
    fn map<'a>(
        device: &'a Self::NativeDevice,
        buffer: &'a Self::NativeBuffer,
        range: Range<u64>,
        mode: MapMode,
    ) -> Self::MapFuture<'a>;
    fn bytes(map: &Self::NativeMap) -> &[u8];
    fn bytes_mut(map: &mut Self::NativeMap) -> &mut [u8];
    fn flush(map: &mut Self::NativeMap, bytes: Range<u64>) -> Result<()>;
    fn invalidate(map: &mut Self::NativeMap, bytes: Range<u64>) -> Result<()>;
    fn unmap(map: Self::NativeMap);
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
            B::unmap(map);
        }
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
/// Readback is staged at an explicitly encoded position. RHI does not scan
/// resources or choose completion; renderer must await covering checkpoint.
pub trait ReadbackBackend: CommandBackend + CompletionBackend {
    type NativeReadback;
    type ReadbackFuture<'a>: Future<Output = Result<ReadbackData>> + 'a
    where
        Self: 'a;
    fn encode_readback(
        encoder: &mut Self::NativeEncoder,
        src: &Self::NativeBuffer,
        range: Range<u64>,
    ) -> Result<Self::NativeReadback>;
    fn map_readback<'a>(
        device: &'a Self::NativeDevice,
        ticket: Self::NativeReadback,
    ) -> Self::ReadbackFuture<'a>;
}
pub struct ReadbackTicket<B: ReadbackBackend> {
    device: DeviceId,
    native: B::NativeReadback,
}
impl<B: ReadbackBackend> CommandEncoder<B> {
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
