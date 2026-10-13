//! Explicit import and export contract for externally owned GPU resources.
//!
//! The platform bridge supplies ownership and synchronization evidence. Core
//! does not copy, adopt, or schedule external resources implicitly.
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

pub struct ExternalResourceDesc {
    pub size: u64,
    pub offset: u64,
}
/// Platform bridge chooses this concrete native handle type; RHI never stores
/// it in dyn Any or downcasts it on command paths.
pub trait ExternalBackend: CommandBackend {
    type NativeExternalHandle;
    type NativeExternalMemory: Clone;
    type NativeExternalTexture: Clone;
    type NativeExternalSync;
    fn import_memory(
        device: &Self::NativeDevice,
        handle: Self::NativeExternalHandle,
        desc: &ExternalResourceDesc,
    ) -> Result<Self::NativeExternalMemory>;
    fn import_texture(
        device: &Self::NativeDevice,
        handle: Self::NativeExternalHandle,
        desc: &TextureDesc,
    ) -> Result<Self::NativeExternalTexture>;
    fn acquire_external(
        encoder: &mut Self::NativeEncoder,
        memory: &Self::NativeExternalMemory,
        sync: Self::NativeExternalSync,
    ) -> Result<()>;
    fn release_external(
        encoder: &mut Self::NativeEncoder,
        memory: &Self::NativeExternalMemory,
    ) -> Result<Self::NativeExternalSync>;
}
pub struct ExternalMemory<B: ExternalBackend> {
    device: DeviceId,
    native: B::NativeExternalMemory,
}
pub struct ExternalTexture<B: ExternalBackend> {
    device: DeviceId,
    native: B::NativeExternalTexture,
}
pub struct ExternalSync<B: ExternalBackend> {
    device: DeviceId,
    native: B::NativeExternalSync,
}
pub struct ExternalDomain<'a, B: ExternalBackend> {
    device: &'a Device<B>,
}
impl<B: ExternalBackend> Device<B> {
    pub fn external(&self) -> Option<ExternalDomain<'_, B>> {
        self.caps
            .external_memory
            .then_some(ExternalDomain { device: self })
    }
}
impl<B: ExternalBackend> ExternalDomain<'_, B> {
    /// # Safety
    /// Platform bridge supplies a valid native handle, ownership and sync.
    pub unsafe fn import_memory(
        &self,
        handle: B::NativeExternalHandle,
        desc: &ExternalResourceDesc,
    ) -> Result<ExternalMemory<B>> {
        Ok(ExternalMemory {
            device: self.device.id,
            native: B::import_memory(&self.device.native, handle, desc)?,
        })
    }
    /// # Safety: platform bridge supplies an external native texture and its
    /// ownership/synchronization guarantees; no hidden import copy is allowed.
    pub unsafe fn import_texture(
        &self,
        handle: B::NativeExternalHandle,
        desc: &TextureDesc,
    ) -> Result<ExternalTexture<B>> {
        Ok(ExternalTexture {
            device: self.device.id,
            native: B::import_texture(&self.device.native, handle, desc)?,
        })
    }
    pub fn acquire(
        &self,
        encoder: &mut CommandEncoder<B>,
        memory: &ExternalMemory<B>,
        sync: ExternalSync<B>,
    ) -> Result<()> {
        if encoder.device != self.device.id
            || memory.device != self.device.id
            || sync.device != self.device.id
        {
            return Err(Error::WrongDevice);
        }
        let result = B::acquire_external(encoder.ready()?, &memory.native, sync.native);
        if result.is_err() {
            // External acquire can alter native ownership as well as command
            // state; no later command may conceal that partial transition.
            encoder.poisoned = true;
        }
        result
    }
    pub fn release(
        &self,
        encoder: &mut CommandEncoder<B>,
        memory: &ExternalMemory<B>,
    ) -> Result<ExternalSync<B>> {
        if encoder.device != self.device.id || memory.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let result = B::release_external(encoder.ready()?, &memory.native);
        if result.is_err() {
            // A failed release leaves external synchronization ambiguous, so
            // retain the same terminal rule as other native commands.
            encoder.poisoned = true;
        }
        result.map(|native| ExternalSync {
            device: self.device.id,
            native,
        })
    }
}
