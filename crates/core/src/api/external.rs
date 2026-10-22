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
    pub usage: BufferUsage,
}
/// Image subresources and matching ownership-transfer layouts. The external
/// producer and consumer must issue matching transitions on their queues.
pub struct ExternalTextureTransition {
    pub aspects: TextureAspects,
    pub mips: Range<u32>,
    pub layers: Range<u32>,
    pub usage: StateTransition<TextureUses>,
    pub layout: (ImageLayout, ImageLayout),
}
/// Platform bridge chooses this concrete native handle type; RHI never stores
/// it in dyn Any or downcasts it on command paths.
pub trait ExternalBackend: CommandBackend {
    type NativeExternalHandle;
    type NativeExternalMemory: Clone;
    type NativeExternalTexture: Clone;
    type NativeExternalSync;
    /// Returns the imported buffer usable by ordinary command operations.
    fn external_buffer(memory: &Self::NativeExternalMemory) -> Result<Self::NativeBuffer>;
    /// Returns the imported texture usable by ordinary view and command operations.
    fn external_texture(texture: &Self::NativeExternalTexture) -> Result<Self::NativeTexture>;
    /// Imports a producer synchronization payload supplied by the platform bridge.
    fn import_external_sync(
        device: &Self::NativeDevice,
        sync: Self::NativeExternalHandle,
    ) -> Result<Self::NativeExternalSync>;
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
    fn acquire_external_texture(
        encoder: &mut Self::NativeEncoder,
        texture: &Self::NativeExternalTexture,
        sync: Self::NativeExternalSync,
        transition: &ExternalTextureTransition,
    ) -> Result<()>;
    fn release_external_texture(
        encoder: &mut Self::NativeEncoder,
        texture: &Self::NativeExternalTexture,
        transition: &ExternalTextureTransition,
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
    /// Import producer synchronization. The handle must be compatible with
    /// the target device and its producer must signal it exactly once.
    ///
    /// # Safety
    /// The bridge guarantees native handle validity and synchronization. Each
    /// binary payload must be signaled and waited in the order required by its
    /// native API; importing aliases does not create independent signals.
    pub unsafe fn import_sync(&self, handle: B::NativeExternalHandle) -> Result<ExternalSync<B>> {
        Ok(ExternalSync {
            device: self.device.id,
            native: B::import_external_sync(&self.device.native, handle)?,
        })
    }
    pub fn buffer(&self, memory: &ExternalMemory<B>) -> Result<Buffer<B>> {
        if memory.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        // SAFETY: the backend returns a buffer from the same checked device.
        Ok(unsafe { Buffer::from_native(self.device.id, B::external_buffer(&memory.native)?) })
    }
    pub fn texture(&self, texture: &ExternalTexture<B>) -> Result<Texture<B>> {
        if texture.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        // SAFETY: the backend returns a texture from the same checked device.
        Ok(unsafe { Texture::from_native(self.device.id, B::external_texture(&texture.native)?) })
    }
    pub fn acquire_texture(
        &self,
        encoder: &mut CommandEncoder<B>,
        texture: &ExternalTexture<B>,
        sync: ExternalSync<B>,
        transition: &ExternalTextureTransition,
    ) -> Result<()> {
        if encoder.device != self.device.id
            || texture.device != self.device.id
            || sync.device != self.device.id
        {
            return Err(Error::WrongDevice);
        }
        let result =
            B::acquire_external_texture(encoder.ready()?, &texture.native, sync.native, transition);
        if result.is_err() {
            encoder.poisoned = true;
        }
        result
    }
    /// Records an ownership release. The returned synchronization payload is
    /// signaled only after the recorded work is accepted and reaches the release.
    pub fn release_texture(
        &self,
        encoder: &mut CommandEncoder<B>,
        texture: &ExternalTexture<B>,
        transition: &ExternalTextureTransition,
    ) -> Result<ExternalSync<B>> {
        if encoder.device != self.device.id || texture.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let result = B::release_external_texture(encoder.ready()?, &texture.native, transition);
        if result.is_err() {
            encoder.poisoned = true;
        }
        result.map(|native| ExternalSync {
            device: self.device.id,
            native,
        })
    }
    /// # Safety
    /// The bridge supplies a valid handle and exact producer allocation and
    /// resource metadata. The producer must release ownership and signal a
    /// synchronization payload before this resource is accessed.
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
    /// # Safety
    /// The bridge supplies a valid handle and exact producer allocation and
    /// texture metadata. It must provide matching ownership transfers, image
    /// layouts and synchronization for every access.
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
    /// Records an ownership release. The returned synchronization payload is
    /// signaled only after the recorded work is accepted and reaches the release.
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

impl<B: ExternalBackend> ExternalSync<B> {
    /// Transfers the native synchronization payload to the platform bridge.
    pub fn into_native(self) -> B::NativeExternalSync {
        self.native
    }
}
