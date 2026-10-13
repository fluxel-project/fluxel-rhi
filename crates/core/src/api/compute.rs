//! Compute pipeline creation, pass recording, and dispatch contract.
//!
//! RenderGraph supplies dispatch order, resource states, and synchronization;
//! this module describes and records only the requested compute work.
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

/// Per-backend fixed static domain implementation. Unavailable is one shared
/// zero-size implementation, not a fake resource or a dynamic vtable.
pub trait ComputeOps<B: Backend> {
    const AVAILABLE: bool;
    type PipelineFuture<'a>: Future<Output = Result<B::NativeComputePipeline>> + 'a
    where
        B: 'a;
    fn create_pipeline<'a>(
        device: &'a B::NativeDevice,
        desc: &'a ComputePipelineDesc<B>,
    ) -> Self::PipelineFuture<'a>;
    fn begin(encoder: &mut B::NativeEncoder) -> Result<()>;
    fn set_pipeline(
        encoder: &mut B::NativeEncoder,
        pipeline: &B::NativeComputePipeline,
    ) -> Result<()>;
    fn set_bind_group(
        encoder: &mut B::NativeEncoder,
        slot: u32,
        group: &B::NativeBindGroup,
        offsets: &[u32],
    ) -> Result<()>;
    fn dispatch(encoder: &mut B::NativeEncoder, groups: [u32; 3]) -> Result<()>;
    fn dispatch_indirect(
        encoder: &mut B::NativeEncoder,
        args: &B::NativeBuffer,
        offset: u64,
    ) -> Result<()>;
    fn set_push_constants(_: &mut B::NativeEncoder, _: u32, _: &[u8]) -> Result<()> {
        Err(Error::Unsupported("compute push constants"))
    }
    fn end(encoder: &mut B::NativeEncoder) -> Result<()>;
    fn abort(encoder: &mut B::NativeEncoder);
}
impl<B: Backend> ComputeOps<B> for Unavailable {
    const AVAILABLE: bool = false;
    type PipelineFuture<'a>
        = std::future::Ready<Result<B::NativeComputePipeline>>
    where
        B: 'a;
    fn create_pipeline<'a>(
        _: &'a B::NativeDevice,
        _: &'a ComputePipelineDesc<B>,
    ) -> Self::PipelineFuture<'a> {
        std::future::ready(Err(Error::Unsupported("compute")))
    }
    fn begin(_: &mut B::NativeEncoder) -> Result<()> {
        Err(Error::Unsupported("compute"))
    }
    fn set_pipeline(_: &mut B::NativeEncoder, _: &B::NativeComputePipeline) -> Result<()> {
        Err(Error::Unsupported("compute"))
    }
    fn set_bind_group(
        _: &mut B::NativeEncoder,
        _: u32,
        _: &B::NativeBindGroup,
        _: &[u32],
    ) -> Result<()> {
        Err(Error::Unsupported("compute"))
    }
    fn dispatch(_: &mut B::NativeEncoder, _: [u32; 3]) -> Result<()> {
        Err(Error::Unsupported("compute"))
    }
    fn dispatch_indirect(_: &mut B::NativeEncoder, _: &B::NativeBuffer, _: u64) -> Result<()> {
        Err(Error::Unsupported("compute"))
    }
    fn end(_: &mut B::NativeEncoder) -> Result<()> {
        Err(Error::Unsupported("compute"))
    }
    fn abort(_: &mut B::NativeEncoder) {}
}

pub struct ComputeDomain<'a, B: Backend> {
    device: &'a Device<B>,
}
impl<B: Backend> Device<B> {
    /// Available to Renderer<B: Backend> with no specialization, but operations
    /// remain statically bound to B::Compute and never inspect a dyn interface.
    pub fn compute(&self) -> Option<ComputeDomain<'_, B>> {
        (self.caps.compute && B::Compute::AVAILABLE).then_some(ComputeDomain { device: self })
    }
}
impl<B: Backend> ComputeDomain<'_, B> {
    pub async fn create_pipeline(
        &self,
        desc: ComputePipelineDesc<B>,
    ) -> Result<ComputePipeline<B>> {
        if desc.layout.device != self.device.id
            || desc.compute.module.device != self.device.id
            || desc
                .cache
                .as_ref()
                .is_some_and(|c| c.device != self.device.id)
        {
            return Err(Error::WrongDevice);
        }
        let native = B::Compute::create_pipeline(&self.device.native, &desc).await?;
        // SAFETY: the SPI created this pipeline for this exact native device.
        Ok(unsafe { ComputePipeline::from_native(self.device.id, native) })
    }
    pub fn begin<'a>(&self, encoder: &'a mut CommandEncoder<B>) -> Result<ComputePass<'a, B>>
    where
        B: CommandBackend,
    {
        if encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let result = B::Compute::begin(encoder.ready()?);
        if let Err(e) = result {
            encoder.poisoned = true;
            return Err(e);
        }
        encoder.in_pass = true;
        Ok(ComputePass {
            encoder,
            active: true,
        })
    }
}
pub struct ComputePass<'a, B: CommandBackend> {
    encoder: &'a mut CommandEncoder<B>,
    active: bool,
}
impl<B: CommandBackend> ComputePass<'_, B> {
    fn call(&mut self, f: impl FnOnce(&mut B::NativeEncoder) -> Result<()>) -> Result<()> {
        if self.encoder.poisoned {
            return Err(Error::InvalidState);
        }
        let result = f(self.encoder.native.as_mut().ok_or(Error::InvalidState)?);
        if result.is_err() {
            self.encoder.poisoned = true;
        }
        result
    }
    pub fn set_pipeline(&mut self, p: &ComputePipeline<B>) -> Result<()> {
        if p.device != self.encoder.device {
            return Err(Error::WrongDevice);
        }
        self.call(|e| B::Compute::set_pipeline(e, &p.native))
    }
    pub fn set_bind_group(
        &mut self,
        slot: u32,
        group: &BindGroup<B>,
        offsets: &[u32],
    ) -> Result<()> {
        if group.device != self.encoder.device {
            return Err(Error::WrongDevice);
        }
        self.call(|e| B::Compute::set_bind_group(e, slot, &group.native, offsets))
    }
    pub fn dispatch(&mut self, groups: [u32; 3]) -> Result<()> {
        self.call(|e| B::Compute::dispatch(e, groups))
    }
    pub fn push_constants(&mut self, offset: u32, bytes: &[u8]) -> Result<()> {
        self.call(|e| B::Compute::set_push_constants(e, offset, bytes))
    }
    pub fn dispatch_indirect(&mut self, buffer: &Buffer<B>, offset: u64) -> Result<()> {
        if buffer.device != self.encoder.device {
            return Err(Error::WrongDevice);
        }
        self.call(|e| B::Compute::dispatch_indirect(e, &buffer.native, offset))
    }
    pub fn finish(mut self) -> Result<()> {
        // Clear `in_pass` only after native end succeeds; otherwise a caller
        // cannot record new work after a potentially half-ended pass.
        if self.encoder.poisoned {
            if let Some(native) = self.encoder.native.as_mut() {
                B::Compute::abort(native);
            }
            self.active = false;
            self.encoder.in_pass = false;
            return Err(Error::InvalidState);
        }
        let result = B::Compute::end(self.encoder.native.as_mut().ok_or(Error::InvalidState)?);
        self.active = false;
        self.encoder.in_pass = false;
        if result.is_err() {
            self.encoder.poisoned = true;
        }
        result
    }
}
impl<B: CommandBackend> Drop for ComputePass<'_, B> {
    fn drop(&mut self) {
        if self.active {
            if let Some(native) = self.encoder.native.as_mut() {
                B::Compute::abort(native);
            }
            self.encoder.in_pass = false;
            self.encoder.poisoned = true;
        }
    }
}
