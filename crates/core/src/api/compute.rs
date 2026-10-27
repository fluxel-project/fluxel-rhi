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
        self.begin_prepared(encoder, |_| Ok(()))
    }
    /// Begins a compute pass and, when the backend exposes compute-pass debug
    /// annotations, records `label` as a native debug group.
    pub fn begin_labeled<'a>(
        &self,
        encoder: &'a mut CommandEncoder<B>,
        label: &str,
    ) -> Result<ComputePass<'a, B>>
    where
        B: CommandBackend,
    {
        self.begin_prepared_labeled(encoder, Some(label), |_| Ok(()))
    }
    pub(crate) fn begin_prepared<'a>(
        &self,
        encoder: &'a mut CommandEncoder<B>,
        prepare: impl FnOnce(&mut B::NativeEncoder) -> Result<()>,
    ) -> Result<ComputePass<'a, B>>
    where
        B: CommandBackend,
    {
        self.begin_prepared_labeled(encoder, None, prepare)
    }
    pub(crate) fn begin_prepared_labeled<'a>(
        &self,
        encoder: &'a mut CommandEncoder<B>,
        label: Option<&str>,
        prepare: impl FnOnce(&mut B::NativeEncoder) -> Result<()>,
    ) -> Result<ComputePass<'a, B>>
    where
        B: CommandBackend,
    {
        if encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        if encoder.debug_group_depth != 0 {
            return Err(Error::InvalidState);
        }
        let result = prepare(encoder.ready()?)
            .and_then(|()| B::Compute::begin(encoder.native.as_mut().expect("ready encoder")));
        if let Err(e) = result {
            encoder.poisoned = true;
            return Err(e);
        }
        let debug_label_active = label.is_some() && encoder.debug_markers_compute_pass;
        if let Some(label) = label.filter(|_| debug_label_active) {
            if let Err(error) = B::debug_marker(
                encoder.native.as_mut().expect("ready encoder"),
                DebugMarkerScope::ComputePass,
                DebugMarkerOp::Push(label),
            ) {
                B::Compute::abort(encoder.native.as_mut().expect("active compute encoder"));
                encoder.poisoned = true;
                return Err(error);
            }
            encoder.debug_group_depth += 1;
        }
        encoder.in_pass = true;
        Ok(ComputePass {
            debug_group_depth_at_begin: encoder.debug_group_depth,
            debug_label_active,
            encoder,
            active: true,
        })
    }
}
pub struct ComputePass<'a, B: CommandBackend> {
    encoder: &'a mut CommandEncoder<B>,
    active: bool,
    debug_group_depth_at_begin: u32,
    debug_label_active: bool,
}
impl<B: CommandBackend> ComputePass<'_, B> {
    fn debug_annotation(&mut self, op: DebugMarkerOp<'_>) -> Result<()> {
        if !self.encoder.debug_markers_compute_pass {
            return Err(Error::Unsupported("compute-pass debug markers"));
        }
        match op {
            DebugMarkerOp::Push(_) => {
                self.encoder.debug_group_depth = self
                    .encoder
                    .debug_group_depth
                    .checked_add(1)
                    .ok_or(Error::InvalidState)?
            }
            DebugMarkerOp::Pop
                if self.encoder.debug_group_depth <= self.debug_group_depth_at_begin =>
            {
                return Err(Error::InvalidState);
            }
            DebugMarkerOp::Pop => self.encoder.debug_group_depth -= 1,
            DebugMarkerOp::Insert(_) => {}
        }
        let result = self.call(|e| B::debug_marker(e, DebugMarkerScope::ComputePass, op));
        if result.is_err() {
            match op {
                DebugMarkerOp::Push(_) => self.encoder.debug_group_depth -= 1,
                DebugMarkerOp::Pop => self.encoder.debug_group_depth += 1,
                _ => {}
            }
        }
        result
    }
    pub fn insert_debug_marker(&mut self, label: &str) -> Result<()> {
        self.debug_annotation(DebugMarkerOp::Insert(label))
    }
    pub fn push_debug_group(&mut self, label: &str) -> Result<()> {
        self.debug_annotation(DebugMarkerOp::Push(label))
    }
    pub fn pop_debug_group(&mut self) -> Result<()> {
        self.debug_annotation(DebugMarkerOp::Pop)
    }
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
        let result = self.call(|e| B::Compute::set_pipeline(e, &p.native));
        if result.is_ok() {
            self.encoder.stats.record_pipeline_bind();
        }
        result
    }
    /// Binds a group and applies dynamic buffer offsets.
    ///
    /// `offsets` must contain exactly one value for each layout binding whose
    /// buffer type enables `dynamic_offset`, in ascending binding-number order.
    /// Each value is added to that buffer resource's bound range start; the
    /// backend validates alignment and that the resulting range stays in the
    /// allocation.
    pub fn set_bind_group(
        &mut self,
        slot: u32,
        group: &BindGroup<B>,
        offsets: &[u32],
    ) -> Result<()> {
        if group.device != self.encoder.device {
            return Err(Error::WrongDevice);
        }
        let result = self.call(|e| B::Compute::set_bind_group(e, slot, &group.native, offsets));
        if result.is_ok() {
            self.encoder.stats.record_bind_group_bind();
        }
        result
    }
    pub fn dispatch(&mut self, groups: [u32; 3]) -> Result<()> {
        let result = self.call(|e| B::Compute::dispatch(e, groups));
        if result.is_ok() {
            self.encoder.stats.record_dispatch();
        }
        result
    }
    pub fn push_constants(&mut self, offset: u32, bytes: &[u8]) -> Result<()> {
        let result = self.call(|e| B::Compute::set_push_constants(e, offset, bytes));
        if result.is_ok() {
            self.encoder.stats.record_push_constant_write();
        }
        result
    }
    /// Reads three `u32` workgroup counts from `buffer` at `offset`.
    ///
    /// The offset must be four-byte aligned, the complete 12-byte record must
    /// fit the buffer, and the buffer must include `BufferUsage::INDIRECT`.
    pub fn dispatch_indirect(&mut self, buffer: &Buffer<B>, offset: u64) -> Result<()> {
        if buffer.device != self.encoder.device {
            return Err(Error::WrongDevice);
        }
        let result = self.call(|e| B::Compute::dispatch_indirect(e, &buffer.native, offset));
        if result.is_ok() {
            self.encoder.stats.record_dispatch();
        }
        result
    }
    pub fn finish(mut self) -> Result<()> {
        if self.encoder.debug_group_depth != self.debug_group_depth_at_begin {
            if let Some(native) = self.encoder.native.as_mut() {
                B::Compute::abort(native);
            }
            self.active = false;
            self.encoder.in_pass = false;
            self.encoder.poisoned = true;
            return Err(Error::InvalidState);
        }
        if self.encoder.poisoned {
            if let Some(native) = self.encoder.native.as_mut() {
                B::Compute::abort(native);
            }
            self.active = false;
            self.encoder.in_pass = false;
            return Err(Error::InvalidState);
        }
        if self.debug_label_active {
            let result = B::debug_marker(
                self.encoder.native.as_mut().ok_or(Error::InvalidState)?,
                DebugMarkerScope::ComputePass,
                DebugMarkerOp::Pop,
            );
            if result.is_err() {
                if let Some(native) = self.encoder.native.as_mut() {
                    B::Compute::abort(native);
                }
                self.active = false;
                self.encoder.in_pass = false;
                self.encoder.poisoned = true;
                return result;
            }
            self.encoder.debug_group_depth -= 1;
        }
        // Clear `in_pass` only after native end succeeds; otherwise a caller
        // cannot record new work after a potentially half-ended pass.
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
