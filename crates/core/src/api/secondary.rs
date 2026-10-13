//! Secondary command recording and execution contract.
//!
//! Secondary work stays device-local and is consumed only by a compatible
//! primary encoder. Core does not reorder, cache, or replay recorded commands.
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

/// Attachment signature inherited by draw-only secondary raster work.
/// The parent pass must use the same formats, sample count, and view mask.
#[derive(Clone)]
pub struct SecondaryRasterDesc {
    pub color_formats: Vec<Option<TextureFormat>>,
    pub depth_stencil_format: Option<TextureFormat>,
    pub samples: u32,
    pub multiview_mask: Option<u32>,
}
impl PartialEq for SecondaryRasterDesc {
    fn eq(&self, other: &Self) -> bool {
        // Labels are diagnostic-only. Including them would reject compatible
        // secondary work merely because logging text changed.
        self.color_formats == other.color_formats
            && self.depth_stencil_format == other.depth_stencil_format
            && self.samples == other.samples
            && self.multiview_mask == other.multiview_mask
    }
}
pub trait SecondaryBackend: RasterBackend + DeferredBackend {
    type NativeSecondaryEncoder;
    type NativeSecondaryWork;
    fn begin_secondary(
        device: &Self::NativeDevice,
        desc: &SecondaryRasterDesc,
    ) -> Result<Self::NativeSecondaryEncoder>;
    fn secondary_set_pipeline(
        encoder: &mut Self::NativeSecondaryEncoder,
        pipeline: &Self::NativeRasterPipeline,
    ) -> Result<()>;
    fn secondary_set_bind_group(
        encoder: &mut Self::NativeSecondaryEncoder,
        slot: u32,
        group: &Self::NativeBindGroup,
        offsets: &[u32],
    ) -> Result<()>;
    fn secondary_set_vertex_buffer(
        encoder: &mut Self::NativeSecondaryEncoder,
        slot: u32,
        buffer: &Self::NativeBuffer,
        offset: u64,
        size: u64,
    ) -> Result<()>;
    fn secondary_set_index_buffer(
        encoder: &mut Self::NativeSecondaryEncoder,
        buffer: &Self::NativeBuffer,
        offset: u64,
        size: u64,
        format: IndexFormat,
    ) -> Result<()>;
    fn secondary_viewport(
        encoder: &mut Self::NativeSecondaryEncoder,
        viewport: Viewport,
    ) -> Result<()>;
    fn secondary_scissor(encoder: &mut Self::NativeSecondaryEncoder, rect: Rect) -> Result<()>;
    fn secondary_set_push_constants(
        encoder: &mut Self::NativeSecondaryEncoder,
        stages: StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> Result<()>;
    fn secondary_draw(
        encoder: &mut Self::NativeSecondaryEncoder,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) -> Result<()>;
    fn secondary_draw_indexed(
        encoder: &mut Self::NativeSecondaryEncoder,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> Result<()>;
    fn finish_secondary(encoder: Self::NativeSecondaryEncoder)
    -> Result<Self::NativeSecondaryWork>;
    fn execute_secondary(
        encoder: &mut Self::NativeEncoder,
        work: &Self::NativeSecondaryWork,
    ) -> Result<()>;
}
pub struct SecondaryEncoder<B: SecondaryBackend> {
    device: DeviceId,
    native: Option<B::NativeSecondaryEncoder>,
    poisoned: bool,
    push_constants: bool,
    signature: SecondaryRasterDesc,
}
impl<B: SecondaryBackend> SecondaryEncoder<B> {
    fn call(&mut self, f: impl FnOnce(&mut B::NativeSecondaryEncoder) -> Result<()>) -> Result<()> {
        if self.poisoned {
            return Err(Error::InvalidState);
        }
        let result = f(self.native.as_mut().ok_or(Error::InvalidState)?);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    pub fn set_pipeline(&mut self, pipeline: &RasterPipeline<B>) -> Result<()> {
        if pipeline.device != self.device {
            return Err(Error::WrongDevice);
        }
        self.call(|encoder| B::secondary_set_pipeline(encoder, &pipeline.native))
    }
    pub fn set_bind_group(
        &mut self,
        slot: u32,
        group: &BindGroup<B>,
        offsets: &[u32],
    ) -> Result<()> {
        if group.device != self.device {
            return Err(Error::WrongDevice);
        }
        self.call(|encoder| B::secondary_set_bind_group(encoder, slot, &group.native, offsets))
    }
    pub fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &Buffer<B>,
        range: Range<u64>,
    ) -> Result<()> {
        if buffer.device != self.device {
            return Err(Error::WrongDevice);
        }
        if range.start > range.end {
            return Err(Error::InvalidRange);
        }
        self.call(|encoder| {
            B::secondary_set_vertex_buffer(
                encoder,
                slot,
                &buffer.native,
                range.start,
                range.end - range.start,
            )
        })
    }
    pub fn set_index_buffer(
        &mut self,
        buffer: &Buffer<B>,
        range: Range<u64>,
        format: IndexFormat,
    ) -> Result<()> {
        if buffer.device != self.device {
            return Err(Error::WrongDevice);
        }
        if range.start > range.end {
            return Err(Error::InvalidRange);
        }
        self.call(|encoder| {
            B::secondary_set_index_buffer(
                encoder,
                &buffer.native,
                range.start,
                range.end - range.start,
                format,
            )
        })
    }
    pub fn viewport(&mut self, viewport: Viewport) -> Result<()> {
        self.call(|encoder| B::secondary_viewport(encoder, viewport))
    }
    pub fn scissor(&mut self, rect: Rect) -> Result<()> {
        self.call(|encoder| B::secondary_scissor(encoder, rect))
    }
    pub fn push_constants(&mut self, stages: StageMask, offset: u32, bytes: &[u8]) -> Result<()> {
        if !self.push_constants {
            return Err(Error::Unsupported("push constants"));
        }
        self.call(|encoder| B::secondary_set_push_constants(encoder, stages, offset, bytes))
    }
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> Result<()> {
        if !valid_u32_range(&vertices) || !valid_u32_range(&instances) {
            return Err(Error::InvalidRange);
        }
        self.call(|encoder| B::secondary_draw(encoder, vertices, instances))
    }
    pub fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> Result<()> {
        if !valid_u32_range(&indices) || !valid_u32_range(&instances) {
            return Err(Error::InvalidRange);
        }
        self.call(|encoder| B::secondary_draw_indexed(encoder, indices, base_vertex, instances))
    }
}
pub struct SecondaryRasterWork<B: SecondaryBackend> {
    device: DeviceId,
    native: B::NativeSecondaryWork,
    signature: SecondaryRasterDesc,
}
pub struct SecondaryDomain<'a, B: SecondaryBackend> {
    device: &'a Device<B>,
}
impl<B: SecondaryBackend> Device<B> {
    pub fn secondary(&self) -> Option<SecondaryDomain<'_, B>> {
        self.caps
            .secondary_commands
            .then_some(SecondaryDomain { device: self })
    }
}
impl<B: SecondaryBackend> SecondaryDomain<'_, B> {
    pub fn begin(&self, desc: &SecondaryRasterDesc) -> Result<SecondaryEncoder<B>> {
        Ok(SecondaryEncoder {
            device: self.device.id,
            native: Some(B::begin_secondary(&self.device.native, desc)?),
            poisoned: false,
            push_constants: self.device.caps.push_constants,
            signature: desc.clone(),
        })
    }
    pub fn finish(&self, encoder: SecondaryEncoder<B>) -> Result<SecondaryRasterWork<B>> {
        if encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        if encoder.poisoned {
            return Err(Error::InvalidState);
        }
        Ok(SecondaryRasterWork {
            device: self.device.id,
            native: B::finish_secondary(encoder.native.ok_or(Error::InvalidState)?)?,
            signature: SecondaryRasterDesc {
                color_formats: encoder.signature.color_formats,
                depth_stencil_format: encoder.signature.depth_stencil_format,
                samples: encoder.signature.samples,
                multiview_mask: encoder.signature.multiview_mask,
            },
        })
    }
    pub fn execute(
        &self,
        pass: &mut RasterPass<'_, B>,
        work: &SecondaryRasterWork<B>,
    ) -> Result<()> {
        if pass.encoder.device != self.device.id || work.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        if pass.secondary_inheritance.as_ref() != Some(&work.signature) {
            return Err(Error::InvalidState);
        }
        pass.call(|e| B::execute_secondary(e, &work.native))
    }
}
