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
pub struct SecondaryEncoder<B: SecondaryBackend> {
    device: DeviceId,
    native: Option<B::NativeSecondaryEncoder>,
    poisoned: bool,
    push_constants: bool,
    dynamic_depth_bias: bool,
    dynamic_depth_bias_clamp: bool,
    stencil_reference_per_face: bool,
    indirect_draw: bool,
    multi_draw: bool,
    bound_dynamic_depth_bias: Option<DynamicDepthBias>,
    signature: SecondaryRasterDesc,
    stats: RhiStatsLocal,
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
        if pipeline.dynamic_depth_bias.is_some()
            && (!self.dynamic_depth_bias
                || !<B as RasterBackend>::DYNAMIC_DEPTH_BIAS_AVAILABLE
                || !B::SECONDARY_DYNAMIC_DEPTH_BIAS_AVAILABLE)
        {
            return Err(Error::Unsupported("secondary dynamic depth bias"));
        }
        let result = self.call(|encoder| B::secondary_set_pipeline(encoder, &pipeline.native));
        let result = if result.is_ok() {
            if let Some(initial) = pipeline.dynamic_depth_bias {
                self.call(|encoder| B::secondary_depth_bias(encoder, initial))
            } else {
                Ok(())
            }
        } else {
            result
        };
        if result.is_ok() {
            self.bound_dynamic_depth_bias = pipeline.dynamic_depth_bias;
            self.stats.record_pipeline_bind();
        }
        result
    }
    pub fn set_depth_bias(&mut self, value: DynamicDepthBias) -> Result<()> {
        if !self.dynamic_depth_bias {
            return Err(Error::Unsupported("dynamic depth bias"));
        }
        if !<B as RasterBackend>::DYNAMIC_DEPTH_BIAS_AVAILABLE
            || !B::SECONDARY_DYNAMIC_DEPTH_BIAS_AVAILABLE
        {
            return Err(Error::Unsupported("secondary dynamic depth bias"));
        }
        if self.bound_dynamic_depth_bias.is_none() {
            return Err(Error::InvalidState);
        }
        if !value.is_finite() {
            return Err(Error::InvalidRange);
        }
        if value.clamp != 0.0 && !self.dynamic_depth_bias_clamp {
            return Err(Error::Unsupported("dynamic depth bias clamp"));
        }
        self.call(|encoder| B::secondary_depth_bias(encoder, value))
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
        let result =
            self.call(|encoder| B::secondary_set_bind_group(encoder, slot, &group.native, offsets));
        if result.is_ok() {
            self.stats.record_bind_group_bind();
        }
        result
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
        let result = self.call(|encoder| {
            B::secondary_set_vertex_buffer(
                encoder,
                slot,
                &buffer.native,
                range.start,
                range.end - range.start,
            )
        });
        if result.is_ok() {
            self.stats.record_vertex_buffer_bind();
        }
        result
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
        let result = self.call(|encoder| {
            B::secondary_set_index_buffer(
                encoder,
                &buffer.native,
                range.start,
                range.end - range.start,
                format,
            )
        });
        if result.is_ok() {
            self.stats.record_index_buffer_bind();
        }
        result
    }
    pub fn viewport(&mut self, viewport: Viewport) -> Result<()> {
        let result = self.call(|encoder| B::secondary_viewport(encoder, viewport));
        if result.is_ok() {
            self.stats.record_viewport_set();
        }
        result
    }
    pub fn scissor(&mut self, rect: Rect) -> Result<()> {
        let result = self.call(|encoder| B::secondary_scissor(encoder, rect));
        if result.is_ok() {
            self.stats.record_scissor_set();
        }
        result
    }

    /// Sets the blend constant for subsequent draws in this secondary work.
    pub fn blend_constant(&mut self, value: [f32; 4]) -> Result<()> {
        if !B::SECONDARY_BLEND_AND_STENCIL_AVAILABLE {
            return Err(Error::Unsupported("secondary blend constant"));
        }
        if value.iter().any(|component| !component.is_finite()) {
            return Err(Error::InvalidRange);
        }
        let result = self.call(|encoder| B::secondary_blend_constant(encoder, value));
        if result.is_ok() {
            self.stats.record_blend_constant_set();
        }
        result
    }

    /// Sets the stencil reference for both faces using the low eight bits.
    pub fn stencil_reference(&mut self, value: u32) -> Result<()> {
        if !B::SECONDARY_BLEND_AND_STENCIL_AVAILABLE {
            return Err(Error::Unsupported("secondary stencil reference"));
        }
        let result = self.call(|encoder| B::secondary_stencil_reference(encoder, value));
        if result.is_ok() {
            self.stats.record_stencil_reference_set();
        }
        result
    }

    /// Sets the front and back stencil references, each using its low eight bits.
    pub fn stencil_reference_faces(&mut self, front: u32, back: u32) -> Result<()> {
        if !self.stencil_reference_per_face {
            return Err(Error::Unsupported("secondary per-face stencil reference"));
        }
        let result =
            self.call(|encoder| B::secondary_stencil_reference_faces(encoder, front, back));
        if result.is_ok() {
            self.stats.record_stencil_reference_set();
        }
        result
    }
    pub fn push_constants(&mut self, stages: StageMask, offset: u32, bytes: &[u8]) -> Result<()> {
        if !self.push_constants {
            return Err(Error::Unsupported("push constants"));
        }
        let result =
            self.call(|encoder| B::secondary_set_push_constants(encoder, stages, offset, bytes));
        if result.is_ok() {
            self.stats.record_push_constant_write();
        }
        result
    }
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> Result<()> {
        if !valid_u32_range(&vertices) || !valid_u32_range(&instances) {
            return Err(Error::InvalidRange);
        }
        let result = self.call(|encoder| B::secondary_draw(encoder, vertices, instances));
        if result.is_ok() {
            self.stats.record_draw();
        }
        result
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
        let result = self
            .call(|encoder| B::secondary_draw_indexed(encoder, indices, base_vertex, instances));
        if result.is_ok() {
            self.stats.record_draw_indexed();
        }
        result
    }

    /// Records one indirect draw. The caller transitions `args` to
    /// `INDIRECT_READ` before the primary pass executes this work.
    /// The record's first instance must be zero unless `indirect_first_instance`
    /// is supported by the device.
    /// A backend may reject a pipeline that needs to modify bindings as part
    /// of its indirect draw lowering.
    pub fn draw_indirect(&mut self, args: &Buffer<B>, offset: u64, indexed: bool) -> Result<()> {
        if !self.indirect_draw {
            return Err(Error::Unsupported("secondary indirect draw"));
        }
        self.record_indirect(
            args,
            offset,
            1,
            encoder::indirect_argument_size(indexed),
            indexed,
        )
    }

    /// Records a fixed number of indirect draws. The argument state and
    /// first-instance requirements are the same as for [`Self::draw_indirect`].
    pub fn multi_draw(
        &mut self,
        args: &Buffer<B>,
        offset: u64,
        count: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<()> {
        if !self.multi_draw {
            return Err(Error::Unsupported("secondary multi draw"));
        }
        self.record_indirect(args, offset, count, stride, indexed)
    }

    fn record_indirect(
        &mut self,
        args: &Buffer<B>,
        offset: u64,
        count: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<()> {
        if args.device != self.device {
            return Err(Error::WrongDevice);
        }
        if offset % 4 != 0
            || (count != 0
                && (stride < encoder::indirect_argument_size(indexed) || stride % 4 != 0))
        {
            return Err(Error::InvalidRange);
        }
        let result = self.call(|encoder| {
            B::secondary_multi_draw(encoder, &args.native, offset, count, stride, indexed)
        });
        if result.is_ok() && count != 0 {
            if indexed {
                self.stats.record_draw_indexed();
            } else {
                self.stats.record_draw();
            }
        }
        result
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
    /// Reports device support for distinct front and back references in secondary work.
    pub fn supports_stencil_reference_per_face(&self) -> bool {
        B::SECONDARY_PER_FACE_STENCIL_AVAILABLE && self.device.caps.stencil_reference_per_face
    }

    /// Reports device support for one indirect draw in secondary raster work.
    pub fn supports_indirect_draw(&self) -> bool {
        B::SECONDARY_INDIRECT_AVAILABLE && self.device.caps.indirect_draw
    }

    /// Reports device support for fixed-count multi-draw in secondary raster work.
    pub fn supports_multi_draw(&self) -> bool {
        B::SECONDARY_INDIRECT_AVAILABLE && self.device.caps.multi_draw
    }

    pub fn begin(&self, desc: &SecondaryRasterDesc) -> Result<SecondaryEncoder<B>> {
        Ok(SecondaryEncoder {
            device: self.device.id,
            native: Some(B::begin_secondary(&self.device.native, desc)?),
            poisoned: false,
            push_constants: self.device.caps.push_constants,
            dynamic_depth_bias: self.device.caps.dynamic_depth_bias,
            dynamic_depth_bias_clamp: self.device.caps.dynamic_depth_bias_clamp,
            stencil_reference_per_face: self.supports_stencil_reference_per_face(),
            indirect_draw: self.supports_indirect_draw(),
            multi_draw: self.supports_multi_draw(),
            bound_dynamic_depth_bias: None,
            signature: desc.clone(),
            stats: self.device.stats.local(),
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
