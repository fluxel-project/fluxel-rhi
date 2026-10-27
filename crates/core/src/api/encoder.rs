//! Command encoder ownership and backend recording protocol.
//!
//! Encoders record work for one device and become unusable after finish or
//! abandonment. They do not schedule passes, infer hazards, or retain a CPU
//! replay stream beyond the backend's native recorded work.
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

/// Value written to every texel of the selected texture subresources.
#[derive(Clone, Copy, Debug)]
pub enum TextureClearValue {
    /// A color value matching the destination format's numeric type.
    Color(ClearColor),
    /// A depth value in the inclusive range 0..=1.
    Depth(f32),
}

pub struct CommandEncoder<B: CommandBackend> {
    pub(crate) device: DeviceId,
    pub(crate) lane: LaneId,
    pub(crate) native: Option<B::NativeEncoder>,
    pub(crate) in_pass: bool,
    pub(crate) poisoned: bool,
    pub(crate) multiview: bool,
    pub(crate) max_multiview_views: u32,
    pub(crate) stats: RhiStatsLocal,
    pub(crate) debug_markers_encoder: bool,
    pub(crate) debug_markers_raster_pass: bool,
    pub(crate) debug_markers_compute_pass: bool,
    pub(crate) dynamic_depth_bias: bool,
    pub(crate) dynamic_depth_bias_clamp: bool,
    pub(crate) rect_attachment_clear: bool,
    pub(crate) debug_group_depth: u32,
}
impl<B: CommandBackend> CommandEncoder<B> {
    pub(crate) fn new(device: &Device<B>, lane: LaneId) -> Result<Self> {
        if lane.device != device.id {
            return Err(Error::WrongLane);
        }
        // Some legacy/direct backends have no lane inventory. When the
        // opened device did publish one, reject an ordinal outside it before
        // crossing the native recording boundary.
        if !device.lanes().is_empty() && !device.lanes().iter().any(|info| info.id == lane) {
            return Err(Error::WrongLane);
        }
        Ok(Self {
            device: device.id,
            lane,
            native: Some(B::begin_encoder(&device.native, lane)?),
            in_pass: false,
            poisoned: false,
            multiview: device.caps.multiview,
            max_multiview_views: device.caps.limits.max_multiview_views,
            stats: device.stats.local(),
            debug_markers_encoder: device.caps.debug_markers_encoder,
            debug_markers_raster_pass: device.caps.debug_markers_raster_pass,
            debug_markers_compute_pass: device.caps.debug_markers_compute_pass,
            dynamic_depth_bias: device.caps.dynamic_depth_bias,
            dynamic_depth_bias_clamp: device.caps.dynamic_depth_bias_clamp,
            rect_attachment_clear: device.caps.rect_attachment_clear,
            debug_group_depth: 0,
        })
    }
    pub(crate) fn ready(&mut self) -> Result<&mut B::NativeEncoder> {
        // A pass owns the native encoder until it finishes; allowing another
        // command here would make pass nesting backend-dependent. A poisoned
        // encoder also cannot be safely resumed because a native call may have
        // changed only part of the driver's state.
        if self.poisoned || self.in_pass {
            return Err(Error::InvalidState);
        }
        self.native.as_mut().ok_or(Error::InvalidState)
    }
    pub(crate) fn call(
        &mut self,
        f: impl FnOnce(&mut B::NativeEncoder) -> Result<()>,
    ) -> Result<()> {
        let result = f(self.ready()?);
        if result.is_err() {
            // Keep the failure sticky. Retrying a later command could hide the
            // point at which native recording became partially observable.
            self.poisoned = true;
        }
        result
    }
    pub fn transition_resources(&mut self, ops: &[ResourceTransition<'_, B>]) -> Result<()> {
        // Core validates provenance and lane endpoints only. It deliberately
        // does not inspect neighboring commands to infer missing transitions.
        if ops.iter().any(|op| !op.valid_for(self.device, self.lane)) {
            return Err(Error::WrongDevice);
        }
        self.call(|e| B::transition(e, ops))
    }
    fn debug_annotation(&mut self, scope: DebugMarkerScope, op: DebugMarkerOp<'_>) -> Result<()> {
        if !self.debug_markers_encoder {
            return Err(Error::Unsupported("encoder debug markers"));
        }
        match op {
            DebugMarkerOp::Push(_) => {
                self.debug_group_depth = self
                    .debug_group_depth
                    .checked_add(1)
                    .ok_or(Error::InvalidState)?
            }
            DebugMarkerOp::Pop if self.debug_group_depth == 0 => return Err(Error::InvalidState),
            DebugMarkerOp::Pop => self.debug_group_depth -= 1,
            DebugMarkerOp::Insert(_) => {}
        }
        let result = self.call(|e| B::debug_marker(e, scope, op));
        if result.is_err() {
            match op {
                DebugMarkerOp::Push(_) => self.debug_group_depth -= 1,
                DebugMarkerOp::Pop => self.debug_group_depth += 1,
                _ => {}
            }
        }
        result
    }
    pub fn insert_debug_marker(&mut self, label: &str) -> Result<()> {
        self.debug_annotation(DebugMarkerScope::Encoder, DebugMarkerOp::Insert(label))
    }
    pub fn push_debug_group(&mut self, label: &str) -> Result<()> {
        self.debug_annotation(DebugMarkerScope::Encoder, DebugMarkerOp::Push(label))
    }
    pub fn pop_debug_group(&mut self) -> Result<()> {
        self.debug_annotation(DebugMarkerScope::Encoder, DebugMarkerOp::Pop)
    }
    pub fn copy_buffer(
        &mut self,
        src: &Buffer<B>,
        src_offset: u64,
        dst: &Buffer<B>,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<()> {
        if src.device != self.device || dst.device != self.device {
            return Err(Error::WrongDevice);
        }
        let result = self
            .call(|e| B::copy_buffer(e, &src.native, src_offset, &dst.native, dst_offset, bytes));
        if result.is_ok() {
            self.stats.record_copy();
        }
        result
    }
    pub fn copy_texture(
        &mut self,
        src: TextureRegion<'_, B>,
        dst: TextureRegion<'_, B>,
    ) -> Result<()> {
        if src.texture.device != self.device || dst.texture.device != self.device {
            return Err(Error::WrongDevice);
        }
        let result = self.call(|e| B::copy_texture(e, src, dst));
        if result.is_ok() {
            self.stats.record_copy();
        }
        result
    }
    pub fn resolve(&mut self, src: TextureRegion<'_, B>, dst: TextureRegion<'_, B>) -> Result<()> {
        if src.texture.device != self.device || dst.texture.device != self.device {
            return Err(Error::WrongDevice);
        }
        let result = self.call(|e| B::resolve(e, src, dst));
        if result.is_ok() {
            self.stats.record_copy();
        }
        result
    }
    pub fn blit(
        &mut self,
        src: TextureRegion<'_, B>,
        dst: TextureRegion<'_, B>,
        filter: FilterMode,
    ) -> Result<()> {
        if src.texture.device != self.device || dst.texture.device != self.device {
            return Err(Error::WrongDevice);
        }
        let result = self.call(|e| B::blit(e, src, dst, filter));
        if result.is_ok() {
            self.stats.record_blit();
        }
        result
    }
    pub fn clear_buffer(&mut self, dst: &Buffer<B>, range: Range<u64>, value: u32) -> Result<()> {
        if dst.device != self.device {
            return Err(Error::WrongDevice);
        }
        let result = self.call(|e| B::clear_buffer(e, &dst.native, range, value));
        if result.is_ok() {
            self.stats.record_clear();
        }
        result
    }
    /// Clears all texels in each selected mip and array layer.
    /// The caller must arrange explicit transitions to the appropriate color
    /// attachment or depth-write state. Backends need not track or validate
    /// the actual current resource state.
    pub fn clear_texture(
        &mut self,
        dst: &Texture<B>,
        mips: Range<u32>,
        layers: Range<u32>,
        value: TextureClearValue,
    ) -> Result<()> {
        if dst.device != self.device {
            return Err(Error::WrongDevice);
        }
        if mips.start >= mips.end || layers.start >= layers.end {
            return Err(Error::InvalidRange);
        }
        if let TextureClearValue::Depth(depth) = value {
            if !depth.is_finite() || !(0.0..=1.0).contains(&depth) {
                return Err(Error::InvalidRange);
            }
        }
        let result = self.call(|e| B::clear_texture(e, &dst.native, mips, layers, value));
        if result.is_ok() {
            self.stats.record_clear();
        }
        result
    }
    pub fn buffer_to_texture(
        &mut self,
        src: &Buffer<B>,
        layout: TexelCopyBufferLayout,
        dst: TextureRegion<'_, B>,
    ) -> Result<()> {
        if src.device != self.device || dst.texture.device != self.device {
            return Err(Error::WrongDevice);
        }
        let result = self.call(|e| B::buffer_to_texture(e, &src.native, layout, dst));
        if result.is_ok() {
            self.stats.record_copy();
        }
        result
    }
    pub fn texture_to_buffer(
        &mut self,
        src: TextureRegion<'_, B>,
        dst: &Buffer<B>,
        layout: TexelCopyBufferLayout,
    ) -> Result<()> {
        if src.texture.device != self.device || dst.device != self.device {
            return Err(Error::WrongDevice);
        }
        let result = self.call(|e| B::texture_to_buffer(e, src, &dst.native, layout));
        if result.is_ok() {
            self.stats.record_copy();
        }
        result
    }
    pub fn write_buffer(&mut self, dst: &Buffer<B>, offset: u64, bytes: &[u8]) -> Result<()> {
        if dst.device != self.device {
            return Err(Error::WrongDevice);
        }
        let result = self.call(|e| B::write_buffer(e, &dst.native, offset, bytes));
        if result.is_ok() {
            self.stats.record_write();
        }
        result
    }
    pub fn write_texture(
        &mut self,
        dst: TextureRegion<'_, B>,
        layout: TexelCopyBufferLayout,
        bytes: &[u8],
    ) -> Result<()> {
        if dst.texture.device != self.device {
            return Err(Error::WrongDevice);
        }
        let result = self.call(|e| B::write_texture(e, dst, layout, bytes));
        if result.is_ok() {
            self.stats.record_write();
        }
        result
    }
}

pub struct RasterPass<'a, B: RasterBackend> {
    pub(crate) encoder: &'a mut CommandEncoder<B>,
    pub(crate) active: bool,
    pub(crate) secondary_inheritance: Option<SecondaryRasterDesc>,
    bound_dynamic_depth_bias: Option<DynamicDepthBias>,
    render_area: Rect,
    color_attachment_slots: Vec<bool>,
    depth_clearable: bool,
    stencil_clearable: bool,
    debug_group_depth_at_min: u32,
    debug_label_active: bool,
}
impl<B: RasterBackend> CommandEncoder<B> {
    pub fn begin_raster<'a>(
        &'a mut self,
        desc: &RasterPassDesc<'_, B>,
    ) -> Result<RasterPass<'a, B>> {
        self.begin_raster_prepared(desc, |_| Ok(()))
    }
    pub(crate) fn begin_raster_prepared<'a>(
        &'a mut self,
        desc: &RasterPassDesc<'_, B>,
        prepare: impl FnOnce(&mut B::NativeEncoder) -> Result<()>,
    ) -> Result<RasterPass<'a, B>> {
        if self.debug_group_depth != 0 {
            return Err(Error::InvalidState);
        }
        if let Some(mask) = desc.multiview_mask {
            if !self.multiview {
                return Err(Error::Unsupported("multiview"));
            }
            if mask == 0 || 32 - mask.leading_zeros() > self.max_multiview_views {
                return Err(Error::InvalidRange);
            }
        }
        if desc.colors.iter().flatten().any(|a| {
            a.view.device() != self.device
                || a.resolve_target
                    .as_ref()
                    .is_some_and(|v| v.device() != self.device)
        }) || desc
            .depth_stencil
            .as_ref()
            .is_some_and(|a| a.view.device() != self.device)
            || desc
                .timestamp_writes
                .as_ref()
                .is_some_and(|t| t.set.device != self.device)
            || desc
                .occlusion_query_set
                .is_some_and(|set| set.device != self.device)
        {
            return Err(Error::WrongDevice);
        }
        if let Some(writes) = &desc.timestamp_writes {
            if writes.set.kind() != QueryKind::Timestamp {
                return Err(Error::InvalidState);
            }
            if writes.begin.is_some_and(|slot| slot >= writes.set.count())
                || writes.end.is_some_and(|slot| slot >= writes.set.count())
            {
                return Err(Error::InvalidRange);
            }
        }
        if desc
            .occlusion_query_set
            .is_some_and(|set| set.kind() != QueryKind::Occlusion)
        {
            return Err(Error::InvalidState);
        }
        if desc.colors.iter().flatten().any(|a| {
            !a.view.is_current() || a.resolve_target.as_ref().is_some_and(|v| !v.is_current())
        }) || desc
            .depth_stencil
            .as_ref()
            .is_some_and(|a| !a.view.is_current())
        {
            return Err(Error::SurfaceOutdated);
        }
        let result = prepare(self.ready()?)
            .and_then(|()| B::begin_raster(self.native.as_mut().expect("ready encoder"), desc));
        if let Err(e) = result {
            self.poisoned = true;
            return Err(e);
        }
        let debug_label_active = desc.label.is_some() && self.debug_markers_raster_pass;
        if let Some(label) = desc.label.filter(|_| debug_label_active) {
            if let Err(error) = B::debug_marker(
                self.native.as_mut().expect("ready encoder"),
                DebugMarkerScope::RasterPass,
                DebugMarkerOp::Push(label),
            ) {
                B::abort_raster(self.native.as_mut().expect("active raster encoder"));
                self.poisoned = true;
                return Err(error);
            }
            self.debug_group_depth += 1;
        }
        self.in_pass = true;
        let debug_group_depth_at_min = self.debug_group_depth;
        Ok(RasterPass {
            encoder: self,
            active: true,
            secondary_inheritance: desc.secondary_inheritance.clone(),
            bound_dynamic_depth_bias: None,
            render_area: desc.render_area,
            color_attachment_slots: desc.colors.iter().map(Option::is_some).collect(),
            depth_clearable: desc.depth_stencil.as_ref().is_some_and(|attachment| {
                attachment.depth.is_some() && !attachment.depth_read_only
            }),
            stencil_clearable: desc.depth_stencil.as_ref().is_some_and(|attachment| {
                attachment.stencil.is_some() && !attachment.stencil_read_only
            }),
            debug_group_depth_at_min,
            debug_label_active,
        })
    }
}
impl<B: RasterBackend> RasterPass<'_, B> {
    fn debug_annotation(&mut self, op: DebugMarkerOp<'_>) -> Result<()> {
        if !self.encoder.debug_markers_raster_pass {
            return Err(Error::Unsupported("raster-pass debug markers"));
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
                if self.encoder.debug_group_depth <= self.debug_group_depth_at_min =>
            {
                return Err(Error::InvalidState);
            }
            DebugMarkerOp::Pop => self.encoder.debug_group_depth -= 1,
            DebugMarkerOp::Insert(_) => {}
        }
        let result = self.call(|native| B::debug_marker(native, DebugMarkerScope::RasterPass, op));
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
    pub(crate) fn call(
        &mut self,
        f: impl FnOnce(&mut B::NativeEncoder) -> Result<()>,
    ) -> Result<()> {
        if self.encoder.poisoned {
            return Err(Error::InvalidState);
        }
        let result = f(self.encoder.native.as_mut().ok_or(Error::InvalidState)?);
        if result.is_err() {
            self.encoder.poisoned = true;
        }
        result
    }
    pub fn set_pipeline(&mut self, p: &RasterPipeline<B>) -> Result<()> {
        if p.device != self.encoder.device {
            return Err(Error::WrongDevice);
        }
        if p.dynamic_depth_bias.is_some() && !B::DYNAMIC_DEPTH_BIAS_AVAILABLE {
            return Err(Error::Unsupported("dynamic depth bias"));
        }
        let result = self.call(|e| {
            B::set_pipeline(e, &p.native)?;
            if let Some(initial) = p.dynamic_depth_bias {
                B::depth_bias(e, initial)?;
            }
            Ok(())
        });
        if result.is_ok() {
            self.bound_dynamic_depth_bias = p.dynamic_depth_bias;
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
    pub fn set_bind_group(&mut self, slot: u32, g: &BindGroup<B>, offsets: &[u32]) -> Result<()> {
        if g.device != self.encoder.device {
            return Err(Error::WrongDevice);
        }
        let result = self.call(|e| B::set_bind_group(e, slot, &g.native, offsets));
        if result.is_ok() {
            self.encoder.stats.record_bind_group_bind();
        }
        result
    }
    pub fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &Buffer<B>,
        range: Range<u64>,
    ) -> Result<()> {
        if buffer.device != self.encoder.device {
            return Err(Error::WrongDevice);
        }
        if range.start > range.end {
            return Err(Error::InvalidRange);
        }
        let result = self.call(|e| {
            B::set_vertex_buffer(
                e,
                slot,
                &buffer.native,
                range.start,
                range.end - range.start,
            )
        });
        if result.is_ok() {
            self.encoder.stats.record_vertex_buffer_bind();
        }
        result
    }
    pub fn set_index_buffer(
        &mut self,
        buffer: &Buffer<B>,
        range: Range<u64>,
        format: IndexFormat,
    ) -> Result<()> {
        if buffer.device != self.encoder.device {
            return Err(Error::WrongDevice);
        }
        if range.start > range.end {
            return Err(Error::InvalidRange);
        }
        let result = self.call(|e| {
            B::set_index_buffer(
                e,
                &buffer.native,
                range.start,
                range.end - range.start,
                format,
            )
        });
        if result.is_ok() {
            self.encoder.stats.record_index_buffer_bind();
        }
        result
    }
    pub fn viewport(&mut self, v: Viewport) -> Result<()> {
        let result = self.call(|e| B::viewport(e, v));
        if result.is_ok() {
            self.encoder.stats.record_viewport_set();
        }
        result
    }
    pub fn scissor(&mut self, v: Rect) -> Result<()> {
        let result = self.call(|e| B::scissor(e, v));
        if result.is_ok() {
            self.encoder.stats.record_scissor_set();
        }
        result
    }
    pub fn blend_constant(&mut self, v: [f32; 4]) -> Result<()> {
        let result = self.call(|e| B::blend_constant(e, v));
        if result.is_ok() {
            self.encoder.stats.record_blend_constant_set();
        }
        result
    }
    pub fn stencil_reference(&mut self, v: u32) -> Result<()> {
        let result = self.call(|e| B::stencil_reference(e, v));
        if result.is_ok() {
            self.encoder.stats.record_stencil_reference_set();
        }
        result
    }
    /// Sets factors for a pipeline created with
    /// [`Device::raster_pipeline_with_dynamic_depth_bias`]. Binding that
    /// pipeline again restores its declared initial factors.
    pub fn set_depth_bias(&mut self, value: DynamicDepthBias) -> Result<()> {
        if !self.encoder.dynamic_depth_bias || !B::DYNAMIC_DEPTH_BIAS_AVAILABLE {
            return Err(Error::Unsupported("dynamic depth bias"));
        }
        if self.bound_dynamic_depth_bias.is_none() {
            return Err(Error::InvalidState);
        }
        if !value.is_finite() {
            return Err(Error::InvalidRange);
        }
        if value.clamp != 0.0 && !self.encoder.dynamic_depth_bias_clamp {
            return Err(Error::Unsupported("dynamic depth bias clamp"));
        }
        self.call(|e| B::depth_bias(e, value))
    }
    /// Clears selected attachments inside this pass without ending or
    /// splitting it. `rect` uses upper-left framebuffer coordinates and must
    /// be fully contained by the pass render area.
    pub fn clear_rect(
        &mut self,
        rect: Rect,
        colors: &[ColorAttachmentClear],
        depth: Option<f32>,
        stencil: Option<u32>,
    ) -> Result<()> {
        if !self.encoder.rect_attachment_clear || !B::RECT_ATTACHMENT_CLEAR_AVAILABLE {
            return Err(Error::Unsupported("raster attachment rect clear"));
        }
        if rect.width == 0
            || rect.height == 0
            || (colors.is_empty() && depth.is_none() && stencil.is_none())
        {
            return Err(Error::InvalidRange);
        }
        let Some(rect_right) = rect.x.checked_add(rect.width) else {
            return Err(Error::InvalidRange);
        };
        let Some(rect_bottom) = rect.y.checked_add(rect.height) else {
            return Err(Error::InvalidRange);
        };
        let Some(area_right) = self.render_area.x.checked_add(self.render_area.width) else {
            return Err(Error::InvalidState);
        };
        let Some(area_bottom) = self.render_area.y.checked_add(self.render_area.height) else {
            return Err(Error::InvalidState);
        };
        if rect.x < self.render_area.x
            || rect.y < self.render_area.y
            || rect_right > area_right
            || rect_bottom > area_bottom
        {
            return Err(Error::InvalidRange);
        }
        for (index, clear) in colors.iter().enumerate() {
            let Some(is_attachment) = self.color_attachment_slots.get(clear.attachment as usize)
            else {
                return Err(Error::InvalidRange);
            };
            if !is_attachment
                || colors[..index]
                    .iter()
                    .any(|prior| prior.attachment == clear.attachment)
            {
                return Err(Error::InvalidRange);
            }
        }
        if depth.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
            return Err(Error::InvalidRange);
        }
        if depth.is_some() && !self.depth_clearable || stencil.is_some() && !self.stencil_clearable
        {
            return Err(Error::InvalidState);
        }
        self.call(|encoder| B::clear_rect(encoder, rect, colors, depth, stencil))
    }
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> Result<()> {
        if !valid_u32_range(&vertices) || !valid_u32_range(&instances) {
            return Err(Error::InvalidRange);
        }
        let result = self.call(|e| B::draw(e, vertices, instances));
        if result.is_ok() {
            self.encoder.stats.record_draw();
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
        let result = self.call(|e| B::draw_indexed(e, indices, base_vertex, instances));
        if result.is_ok() {
            self.encoder.stats.record_draw_indexed();
        }
        result
    }
    pub fn finish(mut self) -> Result<()> {
        if self.encoder.debug_group_depth != self.debug_group_depth_at_min {
            if let Some(native) = self.encoder.native.as_mut() {
                B::abort_raster(native);
            }
            self.active = false;
            self.encoder.in_pass = false;
            self.encoder.poisoned = true;
            return Err(Error::InvalidState);
        }
        if self.encoder.poisoned {
            if let Some(native) = self.encoder.native.as_mut() {
                B::abort_raster(native);
            }
            self.active = false;
            self.encoder.in_pass = false;
            return Err(Error::InvalidState);
        }
        if self.debug_label_active {
            let result = B::debug_marker(
                self.encoder.native.as_mut().ok_or(Error::InvalidState)?,
                DebugMarkerScope::RasterPass,
                DebugMarkerOp::Pop,
            );
            if result.is_err() {
                if let Some(native) = self.encoder.native.as_mut() {
                    B::abort_raster(native);
                }
                self.active = false;
                self.encoder.in_pass = false;
                self.encoder.poisoned = true;
                return result;
            }
            self.encoder.debug_group_depth -= 1;
        }
        let result = B::end_raster(self.encoder.native.as_mut().ok_or(Error::InvalidState)?);
        self.active = false;
        self.encoder.in_pass = false;
        if result.is_err() {
            self.encoder.poisoned = true;
        }
        result
    }
}

/// Ranges cross the public/backend boundary as inclusive starts and exclusive
/// ends. Reject inverted input before a backend derives a count with unsigned
/// subtraction.
pub(crate) fn valid_u32_range(range: &Range<u32>) -> bool {
    range.start <= range.end
}

/// Non-indexed indirect records contain four u32 values and indexed records
/// contain five. The wrapper cannot inspect a native buffer's allocation, but
/// it can reject a record stride that cannot hold one command.
pub(crate) const fn indirect_argument_size(indexed: bool) -> u32 {
    if indexed { 20 } else { 16 }
}
impl<B: RasterBackend> Drop for RasterPass<'_, B> {
    fn drop(&mut self) {
        if self.active {
            if let Some(native) = self.encoder.native.as_mut() {
                B::abort_raster(native);
            }
            self.encoder.in_pass = false;
            self.encoder.poisoned = true;
        }
    }
}
