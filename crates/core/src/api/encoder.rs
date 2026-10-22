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

/// Native command entry points, shared meanings with per-backend lowering.
/// Commands never derive ResourceUse/hazards. Non-GL backends do not use GL state.
pub trait CommandBackend: Backend {
    /// A failed begin leaves no newly issued GPU work. Backend-private
    /// allocation/state setup is rolled back before Err is returned.
    fn begin_encoder(device: &Self::NativeDevice, lane: LaneId) -> Result<Self::NativeEncoder>;
    fn transition(
        encoder: &mut Self::NativeEncoder,
        ops: &[ResourceTransition<'_, Self>],
    ) -> Result<()>;
    /// Copies from a buffer already in CopySrc state to one in CopyDst state.
    /// The caller supplies transitions; recording this operation does not
    /// change either resource's state after the copy.
    fn copy_buffer(
        encoder: &mut Self::NativeEncoder,
        src: &Self::NativeBuffer,
        src_offset: u64,
        dst: &Self::NativeBuffer,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<()>;
    /// Copies between texture regions already in CopySrc and CopyDst states.
    fn copy_texture(
        encoder: &mut Self::NativeEncoder,
        src: TextureRegion<'_, Self>,
        dst: TextureRegion<'_, Self>,
    ) -> Result<()>;
    /// Copies from a CopySrc buffer into a CopyDst texture region.
    fn buffer_to_texture(
        encoder: &mut Self::NativeEncoder,
        src: &Self::NativeBuffer,
        layout: TexelCopyBufferLayout,
        dst: TextureRegion<'_, Self>,
    ) -> Result<()>;
    /// Copies from a CopySrc texture region into a buffer usable as CopyDst.
    /// A readback buffer created with `COPY_DST` may have a fixed native copy
    /// destination state; otherwise the caller supplies required transitions.
    fn texture_to_buffer(
        encoder: &mut Self::NativeEncoder,
        src: TextureRegion<'_, Self>,
        dst: &Self::NativeBuffer,
        layout: TexelCopyBufferLayout,
    ) -> Result<()>;
    /// A convenience upload with a backend-managed temporary copy state.
    /// The destination must be in its backend's documented idle state before
    /// the call and returns to that state afterwards.
    fn write_buffer(
        encoder: &mut Self::NativeEncoder,
        dst: &Self::NativeBuffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()>;
    /// A convenience upload with the same idle-state boundary as write_buffer.
    fn write_texture(
        encoder: &mut Self::NativeEncoder,
        dst: TextureRegion<'_, Self>,
        layout: TexelCopyBufferLayout,
        bytes: &[u8],
    ) -> Result<()>;
    // Capability-specific transfer variants have defaults; an unsupported
    // Backend never needs to fake a native blit or independent clear operation.
    fn resolve(
        _: &mut Self::NativeEncoder,
        _: TextureRegion<'_, Self>,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Err(Error::Unsupported("resolve"))
    }
    /// Blits between texture regions already transitioned to `CopySrc` and
    /// `CopyDst`. The operation preserves those caller-selected states.
    fn blit(
        _: &mut Self::NativeEncoder,
        _: TextureRegion<'_, Self>,
        _: TextureRegion<'_, Self>,
        _: FilterMode,
    ) -> Result<()> {
        Err(Error::Unsupported("blit"))
    }
    fn clear_buffer(
        _: &mut Self::NativeEncoder,
        _: &Self::NativeBuffer,
        _: Range<u64>,
        _: u32,
    ) -> Result<()> {
        Err(Error::Unsupported("clear_buffer"))
    }
    /// Standalone whole-subresource clear; defaults to unsupported.
    fn clear_texture(
        _: &mut Self::NativeEncoder,
        _: &Self::NativeTexture,
        _: Range<u32>,
        _: Range<u32>,
        _: TextureClearValue,
    ) -> Result<()> {
        Err(Error::Unsupported("clear_texture"))
    }
}

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
}
impl<B: CommandBackend> CommandEncoder<B> {
    pub(crate) fn new(device: &Device<B>, lane: LaneId) -> Result<Self> {
        if lane.device != device.id {
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
        self.call(|e| B::copy_buffer(e, &src.native, src_offset, &dst.native, dst_offset, bytes))
    }
    pub fn copy_texture(
        &mut self,
        src: TextureRegion<'_, B>,
        dst: TextureRegion<'_, B>,
    ) -> Result<()> {
        if src.texture.device != self.device || dst.texture.device != self.device {
            return Err(Error::WrongDevice);
        }
        self.call(|e| B::copy_texture(e, src, dst))
    }
    pub fn resolve(&mut self, src: TextureRegion<'_, B>, dst: TextureRegion<'_, B>) -> Result<()> {
        if src.texture.device != self.device || dst.texture.device != self.device {
            return Err(Error::WrongDevice);
        }
        self.call(|e| B::resolve(e, src, dst))
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
        self.call(|e| B::blit(e, src, dst, filter))
    }
    pub fn clear_buffer(&mut self, dst: &Buffer<B>, range: Range<u64>, value: u32) -> Result<()> {
        if dst.device != self.device {
            return Err(Error::WrongDevice);
        }
        self.call(|e| B::clear_buffer(e, &dst.native, range, value))
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
        self.call(|e| B::clear_texture(e, &dst.native, mips, layers, value))
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
        self.call(|e| B::buffer_to_texture(e, &src.native, layout, dst))
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
        self.call(|e| B::texture_to_buffer(e, src, &dst.native, layout))
    }
    pub fn write_buffer(&mut self, dst: &Buffer<B>, offset: u64, bytes: &[u8]) -> Result<()> {
        if dst.device != self.device {
            return Err(Error::WrongDevice);
        }
        self.call(|e| B::write_buffer(e, &dst.native, offset, bytes))
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
        self.call(|e| B::write_texture(e, dst, layout, bytes))
    }
}

pub trait RasterBackend: CommandBackend {
    fn begin_raster(
        encoder: &mut Self::NativeEncoder,
        desc: &RasterPassDesc<'_, Self>,
    ) -> Result<()>;
    fn set_pipeline(
        encoder: &mut Self::NativeEncoder,
        pipeline: &Self::NativeRasterPipeline,
    ) -> Result<()>;
    fn set_bind_group(
        encoder: &mut Self::NativeEncoder,
        slot: u32,
        group: &Self::NativeBindGroup,
        offsets: &[u32],
    ) -> Result<()>;
    fn set_vertex_buffer(
        encoder: &mut Self::NativeEncoder,
        slot: u32,
        buffer: &Self::NativeBuffer,
        offset: u64,
        size: u64,
    ) -> Result<()>;
    fn set_index_buffer(
        encoder: &mut Self::NativeEncoder,
        buffer: &Self::NativeBuffer,
        offset: u64,
        size: u64,
        format: IndexFormat,
    ) -> Result<()>;
    fn viewport(encoder: &mut Self::NativeEncoder, viewport: Viewport) -> Result<()>;
    fn scissor(encoder: &mut Self::NativeEncoder, rect: Rect) -> Result<()>;
    fn blend_constant(encoder: &mut Self::NativeEncoder, value: [f32; 4]) -> Result<()>;
    fn stencil_reference(encoder: &mut Self::NativeEncoder, value: u32) -> Result<()>;
    fn draw(
        encoder: &mut Self::NativeEncoder,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) -> Result<()>;
    fn draw_indexed(
        encoder: &mut Self::NativeEncoder,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> Result<()>;
    fn end_raster(encoder: &mut Self::NativeEncoder) -> Result<()>;
    fn abort_raster(encoder: &mut Self::NativeEncoder);
}

pub struct RasterPass<'a, B: RasterBackend> {
    pub(crate) encoder: &'a mut CommandEncoder<B>,
    pub(crate) active: bool,
    pub(crate) secondary_inheritance: Option<SecondaryRasterDesc>,
}
impl<B: RasterBackend> CommandEncoder<B> {
    pub fn begin_raster<'a>(
        &'a mut self,
        desc: &RasterPassDesc<'_, B>,
    ) -> Result<RasterPass<'a, B>> {
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
        let result = B::begin_raster(self.ready()?, desc);
        if let Err(e) = result {
            self.poisoned = true;
            return Err(e);
        }
        self.in_pass = true;
        Ok(RasterPass {
            encoder: self,
            active: true,
            secondary_inheritance: desc.secondary_inheritance.clone(),
        })
    }
}
impl<B: RasterBackend> RasterPass<'_, B> {
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
        self.call(|e| B::set_pipeline(e, &p.native))
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
        self.call(|e| B::set_bind_group(e, slot, &g.native, offsets))
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
        self.call(|e| {
            B::set_vertex_buffer(
                e,
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
        if buffer.device != self.encoder.device {
            return Err(Error::WrongDevice);
        }
        if range.start > range.end {
            return Err(Error::InvalidRange);
        }
        self.call(|e| {
            B::set_index_buffer(
                e,
                &buffer.native,
                range.start,
                range.end - range.start,
                format,
            )
        })
    }
    pub fn viewport(&mut self, v: Viewport) -> Result<()> {
        self.call(|e| B::viewport(e, v))
    }
    pub fn scissor(&mut self, v: Rect) -> Result<()> {
        self.call(|e| B::scissor(e, v))
    }
    pub fn blend_constant(&mut self, v: [f32; 4]) -> Result<()> {
        self.call(|e| B::blend_constant(e, v))
    }
    pub fn stencil_reference(&mut self, v: u32) -> Result<()> {
        self.call(|e| B::stencil_reference(e, v))
    }
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> Result<()> {
        if !valid_u32_range(&vertices) || !valid_u32_range(&instances) {
            return Err(Error::InvalidRange);
        }
        self.call(|e| B::draw(e, vertices, instances))
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
        self.call(|e| B::draw_indexed(e, indices, base_vertex, instances))
    }
    pub fn finish(mut self) -> Result<()> {
        if self.encoder.poisoned {
            if let Some(native) = self.encoder.native.as_mut() {
                B::abort_raster(native);
            }
            self.active = false;
            self.encoder.in_pass = false;
            return Err(Error::InvalidState);
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
