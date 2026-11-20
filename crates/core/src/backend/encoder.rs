//! Native command recording contract implemented by backend crates.
//!
//! This module defines the lowering boundary for portable Core command and
//! raster operations. It does not schedule passes, infer hazards, or expose a
//! renderer-facing command API.

use super::Backend;
use crate::api::*;
use std::ops::Range;

/// Native command entry points, shared meanings with per-backend lowering.
/// Commands never derive ResourceUse/hazards. Non-GL backends do not use GL state.
pub trait CommandBackend: Backend {
    /// A failed begin leaves no newly issued GPU work. Backend-private
    /// allocation/state setup is rolled back before Err is returned.
    fn begin_encoder(device: &Self::NativeDevice, lane: LaneId) -> Result<Self::NativeEncoder>;
    /// Records an explicit queue boundary for an accepted residency operation.
    /// # Safety
    /// The point must come from a native residency enqueue on this device.
    unsafe fn wait_for_residency(
        _encoder: &mut Self::NativeEncoder,
        _point: &Self::NativeCompletion,
    ) -> Result<()> {
        Err(Error::Unsupported("residency queue waits"))
    }
    /// Adds a native debug annotation. `scope` identifies the currently active
    /// command/pass receiver; backends must return Unsupported when that exact
    /// scope cannot represent the operation.
    fn debug_marker(
        _: &mut Self::NativeEncoder,
        _: DebugMarkerScope,
        _: DebugMarkerOp<'_>,
    ) -> Result<()> {
        Err(Error::Unsupported("debug markers"))
    }
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
    /// Exposes freshly initialized staging bytes to the writer. A false
    /// result from the writer cancels the upload before recording a copy.
    fn write_buffer_with(
        _encoder: &mut Self::NativeEncoder,
        _dst: &Self::NativeBuffer,
        _offset: u64,
        _size: usize,
        _writer: impl FnOnce(&mut [u8]) -> bool,
    ) -> Result<()> {
        Err(Error::Unsupported("direct staging write"))
    }
    /// A convenience upload with the same idle-state boundary as write_buffer.
    fn write_texture(
        encoder: &mut Self::NativeEncoder,
        dst: TextureRegion<'_, Self>,
        layout: TexelCopyBufferLayout,
        bytes: &[u8],
    ) -> Result<()>;
    /// Exposes initialized texture staging with its actual strides.
    /// A false writer result cancels the reservation without recording a copy.
    fn write_texture_with(
        _encoder: &mut Self::NativeEncoder,
        _dst: TextureRegion<'_, Self>,
        _writer: impl FnOnce(&mut [u8], TextureUploadLayout) -> bool,
    ) -> Result<()> {
        Err(Error::Unsupported("direct texture staging write"))
    }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DebugMarkerScope {
    Encoder,
    RasterPass,
    ComputePass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DebugMarkerOp<'a> {
    Insert(&'a str),
    Push(&'a str),
    Pop,
}

/// Native raster pass lowering contract.
pub trait RasterBackend: CommandBackend {
    /// Whether this pipeline rewrites indirect argument records to adjust its
    /// vertex or instance index builtins.
    fn adjusts_indirect_vertex_indices(_pipeline: &Self::NativeRasterPipeline) -> bool {
        false
    }
    /// Whether this backend has a native dynamic depth-bias route.
    const DYNAMIC_DEPTH_BIAS_AVAILABLE: bool = false;
    /// Whether this backend can clear attachments inside an active raster pass.
    const RECT_ATTACHMENT_CLEAR_AVAILABLE: bool = false;
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
    fn stencil_reference_faces(
        _encoder: &mut Self::NativeEncoder,
        _front: u32,
        _back: u32,
    ) -> Result<()> {
        Err(Error::Unsupported("per-face stencil reference"))
    }
    /// Clears selected active-pass attachments inside `rect`.
    fn clear_rect(
        _: &mut Self::NativeEncoder,
        _: Rect,
        _: &[ColorAttachmentClear],
        _: Option<f32>,
        _: Option<u32>,
    ) -> Result<()> {
        Err(Error::Unsupported("raster attachment rect clear"))
    }
    /// Applies dynamic depth bias to a currently bound dynamic-bias pipeline.
    fn depth_bias(_: &mut Self::NativeEncoder, _: DynamicDepthBias) -> Result<()> {
        Err(Error::Unsupported("dynamic depth bias"))
    }
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
