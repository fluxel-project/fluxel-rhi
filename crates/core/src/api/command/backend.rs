//! Native command encoding seam.
//!
//! An encoder receives each validated operation while the caller makes it.
//! `finish` closes the backend command buffer; submission only schedules that
//! already encoded buffer on a queue.

use std::any::Any;

use crate::api::binding::{BindGroup, BindGroupIndex};
use crate::api::command::advanced::{RayTracingScopeDescriptor, RayTracingShaderTable};
use crate::api::command::copy::{
    BufferCopy, BufferTextureCopy, TextureBlit, TextureCopy, TextureResolve,
};
use crate::api::command::geometry::{Color, Rect, Viewport};
use crate::api::command::raster::RasterAttachmentClear;
use crate::api::command::record::{ComputeBegin, ImmediateWrite, RasterBegin};
use crate::api::command::{IndexFormat, ResourceUse, SecondaryRasterWork};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::external::ExternalImageCopyDescriptor;
use crate::api::pipeline::{ComputePipeline, MeshPipeline, RasterPipeline, RayTracingPipeline};
use crate::api::query::QuerySet;
use crate::api::resource::buffer::BufferBinding;
use crate::api::resource::subresource::TextureSubresourceRange;
use crate::api::resource::transfer::{ReadbackTicket, UploadJob};
use crate::api::resource::{
    AccelerationStructure, AccelerationStructureBuildMode, AccelerationStructureCopyMode, Buffer,
    BufferRange, Texture,
};

/// A backend command buffer that has already been finalized for submission.
pub trait CommandBufferBackend: Any + Send {
    /// Accesses the native type inside the backend that created this buffer.
    fn as_any(&self) -> &dyn Any;
}

fn typed_raster_unsupported() -> RhiError {
    RhiError::new(
        RhiErrorKind::Unsupported,
        "this backend does not implement typed raster encoding",
    )
}

/// One open backend command encoder.
///
/// Every method below emits into the native command list or command buffer at
/// the point the portable verb is called. Implementations may rely on their
/// native API's normal recording semantics, but each portable operation must
/// be emitted during its corresponding encoder call.
pub trait CommandEncoderBackend: Send {
    fn raster_set_mesh_pipeline(&mut self, _pipeline: &MeshPipeline) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_dispatch_mesh(
        &mut self,
        _x: u32,
        _y: u32,
        _z: u32,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_dispatch_mesh_indirect(
        &mut self,
        _arguments: &Buffer,
        _offset: u64,
        _count: Option<(&Buffer, u64, u32)>,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn acceleration_structure_build(
        &mut self,
        _destination: &AccelerationStructure,
        _scratch: &Buffer,
        _mode: AccelerationStructureBuildMode,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn acceleration_structure_copy(
        &mut self,
        _source: &AccelerationStructure,
        _destination: &AccelerationStructure,
        _mode: AccelerationStructureCopyMode,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn acceleration_structure_write_compacted_size(
        &mut self,
        _source: &AccelerationStructure,
        _destination: &Buffer,
        _destination_offset: u64,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn ray_begin(&mut self, _desc: &RayTracingScopeDescriptor) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn ray_set_pipeline(&mut self, _pipeline: &RayTracingPipeline) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn ray_set_bind_group(
        &mut self,
        _index: BindGroupIndex,
        _group: &BindGroup,
        _dynamic_offsets: &[u32],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn ray_set_immediates(&mut self, _write: &ImmediateWrite) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn ray_dispatch(
        &mut self,
        _table: &RayTracingShaderTable,
        _width: u32,
        _height: u32,
        _depth: u32,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn ray_end(&mut self) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn copy_external_image_to_texture(
        &mut self,
        _copy: &ExternalImageCopyDescriptor,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn clear_buffer(
        &mut self,
        _buffer: &Buffer,
        _range: BufferRange,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn clear_texture(
        &mut self,
        _texture: &Texture,
        _subresources: TextureSubresourceRange,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn copy_buffer(&mut self, _copy: &BufferCopy, _uses: &[ResourceUse]) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn copy_buffer_to_texture(
        &mut self,
        _copy: &BufferTextureCopy,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn copy_texture_to_buffer(
        &mut self,
        _copy: &BufferTextureCopy,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn copy_texture(&mut self, _copy: &TextureCopy, _uses: &[ResourceUse]) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn resolve_texture(
        &mut self,
        _resolve: &TextureResolve,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn blit_texture(&mut self, _blit: &TextureBlit, _uses: &[ResourceUse]) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn encode_upload(&mut self, _upload: &UploadJob, _uses: &[ResourceUse]) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn encode_readback(
        &mut self,
        _ticket: &ReadbackTicket,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn encoder_write_timestamp(
        &mut self,
        _set: &QuerySet,
        _index: u32,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn resolve_query_set(
        &mut self,
        _set: &QuerySet,
        _first_query: u32,
        _query_count: u32,
        _destination: &Buffer,
        _destination_offset: u64,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn encoder_push_debug_group(&mut self, _label: &str) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn encoder_pop_debug_group(&mut self) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn encoder_insert_debug_marker(&mut self, _label: &str) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn compute_begin(&mut self, _begin: &ComputeBegin) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn compute_set_pipeline(&mut self, _pipeline: &ComputePipeline) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn compute_set_bind_group(
        &mut self,
        _index: BindGroupIndex,
        _group: &BindGroup,
        _dynamic_offsets: &[u32],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn compute_set_immediates(&mut self, _write: &ImmediateWrite) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn compute_dispatch(
        &mut self,
        _x: u32,
        _y: u32,
        _z: u32,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn compute_dispatch_indirect(
        &mut self,
        _arguments: &Buffer,
        _offset: u64,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn compute_begin_query(&mut self, _set: &QuerySet, _index: u32) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn compute_end_query(&mut self, _set: &QuerySet, _index: u32) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn compute_write_timestamp(&mut self, _set: &QuerySet, _index: u32) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn compute_push_debug_group(&mut self, _label: &str) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn compute_pop_debug_group(&mut self) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn compute_insert_debug_marker(&mut self, _label: &str) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn compute_end(&mut self) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    /// Opens one native raster encoder.
    fn raster_begin(&mut self, _begin: &RasterBegin, _uses: &[ResourceUse]) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_clear(
        &mut self,
        _clear: &RasterAttachmentClear,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_execute_secondary(
        &mut self,
        _work: SecondaryRasterWork,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_set_pipeline(&mut self, _pipeline: &RasterPipeline) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_set_bind_group(
        &mut self,
        _index: BindGroupIndex,
        _group: &BindGroup,
        _dynamic_offsets: &[u32],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_set_vertex_buffer(&mut self, _slot: u32, _binding: &BufferBinding) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_set_index_buffer(
        &mut self,
        _binding: &BufferBinding,
        _format: IndexFormat,
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_set_viewport(&mut self, _viewport: Viewport) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_set_scissor(&mut self, _rect: Rect) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_set_blend_constant(&mut self, _color: Color) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_set_stencil_reference(&mut self, _value: u32) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_set_immediates(&mut self, _write: &ImmediateWrite) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_draw(
        &mut self,
        _vertices: core::ops::Range<u32>,
        _instances: core::ops::Range<u32>,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_draw_indexed(
        &mut self,
        _indices: core::ops::Range<u32>,
        _base_vertex: i32,
        _instances: core::ops::Range<u32>,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_draw_indirect(
        &mut self,
        _arguments: &Buffer,
        _arguments_offset: u64,
        _draw_count: u32,
        _stride: u32,
        _count: Option<(&Buffer, u64, u32)>,
        _indexed: bool,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_begin_query(&mut self, _set: &QuerySet, _index: u32) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_end_query(&mut self, _set: &QuerySet, _index: u32) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_write_timestamp(&mut self, _set: &QuerySet, _index: u32) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_push_debug_group(&mut self, _label: &str) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_pop_debug_group(&mut self) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    fn raster_insert_debug_marker(&mut self, _label: &str) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    /// Ends the currently open native raster encoder.
    fn raster_end(&mut self) -> RhiResult<()> {
        Err(typed_raster_unsupported())
    }

    /// Closes the native command buffer.
    fn finish(self: Box<Self>) -> RhiResult<Box<dyn CommandBufferBackend>>;
}
