//! Immediate Direct3D 12 command encoding.
//!
//! A portable recorder still validates every operation and owns its resource-use
//! summary.  This module is the native half: each accepted command is lowered
//! into one open `ID3D12GraphicsCommandList` immediately, and `finish` closes
//! that list.  Submission consequently schedules closed lists only; it never
//! walks portable draw, dispatch, or copy payloads after encoding.

use std::any::Any;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, ID3D12CommandAllocator, ID3D12CommandList, ID3D12Device,
    ID3D12GraphicsCommandList, ID3D12PipelineState,
};

use crate::api::binding::{BindGroup, BindGroupIndex};
use crate::api::command::backend::{CommandBufferBackend, CommandEncoderBackend};
use crate::api::command::record::{BoundGroup, BoundIndexBuffer, ImmediateWrite};
use crate::api::command::{Color, IndexFormat, Rect, ResourceUse, Viewport};
use crate::api::error::RhiResult;
use crate::api::pipeline::{ComputePipeline, RasterPipeline};
use crate::api::platform::DeviceLossInfo;
use crate::api::resource::Texture;
use crate::api::resource::buffer::{Buffer, BufferBinding, BufferRange};
use crate::api::resource::subresource::TextureSubresourceRange;
use crate::api::resource::transfer::{ReadbackTicket, UploadJob};
use crate::backend::dx12::failure::{Dx12Failure, ref_native};
use crate::backend::dx12::platform::device::Dx12LossState;

use super::blit::lower_texture_blit;
use super::compute::{
    ComputeDispatchView, ComputeIndirectView, lower_compute_dispatch_view,
    lower_compute_indirect_view,
};
use super::copy::lower_buffer_copy;
use super::query::{
    begin as lower_query_begin, end as lower_query_end, resolve_parts as lower_query_resolve_parts,
};
use super::raster::{
    RasterDrawView, RasterScopeState, lower_raster_begin, lower_raster_clear,
    lower_raster_draw_view, lower_raster_end, lower_raster_indirect_view,
    lower_secondary_raster_work,
};
use super::spine::{lower_debug_marker, lower_debug_pop, lower_debug_push};
use super::transfer::{
    CommittedBatch, lower_buffer_clear, lower_buffer_texture_copy, lower_readback,
    lower_texture_clear, lower_texture_copy, lower_upload,
};

/// An open D3D12 direct list belonging to exactly one portable recorder.
pub(crate) struct Dx12NativeEncoder {
    device: ID3D12Device,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    committed: CommittedBatch,
    raster: Option<RasterScopeState>,
    secondary: bool,
    secondary_heaps: Option<(ID3D12DescriptorHeap, ID3D12DescriptorHeap)>,
    typed_raster: TypedRasterState,
    typed_compute: TypedComputeState,
    loss: Arc<Dx12LossState>,
}

/// Mutable graphics binding state owned by one native command encoder.  It is
/// never copied into a draw packet: typed setters update it in place and the
/// draw verb consumes it immediately.
struct TypedRasterState {
    pipeline: Option<RasterPipeline>,
    groups: Vec<BoundGroup>,
    vertex_buffers: Vec<(u32, BufferBinding)>,
    index: Option<BoundIndexBuffer>,
    viewport: Option<Viewport>,
    scissor: Option<Rect>,
    blend_constant: Color,
    stencil_reference: u32,
    immediates: Vec<ImmediateWrite>,
}

impl Default for TypedRasterState {
    fn default() -> Self {
        Self {
            pipeline: None,
            groups: Vec::new(),
            vertex_buffers: Vec::new(),
            index: None,
            viewport: None,
            scissor: None,
            blend_constant: Color::new(0.0, 0.0, 0.0, 0.0),
            stencil_reference: 0,
            immediates: Vec::new(),
        }
    }
}

struct TypedComputeState {
    pipeline: Option<ComputePipeline>,
    groups: Vec<BoundGroup>,
    immediates: Vec<ImmediateWrite>,
}

impl Default for TypedComputeState {
    fn default() -> Self {
        Self {
            pipeline: None,
            groups: Vec::new(),
            immediates: Vec::new(),
        }
    }
}

/// A closed D3D12 list.  The mutex makes the portable, borrowed submission
/// plan capable of consuming a list exactly once.
pub(crate) struct Dx12NativeCommandBuffer {
    finished: Mutex<Option<FinishedList>>,
}

pub(super) struct FinishedList {
    pub(super) allocator: ID3D12CommandAllocator,
    pub(super) list: ID3D12GraphicsCommandList,
    pub(super) committed: CommittedBatch,
    pub(super) secondary: bool,
    pub(super) secondary_heaps: Option<(ID3D12DescriptorHeap, ID3D12DescriptorHeap)>,
}

impl Dx12NativeEncoder {
    fn typed_draw(
        &mut self,
        range: core::ops::Range<u32>,
        instances: core::ops::Range<u32>,
        base_vertex: i32,
        indexed: bool,
        uses: &[ResourceUse],
    ) -> RhiResult<()> {
        let scope = self.raster.as_ref().ok_or_else(|| {
            self.error(
                outside_scope("a raster draw"),
                "Dx12CommandEncoder::typed_draw",
            )
        })?;
        let state = &self.typed_raster;
        let pipeline = state.pipeline.as_ref().ok_or_else(|| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "typed raster draw requires a pipeline",
            )
            .at("Dx12CommandEncoder::typed_draw")
        })?;
        let view = RasterDrawView {
            pipeline,
            groups: &state.groups,
            vertex_buffers: &state.vertex_buffers,
            index: state.index.as_ref(),
            viewport: state.viewport,
            scissor: state.scissor,
            blend_constant: state.blend_constant,
            stencil_reference: state.stencil_reference,
            range,
            instances,
            base_vertex,
            immediates: &state.immediates,
        };
        lower_raster_draw_view(&self.list, &view, uses, scope, &mut self.committed, indexed)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::typed_draw"))
    }
    fn typed_compute_dispatch(
        &mut self,
        workgroups: (u32, u32, u32),
        uses: &[ResourceUse],
    ) -> RhiResult<()> {
        let state = &self.typed_compute;
        let pipeline = state.pipeline.as_ref().ok_or_else(|| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "typed compute dispatch requires a pipeline",
            )
            .at("Dx12CommandEncoder::typed_compute_dispatch")
        })?;
        lower_compute_dispatch_view(
            &self.list,
            &ComputeDispatchView {
                pipeline,
                groups: &state.groups,
                immediates: &state.immediates,
                workgroups,
            },
            uses,
            &mut self.committed,
        )
        .map_err(|failure| self.error(failure, "Dx12CommandEncoder::typed_compute_dispatch"))
    }
    pub(crate) fn new(
        device: &ID3D12Device,
        loss: Arc<Dx12LossState>,
    ) -> Result<Self, Dx12Failure> {
        // A recorder gets a private allocator/list pair.  It is intentionally
        // not borrowed from the submission ring: it can remain open while other
        // recorders finish and submit, and its allocator is retained with this
        // exact closed list until the queue fence passes.
        let (allocator, list) = unsafe {
            let allocator = device
                .CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(|error| ref_native(&error))?;
            let list = device
                .CreateCommandList::<_, _, ID3D12GraphicsCommandList>(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &allocator,
                    None::<&ID3D12PipelineState>,
                )
                .map_err(|error| ref_native(&error))?;
            (allocator, list)
        };
        Ok(Self {
            device: device.clone(),
            allocator,
            list,
            committed: CommittedBatch::pending(),
            raster: None,
            secondary: false,
            secondary_heaps: None,
            typed_raster: TypedRasterState::default(),
            typed_compute: TypedComputeState::default(),
            loss,
        })
    }

    pub(crate) fn new_secondary(
        device: &ID3D12Device,
        loss: Arc<Dx12LossState>,
    ) -> Result<Self, Dx12Failure> {
        let mut encoder = Self::new(device, loss)?;
        // Replace the direct list with a bundle before any command is recorded.
        let allocator = unsafe {
            device
                .CreateCommandAllocator::<ID3D12CommandAllocator>(
                    windows::Win32::Graphics::Direct3D12::D3D12_COMMAND_LIST_TYPE_BUNDLE,
                )
                .map_err(|error| ref_native(&error))?
        };
        let list = unsafe {
            device
                .CreateCommandList::<_, _, ID3D12GraphicsCommandList>(
                    0,
                    windows::Win32::Graphics::Direct3D12::D3D12_COMMAND_LIST_TYPE_BUNDLE,
                    &allocator,
                    None::<&ID3D12PipelineState>,
                )
                .map_err(|error| ref_native(&error))?
        };
        encoder.allocator = allocator;
        encoder.list = list;
        encoder.secondary = true;
        Ok(encoder)
    }

    fn error(&self, failure: Dx12Failure, operation: &'static str) -> crate::api::error::RhiError {
        if failure.is_terminal() {
            self.loss.mark_lost(DeviceLossInfo::new(format!(
                "Direct3D 12 command encoding failed: {}",
                failure.message()
            )));
        }
        failure.into_rhi(operation)
    }
}

impl CommandEncoderBackend for Dx12NativeEncoder {
    fn clear_buffer(
        &mut self,
        buffer: &Buffer,
        range: BufferRange,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        lower_buffer_clear(&self.device, &self.list, buffer, range, &mut self.committed)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::clear_buffer"))
    }
    fn clear_texture(
        &mut self,
        texture: &Texture,
        range: TextureSubresourceRange,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        lower_texture_clear(
            &self.device,
            &self.list,
            texture,
            range,
            &mut self.committed,
        )
        .map_err(|failure| self.error(failure, "Dx12CommandEncoder::clear_texture"))
    }
    fn copy_buffer(
        &mut self,
        copy: &crate::api::command::copy::BufferCopy,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        lower_buffer_copy(&self.list, copy)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::copy_buffer"))
    }
    fn copy_buffer_to_texture(
        &mut self,
        copy: &crate::api::command::copy::BufferTextureCopy,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        lower_buffer_texture_copy(&self.device, &self.list, copy, true)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::copy_buffer_to_texture"))
    }
    fn copy_texture_to_buffer(
        &mut self,
        copy: &crate::api::command::copy::BufferTextureCopy,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        lower_buffer_texture_copy(&self.device, &self.list, copy, false)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::copy_texture_to_buffer"))
    }
    fn copy_texture(
        &mut self,
        copy: &crate::api::command::copy::TextureCopy,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        lower_texture_copy(&self.list, copy)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::copy_texture"))
    }
    fn blit_texture(
        &mut self,
        blit: &crate::api::command::copy::TextureBlit,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        lower_texture_blit(&self.device, &self.list, blit, &mut self.committed)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::blit_texture"))
    }
    fn encode_upload(&mut self, upload: &UploadJob, _uses: &[ResourceUse]) -> RhiResult<()> {
        lower_upload(&self.device, &self.list, upload, &mut self.committed)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::encode_upload"))
    }
    fn encode_readback(&mut self, ticket: &ReadbackTicket, _uses: &[ResourceUse]) -> RhiResult<()> {
        lower_readback(&self.device, &self.list, ticket, &mut self.committed)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::encode_readback"))
    }
    fn encoder_write_timestamp(
        &mut self,
        set: &crate::api::query::QuerySet,
        index: u32,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        let _ = (set, index);
        Err(crate::api::error::RhiError::new(
            crate::api::error::RhiErrorKind::Unsupported,
            "DX12 does not expose encoder timestamp writes in its capability facts",
        )
        .at("Dx12CommandEncoder::encoder_write_timestamp"))
    }
    fn resolve_query_set(
        &mut self,
        set: &crate::api::query::QuerySet,
        first_query: u32,
        query_count: u32,
        destination: &Buffer,
        destination_offset: u64,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        lower_query_resolve_parts(
            &self.list,
            set,
            first_query,
            query_count,
            destination,
            destination_offset,
            &mut self.committed,
        )
        .map_err(|failure| self.error(failure, "Dx12CommandEncoder::resolve_query_set"))
    }
    fn encoder_push_debug_group(&mut self, label: &str) -> RhiResult<()> {
        self.raster_push_debug_group(label)
    }
    fn encoder_pop_debug_group(&mut self) -> RhiResult<()> {
        self.raster_pop_debug_group()
    }
    fn encoder_insert_debug_marker(&mut self, label: &str) -> RhiResult<()> {
        self.raster_insert_debug_marker(label)
    }

    fn compute_begin(
        &mut self,
        _begin: &crate::api::command::record::ComputeBegin,
    ) -> RhiResult<()> {
        Ok(())
    }
    fn compute_set_pipeline(&mut self, pipeline: &ComputePipeline) -> RhiResult<()> {
        self.typed_compute.pipeline = Some(pipeline.clone());
        self.typed_compute.immediates.clear();
        Ok(())
    }
    fn compute_set_bind_group(
        &mut self,
        index: BindGroupIndex,
        group: &BindGroup,
        dynamic_offsets: &[u32],
    ) -> RhiResult<()> {
        let value = BoundGroup {
            index,
            group: group.clone(),
            dynamic_offsets: dynamic_offsets.to_vec(),
        };
        if let Some(existing) = self
            .typed_compute
            .groups
            .iter_mut()
            .find(|known| known.index == index)
        {
            *existing = value;
        } else {
            self.typed_compute.groups.push(value);
        }
        Ok(())
    }
    fn compute_set_immediates(&mut self, write: &ImmediateWrite) -> RhiResult<()> {
        if let Some(existing) = self
            .typed_compute
            .immediates
            .iter_mut()
            .find(|known| known.offset == write.offset)
        {
            *existing = write.clone();
        } else {
            self.typed_compute.immediates.push(write.clone());
        }
        Ok(())
    }
    fn compute_dispatch(&mut self, x: u32, y: u32, z: u32, uses: &[ResourceUse]) -> RhiResult<()> {
        self.typed_compute_dispatch((x, y, z), uses)
    }
    fn compute_dispatch_indirect(
        &mut self,
        arguments: &Buffer,
        offset: u64,
        uses: &[ResourceUse],
    ) -> RhiResult<()> {
        let state = &self.typed_compute;
        let pipeline = state.pipeline.as_ref().ok_or_else(|| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "typed indirect compute dispatch requires a pipeline",
            )
            .at("Dx12CommandEncoder::compute_dispatch_indirect")
        })?;
        lower_compute_indirect_view(
            &self.device,
            &self.list,
            &ComputeIndirectView {
                pipeline,
                groups: &state.groups,
                immediates: &state.immediates,
                arguments,
                arguments_offset: offset,
            },
            uses,
            &mut self.committed,
        )
        .map_err(|failure| self.error(failure, "Dx12CommandEncoder::compute_dispatch_indirect"))
    }
    fn compute_begin_query(
        &mut self,
        set: &crate::api::query::QuerySet,
        index: u32,
    ) -> RhiResult<()> {
        lower_query_begin(&self.list, set, index, &mut self.committed)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::compute_begin_query"))
    }
    fn compute_end_query(
        &mut self,
        set: &crate::api::query::QuerySet,
        index: u32,
    ) -> RhiResult<()> {
        lower_query_end(&self.list, set, index, &mut self.committed)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::compute_end_query"))
    }
    fn compute_write_timestamp(
        &mut self,
        set: &crate::api::query::QuerySet,
        index: u32,
    ) -> RhiResult<()> {
        self.encoder_write_timestamp(set, index, &[])
    }
    fn compute_push_debug_group(&mut self, label: &str) -> RhiResult<()> {
        self.raster_push_debug_group(label)
    }
    fn compute_pop_debug_group(&mut self) -> RhiResult<()> {
        self.raster_pop_debug_group()
    }
    fn compute_insert_debug_marker(&mut self, label: &str) -> RhiResult<()> {
        self.raster_insert_debug_marker(label)
    }
    fn compute_end(&mut self) -> RhiResult<()> {
        Ok(())
    }

    fn raster_begin(
        &mut self,
        begin: &crate::api::command::record::RasterBegin,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        self.raster = Some(
            lower_raster_begin(&self.device, &self.list, begin, &mut self.committed)
                .map_err(|failure| self.error(failure, "Dx12CommandEncoder::raster_begin"))?,
        );
        Ok(())
    }

    fn raster_clear(
        &mut self,
        clear: &crate::api::command::RasterAttachmentClear,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        let scope = self.raster.as_ref().ok_or_else(|| {
            self.error(
                outside_scope("a raster attachment clear"),
                "Dx12CommandEncoder::raster_clear",
            )
        })?;
        lower_raster_clear(&self.device, &self.list, clear, scope, &mut self.committed)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::raster_clear"))
    }

    fn raster_end(&mut self) -> RhiResult<()> {
        let scope = self.raster.take().ok_or_else(|| {
            self.error(
                Dx12Failure::Unsupported {
                    what: "a raster-scope end without a scope",
                    why: "the portable recorder never emits it",
                },
                "Dx12CommandEncoder::raster_end",
            )
        })?;
        lower_raster_end(&self.list, scope, &mut self.committed);
        Ok(())
    }

    fn raster_begin_query(
        &mut self,
        set: &crate::api::query::QuerySet,
        index: u32,
    ) -> RhiResult<()> {
        lower_query_begin(&self.list, set, index, &mut self.committed)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::raster_begin_query"))
    }
    fn raster_end_query(&mut self, set: &crate::api::query::QuerySet, index: u32) -> RhiResult<()> {
        lower_query_end(&self.list, set, index, &mut self.committed)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::raster_end_query"))
    }
    fn raster_write_timestamp(
        &mut self,
        set: &crate::api::query::QuerySet,
        index: u32,
    ) -> RhiResult<()> {
        self.encoder_write_timestamp(set, index, &[])
    }
    fn raster_push_debug_group(&mut self, label: &str) -> RhiResult<()> {
        lower_debug_push(
            &self.list,
            &crate::api::identity::Label(Some(label.to_owned())),
        );
        Ok(())
    }
    fn raster_pop_debug_group(&mut self) -> RhiResult<()> {
        lower_debug_pop(&self.list);
        Ok(())
    }
    fn raster_insert_debug_marker(&mut self, label: &str) -> RhiResult<()> {
        lower_debug_marker(
            &self.list,
            &crate::api::identity::Label(Some(label.to_owned())),
        );
        Ok(())
    }
    fn raster_set_pipeline(&mut self, pipeline: &RasterPipeline) -> RhiResult<()> {
        self.typed_raster.pipeline = Some(pipeline.clone());
        self.typed_raster.immediates.clear();
        Ok(())
    }

    fn raster_set_bind_group(
        &mut self,
        index: BindGroupIndex,
        group: &BindGroup,
        dynamic_offsets: &[u32],
    ) -> RhiResult<()> {
        let state = &mut self.typed_raster;
        let value = BoundGroup {
            index,
            group: group.clone(),
            dynamic_offsets: dynamic_offsets.to_vec(),
        };
        if let Some(existing) = state.groups.iter_mut().find(|bound| bound.index == index) {
            *existing = value;
        } else {
            state.groups.push(value);
        }
        Ok(())
    }

    fn raster_set_vertex_buffer(&mut self, slot: u32, binding: &BufferBinding) -> RhiResult<()> {
        let state = &mut self.typed_raster;
        if let Some(existing) = state
            .vertex_buffers
            .iter_mut()
            .find(|(known, _)| *known == slot)
        {
            *existing = (slot, binding.clone());
        } else {
            state.vertex_buffers.push((slot, binding.clone()));
        }
        Ok(())
    }

    fn raster_set_index_buffer(
        &mut self,
        binding: &BufferBinding,
        format: IndexFormat,
    ) -> RhiResult<()> {
        self.typed_raster.index = Some(BoundIndexBuffer {
            binding: binding.clone(),
            format,
        });
        Ok(())
    }

    fn raster_set_viewport(&mut self, viewport: Viewport) -> RhiResult<()> {
        self.typed_raster.viewport = Some(viewport);
        Ok(())
    }
    fn raster_set_scissor(&mut self, rect: Rect) -> RhiResult<()> {
        self.typed_raster.scissor = Some(rect);
        Ok(())
    }
    fn raster_set_blend_constant(&mut self, color: Color) -> RhiResult<()> {
        self.typed_raster.blend_constant = color;
        Ok(())
    }
    fn raster_set_stencil_reference(&mut self, value: u32) -> RhiResult<()> {
        self.typed_raster.stencil_reference = value;
        Ok(())
    }
    fn raster_set_immediates(&mut self, write: &ImmediateWrite) -> RhiResult<()> {
        let state = &mut self.typed_raster;
        if let Some(existing) = state
            .immediates
            .iter_mut()
            .find(|known| known.offset == write.offset)
        {
            *existing = write.clone();
        } else {
            state.immediates.push(write.clone());
        }
        Ok(())
    }

    fn raster_draw(
        &mut self,
        vertices: core::ops::Range<u32>,
        instances: core::ops::Range<u32>,
        uses: &[ResourceUse],
    ) -> RhiResult<()> {
        self.typed_draw(vertices, instances, 0, false, uses)
    }
    fn raster_draw_indexed(
        &mut self,
        indices: core::ops::Range<u32>,
        base_vertex: i32,
        instances: core::ops::Range<u32>,
        uses: &[ResourceUse],
    ) -> RhiResult<()> {
        self.typed_draw(indices, instances, base_vertex, true, uses)
    }
    fn raster_draw_indirect(
        &mut self,
        arguments: &Buffer,
        arguments_offset: u64,
        draw_count: u32,
        stride: u32,
        count: Option<(&Buffer, u64, u32)>,
        indexed: bool,
        uses: &[ResourceUse],
    ) -> RhiResult<()> {
        let scope = self.raster.as_ref().ok_or_else(|| {
            self.error(
                outside_scope("an indirect raster draw"),
                "Dx12CommandEncoder::raster_draw_indirect",
            )
        })?;
        let state = &self.typed_raster;
        let pipeline = state.pipeline.as_ref().ok_or_else(|| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "typed indirect raster draw requires a pipeline",
            )
            .at("Dx12CommandEncoder::raster_draw_indirect")
        })?;
        if indexed != state.index.is_some() {
            return Err(crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "indirect indexed draw state does not match the bound index buffer",
            )
            .at("Dx12CommandEncoder::raster_draw_indirect"));
        }
        let view = RasterDrawView {
            pipeline,
            groups: &state.groups,
            vertex_buffers: &state.vertex_buffers,
            index: state.index.as_ref(),
            viewport: state.viewport,
            scissor: state.scissor,
            blend_constant: state.blend_constant,
            stencil_reference: state.stencil_reference,
            range: 0..0,
            instances: 0..0,
            base_vertex: 0,
            immediates: &state.immediates,
        };
        lower_raster_indirect_view(
            &self.device,
            &self.list,
            &view,
            arguments,
            arguments_offset,
            draw_count,
            stride,
            count,
            uses,
            scope,
            &mut self.committed,
        )
        .map_err(|failure| self.error(failure, "Dx12CommandEncoder::raster_draw_indirect"))
    }
    fn raster_execute_secondary(
        &mut self,
        work: crate::api::command::record::SecondaryRasterWork,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        let scope = self.raster.as_ref().ok_or_else(|| {
            self.error(
                outside_scope("secondary raster work"),
                "Dx12CommandEncoder::raster_execute_secondary",
            )
        })?;
        lower_secondary_raster_work(&self.device, &self.list, &work, scope, &mut self.committed)
            .map_err(|failure| self.error(failure, "Dx12CommandEncoder::raster_execute_secondary"))
    }

    fn finish(mut self: Box<Self>) -> RhiResult<Box<dyn CommandBufferBackend>> {
        if self.raster.is_some() {
            return Err(self.error(
                Dx12Failure::Unsupported {
                    what: "an unterminated raster scope",
                    why: "the portable recorder never emits it",
                },
                "Dx12CommandEncoder::finish",
            ));
        }
        unsafe { self.list.Close() }
            .map_err(|error| self.error(ref_native(&error), "Dx12CommandEncoder::finish"))?;
        Ok(Box::new(Dx12NativeCommandBuffer {
            finished: Mutex::new(Some(FinishedList {
                allocator: self.allocator.clone(),
                list: self.list.clone(),
                committed: std::mem::replace(&mut self.committed, CommittedBatch::pending()),
                secondary: self.secondary,
                secondary_heaps: self.secondary_heaps.take(),
            })),
        }))
    }
}

impl CommandBufferBackend for Dx12NativeCommandBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Dx12NativeCommandBuffer {
    pub(crate) fn is_available(&self) -> bool {
        self.finished
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
    }

    pub(super) fn take(&self) -> Option<FinishedList> {
        self.finished
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }
}

impl FinishedList {
    pub(super) fn command_list(&self) -> ID3D12CommandList {
        ID3D12CommandList::from(self.list.clone())
    }
}

fn outside_scope(what: &'static str) -> Dx12Failure {
    Dx12Failure::Unsupported {
        what,
        why: "the portable recorder never emits it outside a raster scope",
    }
}
