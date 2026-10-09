//! Immediate Vulkan command encoding.
//!
//! Every recorder owns a private command pool and begins its command buffer at
//! creation.  Commands are lowered while the portable recorder receives them;
//! submission only takes the already-ended buffers and gives them to the queue.

use std::any::Any;
use std::sync::{Arc, Mutex};

use ash::vk;

use crate::api::binding::{BindGroup, BindGroupIndex};
use crate::api::command::attachment::{DepthAttachmentMode, StencilAttachmentMode};
use crate::api::command::backend::{CommandBufferBackend, CommandEncoderBackend};
use crate::api::command::geometry::{LoadOp, StoreOp};
use crate::api::command::record::{BoundGroup, BoundIndexBuffer, ImmediateWrite};
use crate::api::command::{Color, IndexFormat, Rect, ResourceUse, Viewport};
use crate::api::error::RhiResult;
use crate::api::pipeline::RasterPipeline;
use crate::api::query::QuerySet;
use crate::api::resource::Buffer;
use crate::api::resource::buffer::BufferBinding;
use crate::api::resource::buffer::BufferRange;
use crate::api::resource::subresource::TextureSubresourceRange;
use crate::api::resource::texture::Texture;
use crate::api::resource::transfer::{
    ReadbackRequest, ReadbackTicket, UploadDescriptor, UploadJob,
};
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::ffi;
use crate::backend::vulkan::platform::device::VulkanShared;

use super::compute;
use super::raster::{self, RasterScopeState};
use super::spine::{native_buffer, native_query_pool};
use super::transfer::{self, TransferRetention};
use super::typed;

pub(crate) struct VulkanNativeEncoder {
    shared: Arc<VulkanShared>,
    pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    retention: TransferRetention,
    raster: Option<RasterScopeState>,
    secondary: bool,
    secondary_scope: Option<raster::SecondaryRasterScope>,
    secondary_compatibility: Vec<raster::SecondaryRasterScope>,
    secondary_buffers: Vec<FinishedBuffer>,
    primary_begin: Option<crate::api::command::record::RasterBegin>,
    active_occlusion_query: bool,
    typed_raster: TypedRasterState,
    typed_compute: typed::TypedComputeState,
}

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

pub(crate) struct VulkanNativeCommandBuffer {
    finished: Mutex<Option<FinishedBuffer>>,
}

pub(super) struct FinishedBuffer {
    pub(super) pool: vk::CommandPool,
    pub(super) command_buffer: vk::CommandBuffer,
    pub(super) retention: TransferRetention,
    pub(super) shared: Arc<VulkanShared>,
    _secondary_compatibility: Vec<raster::SecondaryRasterScope>,
    _secondary_buffers: Vec<FinishedBuffer>,
}

impl VulkanNativeEncoder {
    pub(crate) fn new(shared: Arc<VulkanShared>) -> Result<Self, VulkanFailure> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(shared.graphics_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool =
            unsafe { shared.device.create_command_pool(&pool_info, None) }.map_err(|result| {
                VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanNativeEncoder::create_command_pool",
                ))
            })?;
        let allocation = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = match unsafe { shared.device.allocate_command_buffers(&allocation) } {
            Ok(mut buffers) => buffers
                .pop()
                .expect("one Vulkan command buffer was requested"),
            Err(result) => {
                unsafe { shared.device.destroy_command_pool(pool, None) };
                return Err(VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanNativeEncoder::allocate_command_buffers",
                )));
            }
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if let Err(result) = unsafe { shared.device.begin_command_buffer(command_buffer, &begin) } {
            unsafe { shared.device.destroy_command_pool(pool, None) };
            return Err(VulkanFailure::Native(ffi::NativeError::new(
                result,
                "VulkanNativeEncoder::begin_command_buffer",
            )));
        }
        let retention = TransferRetention::default();
        Ok(Self {
            shared,
            pool,
            command_buffer,
            retention,
            raster: None,
            secondary: false,
            secondary_scope: None,
            secondary_compatibility: Vec::new(),
            secondary_buffers: Vec::new(),
            primary_begin: None,
            active_occlusion_query: false,
            typed_raster: TypedRasterState::default(),
            typed_compute: typed::TypedComputeState::default(),
        })
    }

    pub(crate) fn new_secondary(shared: Arc<VulkanShared>) -> Result<Self, VulkanFailure> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(shared.graphics_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let pool =
            unsafe { shared.device.create_command_pool(&pool_info, None) }.map_err(|result| {
                VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanNativeEncoder::create_secondary_command_pool",
                ))
            })?;
        let allocation = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::SECONDARY)
            .command_buffer_count(1);
        let command_buffer = match unsafe { shared.device.allocate_command_buffers(&allocation) } {
            Ok(mut buffers) => buffers
                .pop()
                .expect("one Vulkan secondary command buffer was requested"),
            Err(result) => {
                unsafe { shared.device.destroy_command_pool(pool, None) };
                return Err(VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanNativeEncoder::allocate_secondary_command_buffer",
                )));
            }
        };
        let retention = TransferRetention::default();
        Ok(Self {
            shared,
            pool,
            command_buffer,
            retention,
            raster: None,
            secondary: true,
            secondary_scope: None,
            secondary_compatibility: Vec::new(),
            secondary_buffers: Vec::new(),
            primary_begin: None,
            active_occlusion_query: false,
            typed_raster: TypedRasterState::default(),
            typed_compute: typed::TypedComputeState::default(),
        })
    }

    fn active_raster_info(&self) -> Option<raster::RasterScopeInfo> {
        self.raster
            .as_ref()
            .map(RasterScopeState::info)
            .or_else(|| {
                self.secondary_scope
                    .as_ref()
                    .map(raster::SecondaryRasterScope::info)
            })
    }

    fn continuation_begin(
        begin: &crate::api::command::record::RasterBegin,
    ) -> crate::api::command::record::RasterBegin {
        let mut begin = begin.clone();
        for (_, color) in &mut begin.colors {
            color.load = LoadOp::Load;
            color.store = StoreOp::Store;
        }
        if let Some(depth_stencil) = &mut begin.depth_stencil {
            depth_stencil.depth = match depth_stencil.depth {
                Some(DepthAttachmentMode::ReadWrite { .. }) => {
                    Some(DepthAttachmentMode::ReadWrite {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    })
                }
                other => other,
            };
            depth_stencil.stencil = match depth_stencil.stencil {
                Some(StencilAttachmentMode::ReadWrite { .. }) => {
                    Some(StencilAttachmentMode::ReadWrite {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    })
                }
                other => other,
            };
        }
        begin
    }

    fn switch_primary_raster_contents(&mut self, secondary_contents: bool) -> RhiResult<()> {
        if self.secondary
            || self
                .raster
                .as_ref()
                .is_none_or(|scope| scope.uses_secondary() == secondary_contents)
        {
            return Ok(());
        }
        if self.active_occlusion_query {
            return Err(crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::Unsupported,
                "Vulkan cannot split a raster pass while an occlusion query is active",
            )
            .at("VulkanNativeEncoder::switch_primary_raster_contents"));
        }
        let scope = self.raster.take().expect("scope was checked above");
        let mut retention = raster::RasterRetention::default();
        raster::lower_raster_end(
            self.command_buffer,
            scope,
            &mut retention,
            &mut self.retention,
        )
        .map_err(|error| error.into_rhi("VulkanNativeEncoder::switch_primary_raster_contents"))?;
        self.retention.retain_raster(retention);
        let begin = Self::continuation_begin(
            self.primary_begin
                .as_ref()
                .expect("primary raster begin is set with scope"),
        );
        self.raster = Some(
            raster::lower_raster_begin(
                Arc::clone(&self.shared),
                self.command_buffer,
                &begin,
                &[],
                None,
                secondary_contents,
                &mut self.retention,
            )
            .map_err(|error| {
                error.into_rhi("VulkanNativeEncoder::switch_primary_raster_contents")
            })?,
        );
        Ok(())
    }

    /// Vulkan forbids pipeline barriers inside a legacy render pass.  Direct
    /// encoding therefore closes the current segment before each draw that has
    /// resource uses, establishes its dependencies, and reopens with LOAD.
    fn prepare_primary_raster_uses(
        &mut self,
        uses: &[ResourceUse],
        secondary_contents: bool,
    ) -> RhiResult<()> {
        if self.secondary || self.raster.is_none() {
            return Ok(());
        }
        if self.active_occlusion_query {
            return Err(crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::Unsupported,
                "Vulkan cannot split a raster pass while an occlusion query is active",
            )
            .at("VulkanNativeEncoder::prepare_primary_raster_uses"));
        }
        let scope = self.raster.take().expect("scope was checked above");
        let mut retention = raster::RasterRetention::default();
        raster::lower_raster_end(
            self.command_buffer,
            scope,
            &mut retention,
            &mut self.retention,
        )
        .map_err(|error| error.into_rhi("VulkanNativeEncoder::prepare_primary_raster_uses"))?;
        self.retention.retain_raster(retention);
        for use_ in uses {
            match use_ {
                ResourceUse::Buffer(use_) => transfer::barrier_raster_buffer(
                    &self.shared,
                    self.command_buffer,
                    &use_.buffer,
                    use_.access,
                    &mut self.retention,
                )
                .map_err(|error| {
                    error.into_rhi("VulkanNativeEncoder::prepare_primary_raster_uses")
                })?,
                ResourceUse::Texture(use_)
                    if matches!(
                        use_.intent,
                        crate::api::command::TextureUseIntent::ShaderRead
                            | crate::api::command::TextureUseIntent::ShaderReadWrite
                    ) =>
                {
                    transfer::transition_shader_texture(
                        &self.shared,
                        self.command_buffer,
                        use_,
                        vk::PipelineStageFlags::ALL_GRAPHICS,
                        &mut self.retention,
                    )
                    .map_err(|error| {
                        error.into_rhi("VulkanNativeEncoder::prepare_primary_raster_uses")
                    })?
                }
                _ => {}
            }
        }
        let begin = Self::continuation_begin(
            self.primary_begin
                .as_ref()
                .expect("primary raster begin is set with scope"),
        );
        self.raster = Some(
            raster::lower_raster_begin(
                Arc::clone(&self.shared),
                self.command_buffer,
                &begin,
                &[],
                None,
                secondary_contents,
                &mut self.retention,
            )
            .map_err(|error| error.into_rhi("VulkanNativeEncoder::prepare_primary_raster_uses"))?,
        );
        Ok(())
    }

    fn typed_draw(
        &mut self,
        range: core::ops::Range<u32>,
        instances: core::ops::Range<u32>,
        base_vertex: i32,
        indexed: bool,
        uses: &[ResourceUse],
    ) -> RhiResult<()> {
        if !self.secondary {
            self.prepare_primary_raster_uses(uses, false)?;
        }
        let scope = self.active_raster_info().ok_or_else(|| {
            VulkanFailure::Unsupported {
                what: "a Vulkan raster draw outside a render pass",
                why: "portable recording should emit RasterBegin first",
            }
            .into_rhi("VulkanNativeEncoder::typed_draw")
        })?;
        let state = &self.typed_raster;
        let pipeline = state.pipeline.as_ref().ok_or_else(|| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "typed Vulkan raster draw requires a pipeline",
            )
            .at("VulkanNativeEncoder::typed_draw")
        })?;
        if indexed != state.index.is_some() {
            return Err(crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "indexed draw state does not match the bound index buffer",
            )
            .at("VulkanNativeEncoder::typed_draw"));
        }
        let view = raster::RasterDrawView {
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
        let retention = raster::lower_raster_draw_view(
            &self.shared,
            self.command_buffer,
            &view,
            &[],
            &scope,
            &mut self.retention,
        )
        .map_err(|error| error.into_rhi("VulkanNativeEncoder::typed_draw"))?;
        self.retention.retain_raster(retention);
        Ok(())
    }
}

impl CommandEncoderBackend for VulkanNativeEncoder {
    fn compute_begin(&mut self, _: &crate::api::command::record::ComputeBegin) -> RhiResult<()> {
        Ok(())
    }
    fn compute_set_pipeline(
        &mut self,
        pipeline: &crate::api::pipeline::ComputePipeline,
    ) -> RhiResult<()> {
        self.typed_compute.set_pipeline(pipeline);
        Ok(())
    }
    fn compute_set_bind_group(
        &mut self,
        index: BindGroupIndex,
        group: &BindGroup,
        offsets: &[u32],
    ) -> RhiResult<()> {
        self.typed_compute.set_bind_group(index, group, offsets);
        Ok(())
    }
    fn compute_set_immediates(&mut self, write: &ImmediateWrite) -> RhiResult<()> {
        self.typed_compute.set_immediates(write);
        Ok(())
    }
    fn compute_dispatch(&mut self, x: u32, y: u32, z: u32, uses: &[ResourceUse]) -> RhiResult<()> {
        typed::lower_compute_dispatch(
            &self.shared,
            self.command_buffer,
            &self.typed_compute,
            (x, y, z),
            uses,
            &mut self.retention,
        )
        .map_err(|e| e.into_rhi("VulkanNativeEncoder::compute_dispatch"))
    }
    fn compute_dispatch_indirect(
        &mut self,
        arguments: &Buffer,
        offset: u64,
        uses: &[ResourceUse],
    ) -> RhiResult<()> {
        typed::lower_compute_indirect(
            &self.shared,
            self.command_buffer,
            &self.typed_compute,
            arguments,
            offset,
            uses,
            &mut self.retention,
        )
        .map_err(|e| e.into_rhi("VulkanNativeEncoder::compute_dispatch_indirect"))
    }
    fn compute_end(&mut self) -> RhiResult<()> {
        Ok(())
    }
    fn clear_buffer(
        &mut self,
        buffer: &Buffer,
        range: BufferRange,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        typed::clear_buffer(
            &self.shared,
            self.command_buffer,
            buffer,
            range,
            &mut self.retention,
        )
        .map_err(|e| e.into_rhi("VulkanNativeEncoder::clear_buffer"))
    }
    fn clear_texture(
        &mut self,
        texture: &Texture,
        range: TextureSubresourceRange,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        typed::clear_texture(
            &self.shared,
            self.command_buffer,
            texture,
            range,
            &mut self.retention,
        )
        .map_err(|e| e.into_rhi("VulkanNativeEncoder::clear_texture"))
    }
    fn copy_buffer(
        &mut self,
        copy: &crate::api::command::BufferCopy,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        typed::copy_buffer(&self.shared, self.command_buffer, copy, &mut self.retention)
            .map_err(|e| e.into_rhi("VulkanNativeEncoder::copy_buffer"))
    }
    fn copy_buffer_to_texture(
        &mut self,
        copy: &crate::api::command::BufferTextureCopy,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        typed::copy_buffer_to_texture(&self.shared, self.command_buffer, copy, &mut self.retention)
            .map_err(|e| e.into_rhi("VulkanNativeEncoder::copy_buffer_to_texture"))
    }
    fn copy_texture_to_buffer(
        &mut self,
        copy: &crate::api::command::BufferTextureCopy,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        typed::copy_texture_to_buffer(&self.shared, self.command_buffer, copy, &mut self.retention)
            .map_err(|e| e.into_rhi("VulkanNativeEncoder::copy_texture_to_buffer"))
    }
    fn copy_texture(
        &mut self,
        copy: &crate::api::command::TextureCopy,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        typed::copy_texture(&self.shared, self.command_buffer, copy, &mut self.retention)
            .map_err(|e| e.into_rhi("VulkanNativeEncoder::copy_texture"))
    }
    fn blit_texture(
        &mut self,
        blit: &crate::api::command::TextureBlit,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        typed::blit_texture(&self.shared, self.command_buffer, blit, &mut self.retention)
            .map_err(|e| e.into_rhi("VulkanNativeEncoder::blit_texture"))
    }
    fn encode_upload(&mut self, upload: &UploadJob, _: &[ResourceUse]) -> RhiResult<()> {
        typed::upload(
            &self.shared,
            self.command_buffer,
            upload,
            &mut self.retention,
        )
        .map_err(|e| e.into_rhi("VulkanNativeEncoder::encode_upload"))
    }
    fn encode_readback(&mut self, ticket: &ReadbackTicket, _: &[ResourceUse]) -> RhiResult<()> {
        typed::readback(
            &self.shared,
            self.command_buffer,
            ticket,
            &mut self.retention,
        )
        .map_err(|e| e.into_rhi("VulkanNativeEncoder::encode_readback"))
    }
    fn encoder_write_timestamp(
        &mut self,
        set: &QuerySet,
        index: u32,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        let pool = native_query_pool(set)
            .map_err(|e| e.into_rhi("VulkanNativeEncoder::encoder_write_timestamp"))?;
        unsafe {
            self.shared
                .device
                .cmd_reset_query_pool(self.command_buffer, pool, index, 1);
            self.shared.device.cmd_write_timestamp(
                self.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                pool,
                index,
            );
        }
        self.retention.query_sets.push(set.clone());
        Ok(())
    }
    fn resolve_query_set(
        &mut self,
        set: &QuerySet,
        first: u32,
        count: u32,
        destination: &Buffer,
        offset: u64,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        let pool = native_query_pool(set)
            .map_err(|e| e.into_rhi("VulkanNativeEncoder::resolve_query_set"))?;
        let buffer = native_buffer(destination)
            .map_err(|e| e.into_rhi("VulkanNativeEncoder::resolve_query_set"))?;
        let stride = super::spine::query_result_stride(set)
            .map_err(|e| e.into_rhi("VulkanNativeEncoder::resolve_query_set"))?;
        unsafe {
            self.shared.device.cmd_copy_query_pool_results(
                self.command_buffer,
                pool,
                first,
                count,
                buffer,
                offset,
                stride,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            );
        }
        self.retention.query_sets.push(set.clone());
        self.retention.buffers.push(destination.clone());
        Ok(())
    }
    fn raster_begin(
        &mut self,
        begin: &crate::api::command::record::RasterBegin,
        _uses: &[ResourceUse],
    ) -> RhiResult<()> {
        if self.raster.is_some() || self.secondary_scope.is_some() {
            return Err(crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "nested Vulkan raster scopes",
            )
            .at("VulkanNativeEncoder::raster_begin"));
        }
        if self.secondary {
            if begin.occlusion_query_set.is_some() {
                return Err(crate::api::error::RhiError::new(
                    crate::api::error::RhiErrorKind::Unsupported,
                    "Vulkan secondary raster encoders cannot contain occlusion queries",
                )
                .at("VulkanNativeEncoder::raster_begin"));
            }
            let scope = raster::secondary_scope(Arc::clone(&self.shared), begin)
                .map_err(|error| error.into_rhi("VulkanNativeEncoder::raster_begin"))?;
            let inheritance = vk::CommandBufferInheritanceInfo::default()
                .render_pass(scope.info().render_pass)
                .subpass(0)
                .framebuffer(vk::Framebuffer::null());
            let info = vk::CommandBufferBeginInfo::default()
                .flags(
                    vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
                        | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
                )
                .inheritance_info(&inheritance);
            unsafe {
                self.shared
                    .device
                    .begin_command_buffer(self.command_buffer, &info)
            }
            .map_err(|result| {
                VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanNativeEncoder::begin_secondary_command_buffer",
                ))
                .into_rhi("VulkanNativeEncoder::raster_begin")
            })?;
            self.secondary_scope = Some(scope);
            return Ok(());
        }
        // `vkCmdResetQueryPool` is forbidden inside a render pass. Reset the
        // complete fixed occlusion set before this pass begins; scope query
        // calls below only begin/end individual slots.
        if let Some(set) = &begin.occlusion_query_set {
            unsafe {
                self.shared.device.cmd_reset_query_pool(
                    self.command_buffer,
                    native_query_pool(set)
                        .map_err(|e| e.into_rhi("VulkanNativeEncoder::raster_begin"))?,
                    0,
                    set.descriptor().count,
                );
            }
            self.retention.query_sets.push(set.clone());
        }
        let mut lowered_begin = begin.clone();
        for (_, color) in &mut lowered_begin.colors {
            color.store = StoreOp::Store;
        }
        self.primary_begin = Some(lowered_begin.clone());
        self.raster = Some(
            raster::lower_raster_begin(
                Arc::clone(&self.shared),
                self.command_buffer,
                &lowered_begin,
                &[],
                None,
                false,
                &mut self.retention,
            )
            .map_err(|error| error.into_rhi("VulkanNativeEncoder::raster_begin"))?,
        );
        Ok(())
    }
    fn raster_execute_secondary(
        &mut self,
        work: crate::api::command::record::SecondaryRasterWork,
        uses: &[ResourceUse],
    ) -> RhiResult<()> {
        if self.secondary {
            return Err(crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "a Vulkan secondary command buffer cannot execute another secondary command buffer",
            )
            .at("VulkanNativeEncoder::raster_execute_secondary"));
        }
        if self.raster.is_none() {
            return Err(crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "Vulkan raster scope is not open",
            )
            .at("VulkanNativeEncoder::raster_execute_secondary"));
        }
        self.prepare_primary_raster_uses(uses, true)?;
        let child = work
            .native()
            .as_any()
            .downcast_ref::<VulkanNativeCommandBuffer>()
            .ok_or_else(|| {
                crate::api::error::RhiError::new(
                    crate::api::error::RhiErrorKind::Unsupported,
                    "Vulkan secondary raster work has no native Vulkan command buffer",
                )
                .at("VulkanNativeEncoder::raster_execute_secondary")
            })?;
        let child = child.take().ok_or_else(|| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "Vulkan secondary raster work was already submitted",
            )
            .at("VulkanNativeEncoder::raster_execute_secondary")
        })?;
        unsafe {
            self.shared
                .device
                .cmd_execute_commands(self.command_buffer, &[child.command_buffer])
        };
        self.secondary_buffers.push(child);
        Ok(())
    }
    fn raster_begin_query(&mut self, set: &QuerySet, index: u32) -> RhiResult<()> {
        unsafe {
            self.shared.device.cmd_begin_query(
                self.command_buffer,
                native_query_pool(set)
                    .map_err(|e| e.into_rhi("VulkanNativeEncoder::raster_begin_query"))?,
                index,
                vk::QueryControlFlags::empty(),
            );
        }
        self.retention.query_sets.push(set.clone());
        self.active_occlusion_query = true;
        Ok(())
    }
    fn raster_end_query(&mut self, set: &QuerySet, index: u32) -> RhiResult<()> {
        unsafe {
            self.shared.device.cmd_end_query(
                self.command_buffer,
                native_query_pool(set)
                    .map_err(|e| e.into_rhi("VulkanNativeEncoder::raster_end_query"))?,
                index,
            );
        }
        self.retention.query_sets.push(set.clone());
        self.active_occlusion_query = false;
        Ok(())
    }
    fn raster_write_timestamp(&mut self, set: &QuerySet, index: u32) -> RhiResult<()> {
        unsafe {
            self.shared.device.cmd_write_timestamp(
                self.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                native_query_pool(set)
                    .map_err(|e| e.into_rhi("VulkanNativeEncoder::raster_write_timestamp"))?,
                index,
            );
        }
        self.retention.query_sets.push(set.clone());
        Ok(())
    }
    fn raster_end(&mut self) -> RhiResult<()> {
        if self.secondary {
            let scope = self.secondary_scope.take().ok_or_else(|| {
                crate::api::error::RhiError::new(
                    crate::api::error::RhiErrorKind::InvalidUsage,
                    "Vulkan secondary raster scope is not open",
                )
                .at("VulkanNativeEncoder::raster_end")
            })?;
            unsafe { self.shared.device.end_command_buffer(self.command_buffer) }.map_err(
                |result| {
                    VulkanFailure::Native(ffi::NativeError::new(
                        result,
                        "VulkanNativeEncoder::end_secondary_command_buffer",
                    ))
                    .into_rhi("VulkanNativeEncoder::raster_end")
                },
            )?;
            self.secondary_compatibility.push(scope);
            return Ok(());
        }
        let scope = self.raster.take().ok_or_else(|| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "Vulkan raster scope is not open",
            )
            .at("VulkanNativeEncoder::raster_end")
        })?;
        self.primary_begin = None;
        let mut retention = raster::RasterRetention::default();
        raster::lower_raster_end(
            self.command_buffer,
            scope,
            &mut retention,
            &mut self.retention,
        )
        .map_err(|error| error.into_rhi("VulkanNativeEncoder::raster_end"))?;
        self.retention.retain_raster(retention);
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
        let value = BoundGroup {
            index,
            group: group.clone(),
            dynamic_offsets: dynamic_offsets.to_vec(),
        };
        if let Some(existing) = self
            .typed_raster
            .groups
            .iter_mut()
            .find(|value| value.index == index)
        {
            *existing = value;
        } else {
            self.typed_raster.groups.push(value);
        }
        Ok(())
    }
    fn raster_set_vertex_buffer(&mut self, slot: u32, binding: &BufferBinding) -> RhiResult<()> {
        if let Some(existing) = self
            .typed_raster
            .vertex_buffers
            .iter_mut()
            .find(|value| value.0 == slot)
        {
            *existing = (slot, binding.clone());
        } else {
            self.typed_raster
                .vertex_buffers
                .push((slot, binding.clone()));
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
        if let Some(existing) = self
            .typed_raster
            .immediates
            .iter_mut()
            .find(|value| value.offset == write.offset)
        {
            *existing = write.clone();
        } else {
            self.typed_raster.immediates.push(write.clone());
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
        if !self.secondary {
            self.prepare_primary_raster_uses(uses, false)?;
        }
        let scope = self.active_raster_info().ok_or_else(|| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "Vulkan raster scope is not open",
            )
            .at("VulkanNativeEncoder::raster_draw_indirect")
        })?;
        let state = &self.typed_raster;
        let pipeline = state.pipeline.as_ref().ok_or_else(|| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "typed Vulkan indirect draw requires a pipeline",
            )
            .at("VulkanNativeEncoder::raster_draw_indirect")
        })?;
        if indexed != state.index.is_some() {
            return Err(crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "indirect indexed draw state does not match the bound index buffer",
            )
            .at("VulkanNativeEncoder::raster_draw_indirect"));
        }
        let draw = raster::RasterIndirectView {
            pipeline,
            groups: &state.groups,
            vertex_buffers: &state.vertex_buffers,
            index: state.index.as_ref(),
            viewport: state.viewport,
            scissor: state.scissor,
            blend_constant: state.blend_constant,
            stencil_reference: state.stencil_reference,
            arguments,
            arguments_offset,
            draw_count,
            stride,
            count,
        };
        let retention = raster::lower_raster_indirect_view(
            &self.shared,
            self.command_buffer,
            &draw,
            &[],
            &scope,
            &mut self.retention,
        )
        .map_err(|error| error.into_rhi("VulkanNativeEncoder::raster_draw_indirect"))?;
        self.retention.retain_raster(retention);
        Ok(())
    }
    fn finish(mut self: Box<Self>) -> RhiResult<Box<dyn CommandBufferBackend>> {
        if self.raster.is_some() || self.secondary_scope.is_some() {
            return Err(VulkanFailure::Unsupported {
                what: "an unterminated Vulkan raster scope",
                why: "portable recording should emit RasterEnd before finish",
            }
            .into_rhi("VulkanNativeEncoder::finish"));
        }
        // A secondary buffer was ended with its inherited raster scope.
        // Primary buffers remain open until finish.
        if !self.secondary {
            unsafe { self.shared.device.end_command_buffer(self.command_buffer) }.map_err(
                |result| {
                    VulkanFailure::Native(ffi::NativeError::new(
                        result,
                        "VulkanNativeEncoder::end_command_buffer",
                    ))
                    .into_rhi("VulkanNativeEncoder::finish")
                },
            )?;
        }
        Ok(Box::new(VulkanNativeCommandBuffer {
            finished: Mutex::new(Some(FinishedBuffer {
                pool: self.pool,
                command_buffer: self.command_buffer,
                retention: std::mem::take(&mut self.retention),
                shared: Arc::clone(&self.shared),
                _secondary_compatibility: std::mem::take(&mut self.secondary_compatibility),
                _secondary_buffers: std::mem::take(&mut self.secondary_buffers),
            })),
        }))
    }
}

impl CommandBufferBackend for VulkanNativeCommandBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl VulkanNativeCommandBuffer {
    pub(super) fn is_available(&self) -> bool {
        self.finished
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some()
    }
    pub(super) fn take(&self) -> Option<FinishedBuffer> {
        self.finished
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }
}

impl Drop for FinishedBuffer {
    fn drop(&mut self) {
        unsafe { self.shared.device.destroy_command_pool(self.pool, None) };
    }
}
