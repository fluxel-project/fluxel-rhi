//! Typed compute and transfer command helpers.
//!
//! This module keeps the mutable state of the portable compute encoder beside
//! Vulkan's immediate lowering path.  It deliberately has no recorded-command
//! queue: a dispatch borrows the encoder state and records it into the open
//! `VkCommandBuffer` immediately.

use ash::vk;

use crate::api::binding::{BindGroup, BindGroupIndex};
use crate::api::command::ResourceUse;
use crate::api::command::copy::{BufferCopy, BufferTextureCopy, TextureBlit, TextureCopy};
use crate::api::command::record::{BoundGroup, ImmediateWrite};
use crate::api::pipeline::ComputePipeline;
use crate::api::resource::buffer::{Buffer, BufferRange};
use crate::api::resource::subresource::TextureSubresourceRange;
use crate::api::resource::texture::Texture;
use crate::api::resource::transfer::{
    ReadbackRequest, ReadbackTicket, UploadDescriptor, UploadJob,
};
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::platform::device::VulkanShared;

use super::transfer::TransferRetention;
use super::{compute, transfer};

/// Mutable binding state for one open typed compute scope.
///
/// This is encoder-owned state. It is consulted and
/// lowered at each dispatch, then remains available for the next dispatch as
/// Vulkan command encoder state normally would.
#[derive(Default)]
pub(super) struct TypedComputeState {
    pipeline: Option<ComputePipeline>,
    groups: Vec<BoundGroup>,
    immediates: Vec<ImmediateWrite>,
}

impl TypedComputeState {
    pub(super) fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        self.pipeline = Some(pipeline.clone());
        // Pipeline layout changes invalidate the old push-constant state.
        self.immediates.clear();
    }

    pub(super) fn set_bind_group(
        &mut self,
        index: BindGroupIndex,
        group: &BindGroup,
        dynamic_offsets: &[u32],
    ) {
        let replacement = BoundGroup {
            index,
            group: group.clone(),
            dynamic_offsets: dynamic_offsets.to_vec(),
        };
        if let Some(existing) = self.groups.iter_mut().find(|bound| bound.index == index) {
            *existing = replacement;
        } else {
            self.groups.push(replacement);
        }
    }

    pub(super) fn set_immediates(&mut self, write: &ImmediateWrite) {
        if let Some(existing) = self
            .immediates
            .iter_mut()
            .find(|existing| existing.offset == write.offset)
        {
            *existing = write.clone();
        } else {
            self.immediates.push(write.clone());
        }
    }

    fn pipeline(&self, operation: &'static str) -> Result<&ComputePipeline, VulkanFailure> {
        self.pipeline.as_ref().ok_or(VulkanFailure::Unsupported {
            what: operation,
            why: "a typed compute dispatch requires a pipeline",
        })
    }
}

/// Immediately lowers a typed direct compute dispatch.
///
/// The lowerer borrows the encoder-owned state.  It retains only the portable
/// handles required to keep native Vulkan objects alive until the submission
/// fence completes.
pub(super) fn lower_compute_dispatch(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    state: &TypedComputeState,
    workgroups: (u32, u32, u32),
    uses: &[ResourceUse],
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let compute = compute::lower_compute_dispatch_view(
        shared,
        command_buffer,
        &compute::ComputeDispatchView {
            pipeline: state.pipeline("a typed Vulkan compute dispatch")?,
            groups: &state.groups,
            workgroups,
            immediates: &state.immediates,
        },
        uses,
        retention,
    )?;
    retention.retain_compute(compute);
    Ok(())
}

/// Immediately lowers a typed indirect compute dispatch.
pub(super) fn lower_compute_indirect(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    state: &TypedComputeState,
    arguments: &Buffer,
    arguments_offset: u64,
    uses: &[ResourceUse],
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let compute = compute::lower_compute_indirect_view(
        shared,
        command_buffer,
        &compute::ComputeIndirectView {
            pipeline: state.pipeline("a typed Vulkan indirect compute dispatch")?,
            groups: &state.groups,
            arguments,
            arguments_offset,
            immediates: &state.immediates,
        },
        uses,
        retention,
    )?;
    retention.retain_compute(compute);
    retention.buffers.push(arguments.clone());
    Ok(())
}

pub(super) fn clear_buffer(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    buffer: &Buffer,
    range: BufferRange,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    transfer::lower_clear_buffer(shared, command_buffer, buffer, range, retention)
}

pub(super) fn clear_texture(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    texture: &Texture,
    range: TextureSubresourceRange,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    transfer::lower_clear_texture(shared, command_buffer, texture, range, retention)
}

pub(super) fn copy_buffer(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    copy: &BufferCopy,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    transfer::lower_buffer_copy(shared, command_buffer, copy, retention)
}

pub(super) fn copy_buffer_to_texture(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    copy: &BufferTextureCopy,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    transfer::lower_buffer_texture_copy(shared, command_buffer, copy, true, retention)
}

pub(super) fn copy_texture_to_buffer(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    copy: &BufferTextureCopy,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    transfer::lower_buffer_texture_copy(shared, command_buffer, copy, false, retention)
}

pub(super) fn copy_texture(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    copy: &TextureCopy,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    transfer::lower_texture_copy(shared, command_buffer, copy, retention)
}

pub(super) fn blit_texture(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    blit: &TextureBlit,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    transfer::lower_texture_blit(shared, command_buffer, blit, retention)
}

pub(super) fn upload(
    shared: &std::sync::Arc<VulkanShared>,
    command_buffer: vk::CommandBuffer,
    upload: &UploadJob,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    match upload.descriptor() {
        UploadDescriptor::Buffer(_) => {
            transfer::lower_upload(shared, command_buffer, upload, retention)
        }
        UploadDescriptor::Texture(_) => {
            transfer::lower_texture_upload(shared, command_buffer, upload, retention)
        }
        _ => Err(VulkanFailure::Unsupported {
            what: "an unsupported Vulkan upload descriptor",
            why: "this Vulkan backend does not implement the newer portable upload variant",
        }),
    }
}

pub(super) fn readback(
    shared: &std::sync::Arc<VulkanShared>,
    command_buffer: vk::CommandBuffer,
    ticket: &ReadbackTicket,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    match ticket.request() {
        ReadbackRequest::Buffer { .. } => {
            transfer::lower_readback(shared, command_buffer, ticket, retention)
        }
        ReadbackRequest::Texture { .. } => {
            transfer::lower_texture_readback(shared, command_buffer, ticket, retention)
        }
        ReadbackRequest::Frame { .. } => {
            transfer::lower_frame_readback(shared, command_buffer, ticket, retention)
        }
        _ => Err(VulkanFailure::Unsupported {
            what: "an unsupported Vulkan readback request",
            why: "this Vulkan backend does not implement the newer portable readback variant",
        }),
    }
}
