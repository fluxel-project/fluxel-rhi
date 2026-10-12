//! Vulkan secondary command-buffer lowering for inherited raster passes.
//!
//! Each `VulkanSecondaryEncoder` owns its command pool, so worker threads
//! never touch the primary encoder's externally synchronized pool.  A closed
//! work item retains that pool and every native object referenced by its
//! commands until the parent recorded work has passed its submission fence.

use super::*;

/// Closed secondary command buffer and its submission-lifetime owners.
#[derive(Clone)]
pub struct VulkanSecondaryWork(std::sync::Arc<SecondaryWork>);

struct SecondaryWork {
    device: std::sync::Arc<DeviceState>,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    signature: api::SecondaryRasterDesc,
    // These values deliberately have no command-time reads: retaining them
    // is what keeps every Vulkan handle valid through GPU completion.
    #[allow(dead_code)]
    buffers: Vec<VulkanBuffer>,
    #[allow(dead_code)]
    groups: Vec<VulkanBindGroup>,
    #[allow(dead_code)]
    pipelines: Vec<VulkanRasterPipeline>,
}

impl Drop for SecondaryWork {
    fn drop(&mut self) {
        // The parent encoder retains this work through its completion fence,
        // so this pool cannot be destroyed while the queue can execute
        // `command`. The retained references drop immediately afterwards.
        unsafe { self.device.device.destroy_command_pool(self.pool, None) };
    }
}

/// Worker-local state for a single secondary command buffer.
pub struct VulkanSecondaryEncoder {
    device: std::sync::Arc<DeviceState>,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    signature: api::SecondaryRasterDesc,
    layout: Option<vk::PipelineLayout>,
    layout_state: Option<super::pipeline::VulkanPipelineLayout>,
    push_constants: Option<super::pipeline::PushConstants>,
    index_bound: bool,
    buffers: Vec<VulkanBuffer>,
    groups: Vec<VulkanBindGroup>,
    pipelines: Vec<VulkanRasterPipeline>,
}

impl Drop for VulkanSecondaryEncoder {
    fn drop(&mut self) {
        if self.pool != vk::CommandPool::null() {
            unsafe { self.device.device.destroy_command_pool(self.pool, None) };
        }
    }
}

impl api::SecondaryBackend for Vulkan {
    type NativeSecondaryEncoder = VulkanSecondaryEncoder;
    type NativeSecondaryWork = VulkanSecondaryWork;

    fn begin_secondary(
        device: &VulkanDevice,
        desc: &api::SecondaryRasterDesc,
    ) -> Result<VulkanSecondaryEncoder> {
        let color_formats = native_signature(desc)?;
        let samples = sample_count(desc.samples)?;
        let depth_format = match desc.depth_stencil_format {
            Some(format) => Some(
                super::resource::format(format)
                    .ok_or(Error::Unsupported("Vulkan secondary depth format"))?,
            ),
            None => None,
        };
        let pool = unsafe {
            device.0.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(device.0.graphics_family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )
        }
        .map_err(vk_error)?;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::SECONDARY)
            .command_buffer_count(1);
        let command = match unsafe { device.0.device.allocate_command_buffers(&alloc) } {
            Ok(buffers) => buffers[0],
            Err(error) => {
                unsafe { device.0.device.destroy_command_pool(pool, None) };
                return Err(vk_error(error));
            }
        };
        let mut inheritance = vk::CommandBufferInheritanceRenderingInfo::default()
            .color_attachment_formats(&color_formats)
            .rasterization_samples(samples);
        if let Some(depth) = depth_format {
            inheritance = inheritance.depth_attachment_format(depth);
        }
        let legacy_inheritance =
            vk::CommandBufferInheritanceInfo::default().push_next(&mut inheritance);
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(
                vk::CommandBufferUsageFlags::SIMULTANEOUS_USE
                    | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
            )
            .inheritance_info(&legacy_inheritance);
        if let Err(error) = unsafe { device.0.device.begin_command_buffer(command, &begin) } {
            unsafe { device.0.device.destroy_command_pool(pool, None) };
            return Err(vk_error(error));
        }
        Ok(VulkanSecondaryEncoder {
            device: device.0.clone(),
            pool,
            command,
            signature: desc.clone(),
            layout: None,
            layout_state: None,
            push_constants: None,
            index_bound: false,
            buffers: Vec::new(),
            groups: Vec::new(),
            pipelines: Vec::new(),
        })
    }

    fn secondary_set_pipeline(
        encoder: &mut VulkanSecondaryEncoder,
        pipeline: &VulkanRasterPipeline,
    ) -> Result<()> {
        if !pipeline.matches_secondary(&encoder.signature) {
            return Err(Error::InvalidState);
        }
        unsafe {
            encoder.device.device.cmd_bind_pipeline(
                encoder.command,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.native(),
            )
        };
        encoder.layout = Some(pipeline.layout());
        encoder.layout_state = Some(pipeline.layout_state());
        encoder.push_constants = Some(pipeline.push_constants());
        encoder.pipelines.push(pipeline.clone());
        Ok(())
    }

    fn secondary_set_bind_group(
        encoder: &mut VulkanSecondaryEncoder,
        slot: u32,
        group: &VulkanBindGroup,
        offsets: &[u32],
    ) -> Result<()> {
        group.validate_dynamic_offsets(offsets)?;
        let layout = encoder.layout.ok_or(Error::InvalidState)?;
        encoder
            .layout_state
            .as_ref()
            .ok_or(Error::InvalidState)?
            .validate_group(slot, group)?;
        unsafe {
            encoder.device.device.cmd_bind_descriptor_sets(
                encoder.command,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                slot,
                &[group.native()],
                offsets,
            )
        };
        encoder.groups.push(group.clone());
        Ok(())
    }

    fn secondary_set_vertex_buffer(
        encoder: &mut VulkanSecondaryEncoder,
        slot: u32,
        buffer: &VulkanBuffer,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        if size == 0
            || !buffer.0.usage.contains(api::BufferUsage::VERTEX)
            || offset
                .checked_add(size)
                .is_none_or(|end| end > buffer.0.size)
        {
            return Err(Error::InvalidRange);
        }
        unsafe {
            encoder.device.device.cmd_bind_vertex_buffers(
                encoder.command,
                slot,
                &[buffer.0.buffer],
                &[offset],
            )
        };
        encoder.buffers.push(buffer.clone());
        Ok(())
    }

    fn secondary_set_index_buffer(
        encoder: &mut VulkanSecondaryEncoder,
        buffer: &VulkanBuffer,
        offset: u64,
        size: u64,
        format: api::IndexFormat,
    ) -> Result<()> {
        let stride = if matches!(format, api::IndexFormat::Uint16) {
            2
        } else {
            4
        };
        if size == 0
            || offset % stride != 0
            || size % stride != 0
            || !buffer.0.usage.contains(api::BufferUsage::INDEX)
            || offset
                .checked_add(size)
                .is_none_or(|end| end > buffer.0.size)
        {
            return Err(Error::InvalidRange);
        }
        unsafe {
            encoder.device.device.cmd_bind_index_buffer(
                encoder.command,
                buffer.0.buffer,
                offset,
                if matches!(format, api::IndexFormat::Uint16) {
                    vk::IndexType::UINT16
                } else {
                    vk::IndexType::UINT32
                },
            )
        };
        encoder.buffers.push(buffer.clone());
        encoder.index_bound = true;
        Ok(())
    }

    fn secondary_viewport(
        encoder: &mut VulkanSecondaryEncoder,
        viewport: api::Viewport,
    ) -> Result<()> {
        if viewport.width <= 0.0
            || viewport.height <= 0.0
            || viewport.min_depth > viewport.max_depth
        {
            return Err(Error::InvalidRange);
        }
        unsafe {
            encoder.device.device.cmd_set_viewport(
                encoder.command,
                0,
                &[vk::Viewport {
                    x: viewport.x,
                    y: viewport.y + viewport.height,
                    width: viewport.width,
                    height: -viewport.height,
                    min_depth: viewport.min_depth,
                    max_depth: viewport.max_depth,
                }],
            )
        };
        Ok(())
    }

    fn secondary_scissor(encoder: &mut VulkanSecondaryEncoder, rect: api::Rect) -> Result<()> {
        if rect.width == 0 || rect.height == 0 {
            return Err(Error::InvalidRange);
        }
        unsafe {
            encoder.device.device.cmd_set_scissor(
                encoder.command,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D {
                        x: rect.x as i32,
                        y: rect.y as i32,
                    },
                    extent: vk::Extent2D {
                        width: rect.width,
                        height: rect.height,
                    },
                }],
            )
        };
        Ok(())
    }

    fn secondary_set_push_constants(
        encoder: &mut VulkanSecondaryEncoder,
        stages: api::StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> Result<()> {
        let layout = encoder.layout.ok_or(Error::InvalidState)?;
        let native_stages = encoder
            .push_constants
            .as_ref()
            .ok_or(Error::InvalidState)?
            .validate(stages, offset, bytes)?;
        unsafe {
            encoder.device.device.cmd_push_constants(
                encoder.command,
                layout,
                native_stages,
                offset,
                bytes,
            )
        };
        Ok(())
    }

    fn secondary_draw(
        encoder: &mut VulkanSecondaryEncoder,
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    ) -> Result<()> {
        if encoder.layout.is_none()
            || vertices.start > vertices.end
            || instances.start > instances.end
        {
            return Err(Error::InvalidState);
        }
        unsafe {
            encoder.device.device.cmd_draw(
                encoder.command,
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            )
        };
        Ok(())
    }

    fn secondary_draw_indexed(
        encoder: &mut VulkanSecondaryEncoder,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) -> Result<()> {
        if encoder.layout.is_none()
            || !encoder.index_bound
            || indices.start > indices.end
            || instances.start > instances.end
        {
            return Err(Error::InvalidState);
        }
        unsafe {
            encoder.device.device.cmd_draw_indexed(
                encoder.command,
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            )
        };
        Ok(())
    }

    fn finish_secondary(mut encoder: VulkanSecondaryEncoder) -> Result<VulkanSecondaryWork> {
        if encoder.layout.is_none() {
            return Err(Error::InvalidState);
        }
        unsafe { encoder.device.device.end_command_buffer(encoder.command) }.map_err(vk_error)?;
        let work = VulkanSecondaryWork(std::sync::Arc::new(SecondaryWork {
            device: encoder.device.clone(),
            pool: encoder.pool,
            command: encoder.command,
            signature: encoder.signature.clone(),
            buffers: std::mem::take(&mut encoder.buffers),
            groups: std::mem::take(&mut encoder.groups),
            pipelines: std::mem::take(&mut encoder.pipelines),
        }));
        // The work Arc now owns this pool.  Prevent the recorder's Drop from
        // destroying it while the command buffer is still referenced.
        encoder.pool = vk::CommandPool::null();
        Ok(work)
    }

    fn execute_secondary(encoder: &mut VulkanEncoder, work: &VulkanSecondaryWork) -> Result<()> {
        if !encoder.raster || encoder.secondary_signature.as_ref() != Some(&work.0.signature) {
            return Err(Error::InvalidState);
        }
        unsafe {
            encoder
                .device
                .device
                .cmd_execute_commands(encoder.command, &[work.0.command])
        };
        encoder.secondary.push(work.clone());
        Ok(())
    }
}

pub(crate) fn native_signature(desc: &api::SecondaryRasterDesc) -> Result<Vec<vk::Format>> {
    if desc.color_formats.len() != 1 || desc.multiview_mask.is_some() {
        return Err(Error::Unsupported("Vulkan secondary raster inheritance"));
    }
    desc.color_formats
        .iter()
        .map(|format| match format {
            Some(format) => super::resource::format(*format)
                .ok_or(Error::Unsupported("Vulkan secondary color format")),
            None => Ok(vk::Format::UNDEFINED),
        })
        .collect()
}

fn sample_count(samples: u32) -> Result<vk::SampleCountFlags> {
    match samples {
        1 => Ok(vk::SampleCountFlags::TYPE_1),
        2 => Ok(vk::SampleCountFlags::TYPE_2),
        4 => Ok(vk::SampleCountFlags::TYPE_4),
        8 => Ok(vk::SampleCountFlags::TYPE_8),
        _ => Err(Error::Unsupported("Vulkan secondary sample count")),
    }
}
