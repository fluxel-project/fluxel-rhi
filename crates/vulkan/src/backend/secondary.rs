//! Vulkan secondary command-buffer lowering for inherited raster passes.
//!
//! Each `VulkanSecondaryEncoder` exclusively leases a command-pool page, so worker threads
//! never touch the primary encoder's externally synchronized pool.  A closed
//! work item retains that pool and every native object referenced by its
//! commands until the parent recorded work has passed its submission fence.
//! This module validates inheritance compatibility but does not schedule work,
//! derive resource barriers, or permit a secondary buffer outside its matching
//! primary dynamic-rendering scope.

use super::*;

/// Closed secondary command buffer and its submission-lifetime owners.
#[derive(Clone)]
pub struct VulkanSecondaryWork(std::sync::Arc<SecondaryWork>);

impl VulkanSecondaryWork {
    pub(crate) fn groups(&self) -> &[VulkanBindGroup] {
        &self.0.groups
    }
}

struct SecondaryWork {
    // Keeps DeviceState alive until Drop returns the native page to its cache.
    #[allow(dead_code)]
    device: std::sync::Arc<DeviceState>,
    pool: Option<super::command_pool::CommandPoolLease>,
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
        // The parent encoder retains this work through its completion fence;
        // only then can the exclusive page be reset and returned.
        drop(self.pool.take());
    }
}

/// Worker-local state for a single secondary command buffer.
pub struct VulkanSecondaryEncoder {
    device: std::sync::Arc<DeviceState>,
    pool: Option<super::command_pool::CommandPoolLease>,
    command: vk::CommandBuffer,
    signature: api::SecondaryRasterDesc,
    layout: Option<vk::PipelineLayout>,
    bound_groups: std::collections::BTreeMap<u32, (VulkanBindGroup, Vec<u32>)>,
    layout_state: Option<super::pipeline::VulkanPipelineLayout>,
    push_constants: Option<super::pipeline::PushConstants>,
    index_bound: bool,
    buffers: Vec<VulkanBuffer>,
    groups: Vec<VulkanBindGroup>,
    pipelines: Vec<VulkanRasterPipeline>,
}

impl Drop for VulkanSecondaryEncoder {
    fn drop(&mut self) {
        // An unclosed secondary was never handed to a parent, so returning its
        // exclusive page is immediately safe.
        drop(self.pool.take());
    }
}

impl api::SecondaryBackend for Vulkan {
    type NativeSecondaryEncoder = VulkanSecondaryEncoder;
    type NativeSecondaryWork = VulkanSecondaryWork;

    fn begin_secondary(
        device: &VulkanDevice,
        desc: &api::SecondaryRasterDesc,
    ) -> Result<VulkanSecondaryEncoder> {
        if let Some(mask) = desc.multiview_mask {
            if !device.0.multiview {
                return Err(Error::Unsupported("Vulkan multiview"));
            }
            if mask == 0 || 32 - mask.leading_zeros() > device.0.max_multiview_views {
                return Err(Error::InvalidRange);
            }
        }
        let color_formats = native_signature(&device.0, desc)?;
        let samples = sample_count(desc.samples)?;
        let depth_format = match desc.depth_stencil_format {
            Some(format) => Some(
                super::resource::native_format(&device.0, format)
                    .ok_or(Error::Unsupported("Vulkan secondary depth format"))?,
            ),
            None => None,
        };
        let mut pool = device
            .0
            .command_pools
            .lease(device.0.graphics_family, vk::CommandBufferLevel::SECONDARY)?;
        let command = pool.allocate()?;
        let mut inheritance = vk::CommandBufferInheritanceRenderingInfo::default()
            .color_attachment_formats(&color_formats)
            .rasterization_samples(samples)
            .view_mask(desc.multiview_mask.unwrap_or(0));
        if let Some(depth) = depth_format {
            if desc.depth_stencil_format != Some(api::TextureFormat::Stencil8) {
                inheritance = inheritance.depth_attachment_format(depth);
            }
            if matches!(
                desc.depth_stencil_format,
                Some(
                    api::TextureFormat::Depth24PlusStencil8
                        | api::TextureFormat::Depth32FloatStencil8
                        | api::TextureFormat::Stencil8
                )
            ) {
                inheritance = inheritance.stencil_attachment_format(depth);
            }
        }
        let legacy_inheritance =
            vk::CommandBufferInheritanceInfo::default().push_next(&mut inheritance);
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(
                vk::CommandBufferUsageFlags::SIMULTANEOUS_USE
                    | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
            )
            .inheritance_info(&legacy_inheritance);
        // SAFETY: `command` came from this exclusive page; inheritance describes
        // the checked secondary signature and is borrowed only for this call.
        if let Err(error) = unsafe { device.0.device.begin_command_buffer(command, &begin) } {
            return Err(vk_error(error));
        }
        // SAFETY: the secondary is recording and owns these dynamic defaults.
        unsafe {
            device.0.device.cmd_set_blend_constants(command, &[0.0; 4]);
            device.0.device.cmd_set_stencil_reference(
                command,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                0,
            );
        }
        Ok(VulkanSecondaryEncoder {
            device: device.0.clone(),
            pool: Some(pool),
            command,
            signature: desc.clone(),
            layout: None,
            bound_groups: Default::default(),
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
        let layout_state = pipeline.layout_state();
        let bindings = super::raster::prepare_groups(&layout_state, &encoder.bound_groups)?;
        // SAFETY: the pipeline was checked against this encoder's inheritance
        // signature and is retained until the parent submission completes.
        unsafe {
            encoder.device.device.cmd_bind_pipeline(
                encoder.command,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.native(),
            )
        };
        super::raster::bind_prepared(
            &encoder.device.device,
            encoder.command,
            layout_state.native(),
            &bindings,
        );
        encoder.layout = Some(pipeline.layout());
        encoder.layout_state = Some(layout_state);
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
        let offsets = group.validate_dynamic_offsets(offsets)?;
        let layout = encoder.layout.ok_or(Error::InvalidState)?;
        let layout_state = encoder.layout_state.as_ref().ok_or(Error::InvalidState)?;
        layout_state.validate_group(slot, group)?;
        let descriptor = layout_state.native_group(slot, group)?;
        // SAFETY: layout/group compatibility and offsets are checked above;
        // retaining the group keeps the native descriptor set alive.
        unsafe {
            encoder.device.device.cmd_bind_descriptor_sets(
                encoder.command,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                slot,
                &[descriptor],
                &offsets,
            )
        };
        encoder.bound_groups.insert(slot, (group.clone(), offsets));
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
        // SAFETY: VERTEX usage and byte range were checked and the buffer is
        // retained by the closed work item before it can be submitted.
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
        // SAFETY: checked index alignment, range and usage match the selected
        // native index type; retaining the buffer protects its handle.
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
        // SAFETY: viewport values were range checked; negative height performs
        // the documented top-left-to-Vulkan coordinate conversion.
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
        // SAFETY: the live secondary command buffer owns this dynamic state and
        // the nonzero scissor extent was checked above.
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
        // SAFETY: the active pipeline supplied this layout and validation
        // checked the pushed byte range and stage visibility.
        for (stage, native_offset) in native_stages {
            unsafe {
                encoder.device.device.cmd_push_constants(
                    encoder.command,
                    layout,
                    stage,
                    native_offset,
                    bytes,
                );
            }
        }
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
        // SAFETY: an inherited-compatible pipeline is bound and the checked
        // ranges produce non-underflowing Vulkan draw counts.
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
        // SAFETY: an index buffer is bound, a compatible pipeline is active,
        // and checked ranges map directly to native draw parameters.
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
        // SAFETY: this encoder exclusively owns its begun secondary command
        // buffer and has recorded only commands admitted by the invariants.
        unsafe { encoder.device.device.end_command_buffer(encoder.command) }.map_err(vk_error)?;
        let work = VulkanSecondaryWork(std::sync::Arc::new(SecondaryWork {
            device: encoder.device.clone(),
            pool: encoder.pool.take(),
            command: encoder.command,
            signature: encoder.signature.clone(),
            buffers: std::mem::take(&mut encoder.buffers),
            groups: std::mem::take(&mut encoder.groups),
            pipelines: std::mem::take(&mut encoder.pipelines),
        }));
        Ok(work)
    }

    fn execute_secondary(encoder: &mut VulkanEncoder, work: &VulkanSecondaryWork) -> Result<()> {
        if !encoder.raster || encoder.secondary_signature.as_ref() != Some(&work.0.signature) {
            return Err(Error::InvalidState);
        }
        // SAFETY: the primary pass opted into secondary contents and checked
        // the exact inheritance signature; retaining `work` protects its pool.
        if work.0.pipelines.iter().any(|p| {
            !p.compatible_read_only(
                encoder.raster_depth_read_only,
                encoder.raster_stencil_read_only,
            )
        }) {
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

pub(crate) fn native_signature(
    device: &DeviceState,
    desc: &api::SecondaryRasterDesc,
) -> Result<Vec<vk::Format>> {
    if desc.multiview_mask == Some(0) {
        return Err(Error::InvalidRange);
    }
    // SAFETY: the device retains the instance and selected physical device.
    let max_colors = unsafe {
        device
            .instance
            .instance
            .get_physical_device_properties(device.physical)
    }
    .limits
    .max_color_attachments;
    if desc.color_formats.len() > max_colors as usize {
        return Err(Error::Unsupported("Vulkan maxColorAttachments"));
    }
    desc.color_formats
        .iter()
        .map(|format| match format {
            Some(format) => super::resource::native_format(device, *format)
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
        16 => Ok(vk::SampleCountFlags::TYPE_16),
        32 => Ok(vk::SampleCountFlags::TYPE_32),
        64 => Ok(vk::SampleCountFlags::TYPE_64),
        _ => Err(Error::Unsupported("Vulkan secondary sample count")),
    }
}
