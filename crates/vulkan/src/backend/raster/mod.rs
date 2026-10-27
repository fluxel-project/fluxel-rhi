//! Native Vulkan dynamic rendering and draw lowering.
//!
//! This module records directly into a live primary command buffer.  It owns
//! no render graph and does not infer resource state.  Texture attachments and
//! surface-image leases are retained by the encoder through submission; a
//! surface image is deliberately never converted into a durable TextureView.
//! The state machine is `idle -> raster -> idle`; all draw setters require the
//! middle state and abort closes any native scope it opened. RenderGraph owns
//! cross-pass hazards and layouts, while this module handles only attachment
//! setup that Vulkan requires inside the pass.

use super::*;

mod extras;
mod pass;
pub(super) mod resolve;
mod surface;

pub use extras::RasterExtras;
use surface::{surface_attachment, transition_surface_to_color};

impl fluxel_rhi_core::backend::RasterBackend for Vulkan {
    const DYNAMIC_DEPTH_BIAS_AVAILABLE: bool = true;
    const RECT_ATTACHMENT_CLEAR_AVAILABLE: bool = true;

    fn begin_raster(e: &mut VulkanEncoder, d: &api::RasterPassDesc<'_, Self>) -> Result<()> {
        pass::begin_raster(e, d)
    }
    fn set_pipeline(e: &mut VulkanEncoder, p: &VulkanRasterPipeline) -> Result<()> {
        if !e.raster
            || e.secondary_signature.is_some()
            || e.raster_multiview_mask != p.multiview_mask()
            || !p.compatible_read_only(e.raster_depth_read_only, e.raster_stencil_read_only)
            || !e
                .raster_signature
                .as_ref()
                .is_some_and(|s| p.matches_rendering(s))
        {
            return Err(Error::InvalidState);
        }
        let layout_state = p.layout_state();
        let bindings = prepare_groups(&layout_state, &e.raster_bound_groups)?;
        // SAFETY: the pipeline matches this pass's multiview state and is
        // retained through submission before its native handle is recorded.
        unsafe {
            e.device.device.cmd_bind_pipeline(
                e.command,
                vk::PipelineBindPoint::GRAPHICS,
                p.native(),
            )
        };
        bind_prepared(
            &e.device.device,
            e.command,
            layout_state.native(),
            &bindings,
        );
        e.mesh_active = None;
        e.raster_layout = Some(p.layout());
        e.raster_layout_state = Some(layout_state);
        e.raster_push_constants = Some(p.push_constants());
        e.raster_dynamic_depth_bias = p.has_dynamic_depth_bias();
        e.pipelines.push(p.clone());
        Ok(())
    }
    fn set_bind_group(
        e: &mut VulkanEncoder,
        index: u32,
        group: &VulkanBindGroup,
        offsets: &[u32],
    ) -> Result<()> {
        if e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
        let offsets = group.validate_dynamic_offsets(offsets)?;
        let layout = e.raster_layout.ok_or(Error::InvalidState)?;
        let layout_state = e.raster_layout_state.as_ref().ok_or(Error::InvalidState)?;
        layout_state.validate_group(index, group)?;
        let descriptor = layout_state.native_group(index, group)?;
        // SAFETY: layout compatibility and dynamic-offset alignment were
        // checked above; retaining the group preserves its descriptor set.
        unsafe {
            e.device.device.cmd_bind_descriptor_sets(
                e.command,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                index,
                &[descriptor],
                &offsets,
            );
        }
        e.raster_bound_groups
            .insert(index, (group.clone(), offsets));
        e.groups.push(group.clone());
        Ok(())
    }
    fn set_vertex_buffer(
        e: &mut VulkanEncoder,
        slot: u32,
        b: &VulkanBuffer,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        if e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
        if size == 0
            || !b.0.usage.contains(api::BufferUsage::VERTEX)
            || offset.checked_add(size).is_none_or(|end| end > b.0.size)
        {
            return Err(Error::InvalidRange);
        }
        // SAFETY: range and VERTEX usage were validated, and the cloned buffer
        // remains owned by the encoder until the GPU has executed this bind.
        unsafe {
            e.device
                .device
                .cmd_bind_vertex_buffers(e.command, slot, &[b.0.buffer], &[offset])
        };
        e.buffers.push(b.clone());
        Ok(())
    }
    fn set_index_buffer(
        e: &mut VulkanEncoder,
        b: &VulkanBuffer,
        offset: u64,
        size: u64,
        f: api::IndexFormat,
    ) -> Result<()> {
        if e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
        let stride = if matches!(f, api::IndexFormat::Uint16) {
            2
        } else {
            4
        };
        if size == 0
            || offset % stride != 0
            || size % stride != 0
            || !b.0.usage.contains(api::BufferUsage::INDEX)
            || offset.checked_add(size).is_none_or(|end| end > b.0.size)
        {
            return Err(Error::InvalidRange);
        }
        // SAFETY: index type alignment, extent and INDEX usage were validated;
        // the encoder retains the buffer for this recorded command.
        unsafe {
            e.device.device.cmd_bind_index_buffer(
                e.command,
                b.0.buffer,
                offset,
                if matches!(f, api::IndexFormat::Uint16) {
                    vk::IndexType::UINT16
                } else {
                    vk::IndexType::UINT32
                },
            )
        };
        e.buffers.push(b.clone());
        e.raster_index_bound = true;
        Ok(())
    }
    fn viewport(e: &mut VulkanEncoder, v: api::Viewport) -> Result<()> {
        if e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
        // SAFETY: this is a live primary raster command buffer. Negative height
        // intentionally maps the public top-left viewport convention to Vulkan.
        unsafe {
            e.device.device.cmd_set_viewport(
                e.command,
                0,
                &[vk::Viewport {
                    x: v.x,
                    y: v.y + v.height,
                    width: v.width,
                    height: -v.height,
                    min_depth: v.min_depth,
                    max_depth: v.max_depth,
                }],
            )
        };
        Ok(())
    }
    fn scissor(e: &mut VulkanEncoder, r: api::Rect) -> Result<()> {
        if e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
        // SAFETY: a live primary command buffer owns this dynamic state and the
        // nonzero extent was checked before it is lowered to Vulkan.
        unsafe {
            e.device.device.cmd_set_scissor(
                e.command,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D {
                        x: r.x as i32,
                        y: r.y as i32,
                    },
                    extent: vk::Extent2D {
                        width: r.width,
                        height: r.height,
                    },
                }],
            )
        };
        Ok(())
    }
    fn blend_constant(e: &mut VulkanEncoder, value: [f32; 4]) -> Result<()> {
        if !e.raster || e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
        // SAFETY: this command buffer is recording an active raster pass.
        unsafe { e.device.device.cmd_set_blend_constants(e.command, &value) };
        Ok(())
    }
    fn stencil_reference(e: &mut VulkanEncoder, value: u32) -> Result<()> {
        if !e.raster || e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
        // SAFETY: this command buffer is recording an active raster pass.
        unsafe {
            e.device.device.cmd_set_stencil_reference(
                e.command,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                value,
            )
        };
        Ok(())
    }
    fn clear_rect(
        e: &mut VulkanEncoder,
        rect: api::Rect,
        colors: &[api::ColorAttachmentClear],
        depth: Option<f32>,
        stencil: Option<u32>,
    ) -> Result<()> {
        if !e.raster || e.secondary_signature.is_some() || rect.width == 0 || rect.height == 0 {
            return Err(Error::InvalidState);
        }
        // vkCmdClearAttachments has an explicit array-layer range, while a
        // dynamic-rendering multiview pass maps views to several layers.  The
        // public command has no layer selector, so claiming that a one-layer
        // clear covers all selected views would be wrong.
        if e.raster_multiview_mask.is_some() {
            return Err(Error::Unsupported("Vulkan rect clear in multiview pass"));
        }
        let mut attachments =
            Vec::with_capacity(colors.len() + usize::from(depth.is_some() || stencil.is_some()));
        for color in colors {
            let format = e
                .raster_signature
                .as_ref()
                .and_then(|(formats, _, _)| formats.get(color.attachment as usize))
                .copied()
                .filter(|format| *format != vk::Format::UNDEFINED)
                .ok_or(Error::InvalidState)?;
            if !clear_color_matches_format(format, color.value) {
                return Err(Error::Unsupported("Vulkan rect clear format/value"));
            }
            attachments.push(vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                color_attachment: color.attachment,
                clear_value: vk::ClearValue {
                    color: match color.value {
                        api::ClearColor::Float(value) => vk::ClearColorValue { float32: value },
                        api::ClearColor::Uint(value) => vk::ClearColorValue { uint32: value },
                        api::ClearColor::Sint(value) => vk::ClearColorValue { int32: value },
                    },
                },
            });
        }
        if depth.is_some() || stencil.is_some() {
            let mut aspects = vk::ImageAspectFlags::empty();
            if depth.is_some() {
                aspects |= vk::ImageAspectFlags::DEPTH;
            }
            if stencil.is_some() {
                aspects |= vk::ImageAspectFlags::STENCIL;
            }
            attachments.push(vk::ClearAttachment {
                aspect_mask: aspects,
                color_attachment: 0,
                clear_value: vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: depth.unwrap_or(0.0),
                        stencil: stencil.unwrap_or(0),
                    },
                },
            });
        }
        let rect = vk::ClearRect {
            rect: vk::Rect2D {
                offset: vk::Offset2D {
                    x: rect.x as i32,
                    y: rect.y as i32,
                },
                extent: vk::Extent2D {
                    width: rect.width,
                    height: rect.height,
                },
            },
            base_array_layer: 0,
            layer_count: 1,
        };
        // SAFETY: Core validated the writable selected attachments and rect;
        // this is a live primary dynamic-rendering scope retaining them.
        unsafe {
            e.device
                .device
                .cmd_clear_attachments(e.command, &attachments, &[rect])
        };
        Ok(())
    }
    fn depth_bias(e: &mut VulkanEncoder, value: api::DynamicDepthBias) -> Result<()> {
        if !e.raster || e.secondary_signature.is_some() || !e.raster_dynamic_depth_bias {
            return Err(Error::InvalidState);
        }
        if !value.slope_scale.is_finite() || !value.clamp.is_finite() {
            return Err(Error::InvalidRange);
        }
        if value.clamp != 0.0 && e.device.features.depth_bias_clamp != vk::TRUE {
            return Err(Error::Unsupported("Vulkan depth bias clamp"));
        }
        // SAFETY: the active pipeline was created with DEPTH_BIAS dynamic
        // state and enabled depth bias; values have been validated above.
        unsafe {
            e.device.device.cmd_set_depth_bias(
                e.command,
                value.constant as f32,
                value.clamp,
                value.slope_scale,
            )
        };
        Ok(())
    }
    fn draw(e: &mut VulkanEncoder, v: std::ops::Range<u32>, i: std::ops::Range<u32>) -> Result<()> {
        if e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
        if !e.raster
            || e.raster_layout.is_none()
            || e.mesh_active.is_some()
            || v.start > v.end
            || i.start > i.end
        {
            return Err(Error::InvalidState);
        }
        // SAFETY: raster and pipeline state are active, and checked ranges avoid
        // arithmetic underflow when converted into Vulkan draw counts.
        unsafe {
            e.device.device.cmd_draw(
                e.command,
                v.end - v.start,
                i.end - i.start,
                v.start,
                i.start,
            )
        };
        Ok(())
    }
    fn draw_indexed(
        e: &mut VulkanEncoder,
        x: std::ops::Range<u32>,
        base: i32,
        i: std::ops::Range<u32>,
    ) -> Result<()> {
        if e.secondary_signature.is_some() {
            return Err(Error::InvalidState);
        }
        if !e.raster
            || e.raster_layout.is_none()
            || e.mesh_active.is_some()
            || !e.raster_index_bound
            || x.start > x.end
            || i.start > i.end
        {
            return Err(Error::InvalidState);
        }
        // SAFETY: an index buffer and compatible pipeline are bound; validated
        // ranges are converted directly to the native indexed draw parameters.
        unsafe {
            e.device.device.cmd_draw_indexed(
                e.command,
                x.end - x.start,
                i.end - i.start,
                x.start,
                base,
                i.start,
            )
        };
        Ok(())
    }
    fn end_raster(e: &mut VulkanEncoder) -> Result<()> {
        if !e.raster || e.active_occlusion.is_some() {
            return Err(Error::InvalidState);
        }
        let image = e.color_surface_image.take();
        // SAFETY: `e.raster` proves this primary command buffer has a matching
        // dynamic-rendering begin which has not yet been ended.
        unsafe { e.device.device.cmd_end_rendering(e.command) };
        resolve::finish(e, &e.pending_resolves);
        e.pending_resolves.clear();
        super::query::end_pass_timestamps(e);
        if let Some(image) = image {
            capture::finish(e, image);
        }
        e.raster = false;
        e.raster_index_bound = false;
        e.raster_dynamic_depth_bias = false;
        e.raster_multiview_mask = None;
        e.raster_bound_groups.clear();
        e.raster_layout = None;
        e.raster_layout_state = None;
        e.mesh_active = None;
        e.raster_signature = None;
        e.raster_push_constants = None;
        e.secondary_signature = None;
        Ok(())
    }
    fn abort_raster(e: &mut VulkanEncoder) {
        if e.raster {
            if let Some((set, slot)) = e.active_occlusion.take() {
                // SAFETY: the encoder owns the still-active query and command
                // buffer; ending it makes abort leave native state balanced.
                unsafe { e.device.device.cmd_end_query(e.command, set.pool(), slot) };
            }
            // SAFETY: the branch is entered only for a begun rendering scope.
            unsafe { e.device.device.cmd_end_rendering(e.command) };
            e.pending_resolves.clear();
            super::query::end_pass_timestamps(e);
            e.raster = false;
            e.raster_index_bound = false;
            e.raster_dynamic_depth_bias = false;
            e.raster_multiview_mask = None;
            e.color_surface_image = None;
            e.raster_bound_groups.clear();
            e.raster_layout = None;
            e.raster_layout_state = None;
            e.mesh_active = None;
            e.raster_signature = None;
            e.raster_push_constants = None;
            e.secondary_signature = None;
        }
    }
}

/// Vulkan's clear union must use the numeric member matching the attachment.
/// Non-integer color formats use the float member (including UNORM/SNORM and
/// SRGB); the supported Core integer formats are enumerated explicitly.
fn clear_color_matches_format(format: vk::Format, value: api::ClearColor) -> bool {
    let uint = matches!(
        format,
        vk::Format::R8_UINT
            | vk::Format::R8G8_UINT
            | vk::Format::R8G8B8A8_UINT
            | vk::Format::R16_UINT
            | vk::Format::R16G16_UINT
            | vk::Format::R16G16B16A16_UINT
            | vk::Format::R32_UINT
            | vk::Format::R32G32_UINT
            | vk::Format::R32G32B32A32_UINT
            | vk::Format::A2B10G10R10_UINT_PACK32
            | vk::Format::R64_UINT
            | vk::Format::R64G64_UINT
    );
    let sint = matches!(
        format,
        vk::Format::R8_SINT
            | vk::Format::R8G8_SINT
            | vk::Format::R8G8B8A8_SINT
            | vk::Format::R16_SINT
            | vk::Format::R16G16_SINT
            | vk::Format::R16G16B16A16_SINT
            | vk::Format::R32_SINT
            | vk::Format::R32G32_SINT
            | vk::Format::R32G32B32A32_SINT
    );
    matches!(value, api::ClearColor::Uint(_)) && uint
        || matches!(value, api::ClearColor::Sint(_)) && sint
        || matches!(value, api::ClearColor::Float(_)) && !uint && !sint
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_clear_color_value_matches_attachment_numeric_kind() {
        let float = api::ClearColor::Float([0.0; 4]);
        let uint = api::ClearColor::Uint([0; 4]);
        let sint = api::ClearColor::Sint([0; 4]);
        assert!(clear_color_matches_format(
            vk::Format::R8G8B8A8_UNORM,
            float
        ));
        assert!(clear_color_matches_format(vk::Format::R32_UINT, uint));
        assert!(clear_color_matches_format(vk::Format::R32_SINT, sint));
        assert!(!clear_color_matches_format(vk::Format::R32_UINT, float));
        assert!(!clear_color_matches_format(vk::Format::R32_SINT, uint));
        assert!(!clear_color_matches_format(
            vk::Format::R8G8B8A8_UNORM,
            sint
        ));
    }
}

// Rebind live public groups when a pipeline selects another native layout.
// The group and selected layout owners remain retained by the encoder/work.
pub(super) fn prepare_groups(
    layout: &super::pipeline::VulkanPipelineLayout,
    groups: &std::collections::BTreeMap<u32, (VulkanBindGroup, Vec<u32>)>,
) -> Result<Vec<(u32, vk::DescriptorSet, Vec<u32>)>> {
    let mut prepared = Vec::new();
    for (&slot, (group, offsets)) in groups {
        if layout.validate_group(slot, group).is_ok() {
            prepared.push((slot, layout.native_group(slot, group)?, offsets.clone()));
        }
    }
    Ok(prepared)
}

pub(super) fn bind_prepared(
    device: &ash::Device,
    command: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    groups: &[(u32, vk::DescriptorSet, Vec<u32>)],
) {
    for (slot, set, offsets) in groups {
        // SAFETY: preparation checked each group's layout and retained its set;
        // offsets were validated at the original binding call.
        unsafe {
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                *slot,
                &[*set],
                offsets,
            );
        }
    }
}
