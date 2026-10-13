//! Vulkan dynamic-rendering raster-pipeline creation.
//!
//! This module builds one dynamic-rendering pipeline from a Core descriptor.
//! The pipeline retains its layout and stores the render-target signature used
//! by secondary-command validation; encoders keep the pipeline alive until the
//! submission fence proves native command execution complete.

use super::super::{DeviceState, VulkanDevice, VulkanShader, vk_error};
use super::layout::{PushConstants, VulkanPipelineLayout};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::{ffi::CString, sync::Arc};

#[derive(Clone)]
pub struct VulkanRasterPipeline(Arc<Raster>);
impl VulkanRasterPipeline {
    pub(crate) fn native(&self) -> vk::Pipeline {
        self.0.pipeline
    }
    pub(crate) fn layout(&self) -> vk::PipelineLayout {
        self.0.layout.native()
    }
    pub(crate) fn layout_state(&self) -> VulkanPipelineLayout {
        self.0.layout.clone()
    }
    pub(crate) fn push_constants(&self) -> PushConstants {
        self.0.layout.push_constants()
    }
    pub(crate) fn multiview_mask(&self) -> Option<u32> {
        self.0.multiview_mask
    }
    pub(crate) fn matches_secondary(&self, signature: &api::SecondaryRasterDesc) -> bool {
        self.0.color_formats.as_ref() == signature.color_formats.as_slice()
            && self.0.depth_format == signature.depth_stencil_format
            && self.0.samples == signature.samples
            && self.0.multiview_mask == signature.multiview_mask
    }
}
struct Raster {
    device: Arc<DeviceState>,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: VulkanPipelineLayout,
    color_formats: Arc<[Option<api::TextureFormat>]>,
    depth_format: Option<api::TextureFormat>,
    samples: u32,
    multiview_mask: Option<u32>,
}
impl Drop for Raster {
    fn drop(&mut self) {
        // SAFETY: encoders retain the Raster Arc through fence retirement, so
        // no in-flight command buffer can still reference this pipeline.
        unsafe { self.device.device.destroy_pipeline(self.pipeline, None) }
    }
}

pub(crate) fn create_raster(
    device: &VulkanDevice,
    desc: &api::RasterPipelineDesc<super::super::Vulkan>,
) -> Result<VulkanRasterPipeline> {
    if !desc.vertex.constants.is_empty()
        || desc
            .fragment
            .as_ref()
            .is_some_and(|stage| !stage.constants.is_empty())
    {
        return Err(Error::Unsupported("Vulkan raster specialization constants"));
    }
    if desc.cache.is_some() {
        return Err(Error::Unsupported("Vulkan raster pipeline cache"));
    }
    let vertex_name = CString::new(desc.vertex.entry.as_str())
        .map_err(|_| Error::Shader("shader entry contains NUL".into()))?;
    let fragment_name = desc
        .fragment
        .as_ref()
        .map(|s| CString::new(s.entry.as_str()))
        .transpose()
        .map_err(|_| Error::Shader("shader entry contains NUL".into()))?;
    // SAFETY: `Device::raster_pipeline` checked every stage and layout DeviceId
    // before this backend future; the shader borrow is needed only for creation.
    let vertex_shader = unsafe { desc.vertex.module.native_ref() };
    let vs = stage(vertex_shader, &vertex_name, vk::ShaderStageFlags::VERTEX)?;
    let fs = desc
        .fragment
        .as_ref()
        .map(|s| {
            // SAFETY: the public raster-pipeline constructor checked this
            // optional fragment module against the device before this call.
            let shader = unsafe { s.module.native_ref() };
            stage(
                shader,
                fragment_name.as_ref().expect("fragment name exists"),
                vk::ShaderStageFlags::FRAGMENT,
            )
        })
        .transpose()?;
    let mut stages = vec![vs];
    if let Some(x) = fs {
        stages.push(x)
    };
    let attributes: Vec<_> = desc
        .vertex_buffers
        .iter()
        .enumerate()
        .flat_map(|(b, v)| {
            v.attributes
                .iter()
                .map(move |a| vk::VertexInputAttributeDescription {
                    location: a.location,
                    binding: b as u32,
                    format: vformat(a.format).unwrap_or(vk::Format::UNDEFINED),
                    offset: a.offset as u32,
                })
        })
        .collect();
    if attributes.iter().any(|a| a.format == vk::Format::UNDEFINED) {
        return Err(Error::Unsupported("Vulkan vertex format"));
    };
    let bindings: Vec<_> = desc
        .vertex_buffers
        .iter()
        .enumerate()
        .map(|(i, v)| vk::VertexInputBindingDescription {
            binding: i as u32,
            stride: v.stride as u32,
            input_rate: if matches!(v.step_mode, api::VertexStepMode::Instance) {
                vk::VertexInputRate::INSTANCE
            } else {
                vk::VertexInputRate::VERTEX
            },
        })
        .collect();
    let vi = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&attributes)
        .vertex_binding_descriptions(&bindings);
    if desc.primitive.strip_index_format.is_some() {
        return Err(Error::Unsupported("Vulkan primitive restart"));
    }
    let topology = match desc.primitive.topology {
        api::PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        api::PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
        _ => return Err(Error::Unsupported("Vulkan primitive topology")),
    };
    if !matches!(desc.primitive.polygon_mode, api::PolygonMode::Fill)
        || desc.primitive.conservative
        || desc.primitive.unclipped_depth
    {
        return Err(Error::Unsupported("Vulkan raster primitive state"));
    }
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let cull = match desc.primitive.cull_mode {
        api::CullMode::None => vk::CullModeFlags::NONE,
        api::CullMode::Front => vk::CullModeFlags::FRONT,
        api::CullMode::Back => vk::CullModeFlags::BACK,
    };
    let front = match desc.primitive.front_face {
        // Both RHI and Vulkan define front facing after viewport conversion,
        // in framebuffer coordinates. The negative-height viewport realizes
        // RHI's Y-up NDC contract, so no second winding compensation applies.
        api::FrontFace::Ccw => vk::FrontFace::COUNTER_CLOCKWISE,
        api::FrontFace::Cw => vk::FrontFace::CLOCKWISE,
    };
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(cull)
        .front_face(front)
        .line_width(1.0);
    let samples = match desc.multisample.count {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        _ => return Err(Error::Unsupported("Vulkan multisample count")),
    };
    if desc.multisample.alpha_to_coverage_enabled {
        return Err(Error::Unsupported("Vulkan alpha-to-coverage"));
    }
    if desc.multisample.mask != u64::from(u32::MAX) {
        return Err(Error::Unsupported("Vulkan multisample mask"));
    }
    let sample_mask = u32::MAX;
    let ms = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(samples)
        .sample_mask(std::slice::from_ref(&sample_mask));
    let blends: Result<Vec<_>> = desc
        .color_targets
        .iter()
        .map(|target| match target {
            Some(target) if target.write_mask == 0xf => {
                let mut native = vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA);
                if let Some(blend) = target.blend {
                    native = native
                        .blend_enable(true)
                        .src_color_blend_factor(blend_factor(blend.color.src_factor)?)
                        .dst_color_blend_factor(blend_factor(blend.color.dst_factor)?)
                        .color_blend_op(blend_op(blend.color.operation)?)
                        .src_alpha_blend_factor(blend_factor(blend.alpha.src_factor)?)
                        .dst_alpha_blend_factor(blend_factor(blend.alpha.dst_factor)?)
                        .alpha_blend_op(blend_op(blend.alpha.operation)?);
                }
                Ok(native)
            }
            Some(_) => Err(Error::Unsupported("Vulkan color write mask")),
            None => Ok(vk::PipelineColorBlendAttachmentState::default()),
        })
        .collect();
    let blends = blends?;
    let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
    let depth = desc
        .depth_stencil
        .as_ref()
        .map(|d| {
            if d.format != api::TextureFormat::Depth32Float {
                return Err(Error::Unsupported("Vulkan depth format"));
            }
            if d.depth_bias_constant != 0
                || d.depth_bias_slope_scale != 0.0
                || d.depth_bias_clamp != 0.0
                || !stencil_disabled(d)
            {
                return Err(Error::Unsupported("Vulkan depth bias or stencil state"));
            }
            Ok(vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(d.depth_write_enabled)
                .depth_compare_op(compare(d.depth_compare)))
        })
        .transpose()?;
    let formats: Result<Vec<_>> = desc
        .color_targets
        .iter()
        .map(|target| match target {
            Some(t) if t.write_mask == 0xf => super::super::resource::format(t.format)
                .ok_or(Error::Unsupported("Vulkan color format")),
            Some(_) => Err(Error::Unsupported("Vulkan color blend/write mask")),
            None => Ok(vk::Format::UNDEFINED),
        })
        .collect();
    let formats = formats?;
    let multiview_mask = desc
        .multiview
        .map(|views| (1_u32.checked_shl(views).unwrap_or(0)).wrapping_sub(1));
    if desc.multiview.is_some() && multiview_mask == Some(0) {
        return Err(Error::InvalidRange);
    }
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&formats)
        .view_mask(multiview_mask.unwrap_or(0));
    if depth.is_some() {
        rendering = rendering.depth_attachment_format(vk::Format::D32_SFLOAT);
    }
    // SAFETY: Core validated this layout's DeviceId; keeping a clone below
    // retains all dependent descriptor layouts for the native pipeline lifetime.
    let layout = unsafe { desc.layout.native_ref() }.clone();
    let mut info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vi)
        .input_assembly_state(&ia)
        .viewport_state(&viewport)
        .dynamic_state(&dynamic)
        .rasterization_state(&raster)
        .multisample_state(&ms)
        .color_blend_state(&cb)
        .layout(layout.native())
        .push_next(&mut rendering);
    if let Some(depth) = depth.as_ref() {
        info = info.depth_stencil_state(depth);
    }
    // SAFETY: every create-info slice and pNext structure above remains alive
    // for the call, and all Vulkan handles belong to this live DeviceState.
    let pipeline = unsafe {
        device
            .0
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
    }
    .map_err(|(_, e)| vk_error(e))?[0];
    Ok(VulkanRasterPipeline(Arc::new(Raster {
        device: device.0.clone(),
        pipeline,
        layout,
        color_formats: Arc::from(
            desc.color_targets
                .iter()
                .map(|target| target.as_ref().map(|target| target.format))
                .collect::<Vec<_>>(),
        ),
        depth_format: desc.depth_stencil.as_ref().map(|depth| depth.format),
        samples: desc.multisample.count,
        multiview_mask,
    })))
}

fn blend_factor(value: api::BlendFactor) -> Result<vk::BlendFactor> {
    Ok(match value {
        api::BlendFactor::Zero => vk::BlendFactor::ZERO,
        api::BlendFactor::One => vk::BlendFactor::ONE,
        api::BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
        api::BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        api::BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        api::BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        api::BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
        api::BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        api::BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        api::BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        _ => return Err(Error::Unsupported("Vulkan blend factor")),
    })
}

fn blend_op(value: api::BlendOperation) -> Result<vk::BlendOp> {
    Ok(match value {
        api::BlendOperation::Add => vk::BlendOp::ADD,
        api::BlendOperation::Subtract => vk::BlendOp::SUBTRACT,
        api::BlendOperation::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        api::BlendOperation::Min => vk::BlendOp::MIN,
        api::BlendOperation::Max => vk::BlendOp::MAX,
    })
}
fn compare(v: api::CompareFunction) -> vk::CompareOp {
    match v {
        api::CompareFunction::Never => vk::CompareOp::NEVER,
        api::CompareFunction::Less => vk::CompareOp::LESS,
        api::CompareFunction::Equal => vk::CompareOp::EQUAL,
        api::CompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        api::CompareFunction::Greater => vk::CompareOp::GREATER,
        api::CompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
        api::CompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        api::CompareFunction::Always => vk::CompareOp::ALWAYS,
    }
}
fn stencil_disabled(state: &api::DepthStencilState) -> bool {
    state.stencil_read_mask == 0
        && state.stencil_write_mask == 0
        && matches!(state.stencil_front.compare, api::CompareFunction::Always)
        && matches!(state.stencil_back.compare, api::CompareFunction::Always)
        && [state.stencil_front, state.stencil_back]
            .iter()
            .all(|face| {
                matches!(face.fail, api::StencilOperation::Keep)
                    && matches!(face.depth_fail, api::StencilOperation::Keep)
                    && matches!(face.pass, api::StencilOperation::Keep)
            })
}
fn stage<'a>(
    shader: &VulkanShader,
    entry: &'a std::ffi::CStr,
    flags: vk::ShaderStageFlags,
) -> Result<vk::PipelineShaderStageCreateInfo<'a>> {
    Ok(vk::PipelineShaderStageCreateInfo::default()
        .stage(flags)
        .module(shader.module())
        .name(entry))
}

fn vformat(v: api::VertexFormat) -> Option<vk::Format> {
    match v {
        api::VertexFormat::Float32(1) => Some(vk::Format::R32_SFLOAT),
        api::VertexFormat::Float32(2) => Some(vk::Format::R32G32_SFLOAT),
        api::VertexFormat::Float32(3) => Some(vk::Format::R32G32B32_SFLOAT),
        api::VertexFormat::Float32(4) => Some(vk::Format::R32G32B32A32_SFLOAT),
        api::VertexFormat::Sint32(1) => Some(vk::Format::R32_SINT),
        _ => None,
    }
}
