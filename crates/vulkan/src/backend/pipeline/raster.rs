//! Vulkan dynamic-rendering raster-pipeline creation.
//!
//! This module builds one dynamic-rendering pipeline from a Core descriptor.
//! The pipeline retains its layout and stores the render-target signature used
//! by secondary-command validation; encoders keep the pipeline alive until the
//! submission fence proves native command execution complete.

use super::super::{DeviceState, VulkanDevice, VulkanShader, vk_error};
use super::fixed::{
    blend_factor, blend_op, compare, polygon_mode, stencil_disabled, stencil_face,
    validate_attachment_format, validate_blends, validate_primitive,
};
use super::layout::{PushConstants, VulkanPipelineLayout};
use super::vertex_format::vformat;
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
    pub(crate) fn matches_rendering(
        &self,
        signature: &(Vec<vk::Format>, Option<vk::Format>, u32),
    ) -> bool {
        (signature.2 == 0 || self.0.samples == signature.2)
            && self
                .0
                .depth_format
                .and_then(|f| super::super::resource::native_format(&self.0.device, f))
                == signature.1
            && self
                .0
                .color_formats
                .iter()
                .map(|f| {
                    f.and_then(|f| super::super::resource::native_format(&self.0.device, f))
                        .unwrap_or(vk::Format::UNDEFINED)
                })
                .eq(signature.0.iter().copied())
    }
    pub(crate) fn compatible_read_only(&self, depth: bool, stencil: bool) -> bool {
        !(depth && self.0.depth_writes || stencil && self.0.stencil_writes)
    }
    pub(crate) fn has_dynamic_depth_bias(&self) -> bool {
        self.0.dynamic_depth_bias
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
    depth_writes: bool,
    stencil_writes: bool,
    dynamic_depth_bias: bool,
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
    dynamic_depth_bias: Option<api::DynamicDepthBias>,
) -> Result<VulkanRasterPipeline> {
    if dynamic_depth_bias.is_some() && desc.depth_stencil.is_none() {
        return Err(Error::InvalidState);
    }
    if dynamic_depth_bias.is_some_and(|bias| bias.clamp != 0.0)
        && device.0.features.depth_bias_clamp != vk::TRUE
    {
        return Err(Error::Unsupported("Vulkan depth bias clamp"));
    }
    let vertex_name = CString::new(desc.vertex.entry.as_str())
        .map_err(|_| Error::Shader("shader entry contains NUL".into()))?;
    let fragment_name = desc
        .fragment
        .as_ref()
        .map(|s| CString::new(s.entry.as_str()))
        .transpose()
        .map_err(|_| Error::Shader("shader entry contains NUL".into()))?;
    // SAFETY: Core has validated shader identity against this device.
    let (vertex_shader, _vertex_consumed) = super::super::shader::compile_variant(
        device,
        unsafe { desc.vertex.module.native_ref() },
        unsafe { desc.layout.native_ref() },
        &desc.vertex.entry,
        api::ShaderStage::Vertex,
        &desc.vertex.constants,
        unsafe { desc.layout.native_ref() }
            .push_constants()
            .base(api::ShaderStage::Vertex),
    )?;
    let fragment_shader = desc
        .fragment
        .as_ref()
        .map(|s| {
            // SAFETY: Core has validated shader identity against this device.
            super::super::shader::compile_variant(
                device,
                unsafe { s.module.native_ref() },
                unsafe { desc.layout.native_ref() },
                &s.entry,
                api::ShaderStage::Fragment,
                &s.constants,
                unsafe { desc.layout.native_ref() }
                    .push_constants()
                    .base(api::ShaderStage::Fragment),
            )
        })
        .transpose()?;
    let mut shader_refs = vec![(
        &vertex_shader,
        desc.vertex.entry.as_str(),
        api::ShaderStage::Vertex,
    )];
    if let Some((shader, _)) = &fragment_shader {
        shader_refs.push((
            shader,
            desc.fragment.as_ref().unwrap().entry.as_str(),
            api::ShaderStage::Fragment,
        ));
    }
    let layout = unsafe { desc.layout.native_ref() }.for_shaders(&shader_refs)?;
    let (vertex_shader, vertex_consumed) = super::super::shader::compile_variant(
        device,
        unsafe { desc.vertex.module.native_ref() },
        &layout,
        &desc.vertex.entry,
        api::ShaderStage::Vertex,
        &desc.vertex.constants,
        layout.push_constants().base(api::ShaderStage::Vertex),
    )?;
    let fragment_shader = desc
        .fragment
        .as_ref()
        .map(|s| {
            super::super::shader::compile_variant(
                device,
                unsafe { s.module.native_ref() },
                &layout,
                &s.entry,
                api::ShaderStage::Fragment,
                &s.constants,
                layout.push_constants().base(api::ShaderStage::Fragment),
            )
        })
        .transpose()?;
    let vertex_specialization = super::specialization::Specialization::new(
        &vertex_shader,
        if vertex_consumed {
            &[]
        } else {
            &desc.vertex.constants
        },
    )?;
    let vertex_specialization_info = vertex_specialization.info();
    let vs = stage(&vertex_shader, &vertex_name, vk::ShaderStageFlags::VERTEX)?
        .specialization_info(&vertex_specialization_info);
    let fragment_specialization = desc
        .fragment
        .as_ref()
        .zip(fragment_shader.as_ref())
        .map(|(s, (shader, consumed))| {
            super::specialization::Specialization::new(
                shader,
                if *consumed { &[] } else { &s.constants },
            )
        })
        .transpose()?;
    let fragment_specialization_info = fragment_specialization.as_ref().map(|s| s.info());
    let fs = fragment_shader
        .as_ref()
        .map(|(shader, _)| {
            Ok(stage(
                shader,
                fragment_name.as_ref().expect("fragment name exists"),
                vk::ShaderStageFlags::FRAGMENT,
            )?
            .specialization_info(
                fragment_specialization_info
                    .as_ref()
                    .expect("fragment specialization"),
            ))
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
    for attribute in &attributes {
        if attribute.format == vk::Format::UNDEFINED {
            return Err(Error::InvalidRange);
        }
        // SAFETY: the physical device and instance remain live through pipeline creation.
        let properties = unsafe {
            device
                .0
                .instance
                .instance
                .get_physical_device_format_properties(device.0.physical, attribute.format)
        };
        if !properties
            .buffer_features
            .contains(vk::FormatFeatureFlags::VERTEX_BUFFER)
        {
            return Err(Error::Unsupported("Vulkan device vertex format"));
        }
    }
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
    let topology = match desc.primitive.topology {
        api::PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
        api::PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
        api::PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        api::PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        api::PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
    };
    validate_primitive(device, &desc.primitive, desc.depth_stencil.as_ref())?;
    if desc.primitive.strip_index_format.is_some()
        && !matches!(
            desc.primitive.topology,
            api::PrimitiveTopology::LineStrip | api::PrimitiveTopology::TriangleStrip
        )
    {
        return Err(Error::InvalidRange);
    }
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(topology)
        .primitive_restart_enable(desc.primitive.strip_index_format.is_some());
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let mut dynamic_states = vec![
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::BLEND_CONSTANTS,
        vk::DynamicState::STENCIL_REFERENCE,
    ];
    if dynamic_depth_bias.is_some() {
        dynamic_states.push(vk::DynamicState::DEPTH_BIAS);
    }
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
    validate_blends(device, &desc.color_targets)?;
    let mut raster = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(desc.primitive.unclipped_depth)
        .polygon_mode(polygon_mode(desc.primitive.polygon_mode))
        .cull_mode(cull)
        .front_face(front)
        .line_width(1.0)
        .depth_bias_enable(
            dynamic_depth_bias.is_some()
                || desc.depth_stencil.as_ref().is_some_and(|d| {
                    d.depth_bias_constant != 0
                        || d.depth_bias_slope_scale != 0.0
                        || d.depth_bias_clamp != 0.0
                }),
        )
        .depth_bias_constant_factor(
            desc.depth_stencil
                .as_ref()
                .map_or(0.0, |d| d.depth_bias_constant as f32),
        )
        .depth_bias_slope_factor(
            desc.depth_stencil
                .as_ref()
                .map_or(0.0, |d| d.depth_bias_slope_scale),
        )
        .depth_bias_clamp(
            desc.depth_stencil
                .as_ref()
                .map_or(0.0, |d| d.depth_bias_clamp),
        );
    let mut conservative = vk::PipelineRasterizationConservativeStateCreateInfoEXT::default()
        .conservative_rasterization_mode(vk::ConservativeRasterizationModeEXT::OVERESTIMATE);
    if desc.primitive.conservative {
        raster = raster.push_next(&mut conservative);
    }
    let samples = match desc.multisample.count {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        32 => vk::SampleCountFlags::TYPE_32,
        64 => vk::SampleCountFlags::TYPE_64,
        _ => return Err(Error::Unsupported("Vulkan multisample count")),
    };
    let sample_mask = [
        desc.multisample.mask as u32,
        (desc.multisample.mask >> 32) as u32,
    ];
    let ms = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(samples)
        .alpha_to_coverage_enable(desc.multisample.alpha_to_coverage_enabled)
        .sample_mask(&sample_mask[..if desc.multisample.count > 32 { 2 } else { 1 }]);
    let blends: Result<Vec<_>> = desc
        .color_targets
        .iter()
        .map(|target| match target {
            Some(target) if target.write_mask & !0xf == 0 => {
                let mut native = vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::from_raw(target.write_mask as u32));
                if let Some(blend) = target.blend {
                    native = native
                        .blend_enable(true)
                        .src_color_blend_factor(blend_factor(blend.color.src_factor, false)?)
                        .dst_color_blend_factor(blend_factor(blend.color.dst_factor, false)?)
                        .color_blend_op(blend_op(blend.color.operation)?)
                        .src_alpha_blend_factor(blend_factor(blend.alpha.src_factor, true)?)
                        .dst_alpha_blend_factor(blend_factor(blend.alpha.dst_factor, true)?)
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
    let depth = desc.depth_stencil.as_ref().map(|d| {
        vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(d.depth_write_enabled)
            .depth_compare_op(compare(d.depth_compare))
            .stencil_test_enable(!stencil_disabled(d))
            .front(stencil_face(
                d.stencil_front,
                d.stencil_read_mask,
                d.stencil_write_mask,
            ))
            .back(stencil_face(
                d.stencil_back,
                d.stencil_read_mask,
                d.stencil_write_mask,
            ))
    });
    let formats: Result<Vec<_>> = desc
        .color_targets
        .iter()
        .map(|target| match target {
            Some(t) => super::super::resource::native_format(&device.0, t.format)
                .ok_or(Error::Unsupported("Vulkan color format")),
            None => Ok(vk::Format::UNDEFINED),
        })
        .collect();
    let formats = formats?;
    for (format, target) in formats.iter().zip(&desc.color_targets) {
        if let Some(target) = target {
            validate_attachment_format(device, *format, false, target.blend.is_some())?;
        }
    }
    if let Some(depth) = &desc.depth_stencil {
        validate_attachment_format(
            device,
            super::super::resource::native_format(&device.0, depth.format)
                .ok_or(Error::Unsupported("Vulkan depth format"))?,
            true,
            false,
        )?;
    }
    let multiview_mask = desc
        .multiview
        .map(|views| (1_u32.checked_shl(views).unwrap_or(0)).wrapping_sub(1));
    if let Some(views) = desc.multiview {
        if views == 0 || views > device.0.max_multiview_views || multiview_mask == Some(0) {
            return Err(Error::InvalidRange);
        }
        if !device.0.multiview {
            return Err(Error::Unsupported("Vulkan multiview"));
        }
    }
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&formats)
        .view_mask(multiview_mask.unwrap_or(0));
    if let Some(depth) = &desc.depth_stencil {
        let format = super::super::resource::native_format(&device.0, depth.format)
            .ok_or(Error::Unsupported("Vulkan depth format"))?;
        if depth.format != api::TextureFormat::Stencil8 {
            rendering = rendering.depth_attachment_format(format);
        }
        if matches!(
            depth.format,
            api::TextureFormat::Depth24PlusStencil8
                | api::TextureFormat::Depth32FloatStencil8
                | api::TextureFormat::Stencil8
        ) {
            rendering = rendering.stencil_attachment_format(format);
        }
    }
    // SAFETY: Core validated this layout's DeviceId; keeping a clone below
    // retains all dependent descriptor layouts for the native pipeline lifetime.

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
    let pipeline = super::cache::with_cache(desc.cache.as_ref(), |cache| {
        // SAFETY: the create-info data lives for the native call.
        unsafe {
            device
                .0
                .device
                .create_graphics_pipelines(cache, &[info], None)
        }
        .map_err(|(partial, error)| {
            for pipeline in partial {
                if pipeline != vk::Pipeline::null() {
                    // SAFETY: a failed native creation has not published these handles.
                    unsafe { device.0.device.destroy_pipeline(pipeline, None) };
                }
            }
            vk_error(error)
        })
        .map(|v| v[0])
    })?;
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
        depth_writes: desc
            .depth_stencil
            .as_ref()
            .is_some_and(|d| d.depth_write_enabled),
        stencil_writes: desc
            .depth_stencil
            .as_ref()
            .is_some_and(super::fixed::stencil_writes),
        dynamic_depth_bias: dynamic_depth_bias.is_some(),
    })))
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
