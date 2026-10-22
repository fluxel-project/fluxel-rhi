//! Vulkan EXT mesh pipelines and dispatch, retained by the recording encoder.
use super::pipeline::VulkanPipelineLayout;
use super::{
    DeviceState, Vulkan, VulkanBuffer, VulkanDevice, VulkanEncoder, VulkanShader, vk_error,
};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::{
    ffi::CString,
    future::{Ready, ready},
    sync::Arc,
};

pub(crate) struct MeshDevice {
    loader: ash::ext::mesh_shader::Device,
    pub(super) task: bool,
    mesh_count: [u32; 3],
    mesh_total: u32,
    task_count: [u32; 3],
    task_total: u32,
}
impl MeshDevice {
    pub(crate) fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        task: bool,
        p: &vk::PhysicalDeviceMeshShaderPropertiesEXT<'_>,
    ) -> Self {
        Self {
            loader: ash::ext::mesh_shader::Device::new(instance, device),
            task,
            mesh_count: p.max_mesh_work_group_count,
            mesh_total: p.max_mesh_work_group_total_count,
            task_count: p.max_task_work_group_count,
            task_total: p.max_task_work_group_total_count,
        }
    }
}
#[derive(Clone)]
pub struct VulkanMeshPipeline(Arc<MeshPipeline>);
struct MeshPipeline {
    device: Arc<DeviceState>,
    pipeline: vk::Pipeline,
    layout: VulkanPipelineLayout,
    task: bool,
    signature: (vk::Format, Option<vk::Format>, u32),
}
impl Drop for MeshPipeline {
    fn drop(&mut self) {
        // SAFETY: encoders retain this owner until completion.
        unsafe { self.device.device.destroy_pipeline(self.pipeline, None) }
    }
}
pub struct Mesh;
impl api::MeshOps<Vulkan> for Mesh {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = Ready<Result<VulkanMeshPipeline>>;
    fn create_pipeline<'a>(
        d: &'a VulkanDevice,
        desc: &'a api::MeshPipelineDesc<Vulkan>,
    ) -> Self::PipelineFuture<'a> {
        ready(create(d, desc))
    }
    fn bind_pipeline(e: &mut VulkanEncoder, p: &VulkanMeshPipeline) -> Result<()> {
        if !e.raster
            || e.secondary_signature.is_some()
            || e.raster_multiview_mask.is_some()
            || e.raster_signature != Some(p.0.signature)
        {
            return Err(Error::InvalidState);
        }
        // SAFETY: the compatible pipeline is retained by the encoder until completion.
        unsafe {
            e.device.device.cmd_bind_pipeline(
                e.command,
                vk::PipelineBindPoint::GRAPHICS,
                p.0.pipeline,
            );
        }
        e.raster_layout = Some(p.0.layout.native());
        e.raster_layout_state = Some(p.0.layout.clone());
        e.raster_push_constants = Some(p.0.layout.push_constants());
        e.mesh_active = Some(p.0.task);
        e.mesh_pipelines.push(p.clone());
        Ok(())
    }
    fn dispatch(e: &mut VulkanEncoder, groups: [u32; 3]) -> Result<()> {
        let task = e.mesh_active.ok_or(Error::InvalidState)?;
        let m = e
            .device
            .mesh
            .as_ref()
            .ok_or(Error::Unsupported("Vulkan mesh shader"))?;
        let (limit, total) = if task {
            (m.task_count, m.task_total)
        } else {
            (m.mesh_count, m.mesh_total)
        };
        if groups.iter().zip(limit).any(|(v, l)| *v > l)
            || groups
                .iter()
                .try_fold(1u64, |n, v| n.checked_mul(u64::from(*v)))
                .is_none_or(|n| n > u64::from(total))
        {
            return Err(Error::InvalidRange);
        }
        // SAFETY: feature enabled, active mesh pipeline and validated group limits.
        unsafe {
            m.loader
                .cmd_draw_mesh_tasks(e.command, groups[0], groups[1], groups[2]);
        }
        Ok(())
    }
    fn dispatch_indirect(e: &mut VulkanEncoder, b: &VulkanBuffer, offset: u64) -> Result<()> {
        if e.mesh_active.is_none()
            || !b.0.usage.contains(api::BufferUsage::INDIRECT)
            || offset % 4 != 0
            || offset.checked_add(12).is_none_or(|x| x > b.0.size)
        {
            return Err(Error::InvalidRange);
        }
        let m = e
            .device
            .mesh
            .as_ref()
            .ok_or(Error::Unsupported("Vulkan mesh shader"))?;
        // SAFETY: indirect usage and range checked, buffer retained through submission; caller supplies valid GPU counts.
        unsafe {
            m.loader
                .cmd_draw_mesh_tasks_indirect(e.command, b.0.buffer, offset, 1, 12);
        }
        e.buffers.push(b.clone());
        Ok(())
    }
}
fn topology_matches(topology: api::PrimitiveTopology, output_mode: u32) -> bool {
    matches!(
        (topology, output_mode),
        (api::PrimitiveTopology::PointList, 27)
            | (api::PrimitiveTopology::LineList, 5269)
            | (api::PrimitiveTopology::TriangleList, 5298)
    )
}

#[cfg(test)]
mod topology_tests {
    use super::*;

    #[test]
    fn mesh_topology_rejects_descriptor_shader_mismatch_and_strips() {
        assert!(topology_matches(api::PrimitiveTopology::TriangleList, 5298));
        assert!(!topology_matches(api::PrimitiveTopology::PointList, 5298));
        assert!(!topology_matches(
            api::PrimitiveTopology::TriangleStrip,
            5298
        ));
        assert!(topology_matches(api::PrimitiveTopology::LineList, 5269));
    }
}

fn create(
    device: &VulkanDevice,
    desc: &api::MeshPipelineDesc<Vulkan>,
) -> Result<VulkanMeshPipeline> {
    if desc.targets.len() != 1 || desc.targets[0].is_none() {
        return Err(Error::Unsupported("Vulkan mesh color target"));
    }
    let mesh = device
        .0
        .mesh
        .as_ref()
        .ok_or(Error::Unsupported("Vulkan mesh shader"))?;
    if desc.task.is_some() && !mesh.task {
        return Err(Error::Unsupported("Vulkan task shader"));
    }
    if desc.cache.is_some()
        || !desc.mesh.constants.is_empty()
        || desc.task.as_ref().is_some_and(|s| !s.constants.is_empty())
        || desc
            .fragment
            .as_ref()
            .is_some_and(|s| !s.constants.is_empty())
    {
        return Err(Error::Unsupported("Vulkan mesh specialization or cache"));
    }
    if desc.primitive.strip_index_format.is_some()
        || !matches!(desc.primitive.polygon_mode, api::PolygonMode::Fill)
        || desc.primitive.conservative
        || desc.primitive.unclipped_depth
    {
        return Err(Error::Unsupported("Vulkan mesh primitive state"));
    }
    // A mesh shader supplies its own primitive topology through SPIR-V
    // OpExecutionMode. There is no Vulkan input-assembly state to override it.
    let output_mode = unsafe { desc.mesh.module.native_ref() }
        .mesh_output_mode(&desc.mesh.entry)
        .ok_or(Error::Unsupported("Vulkan mesh output topology"))?;
    if !topology_matches(desc.primitive.topology, output_mode) {
        return Err(Error::Unsupported("Vulkan mesh output topology mismatch"));
    }
    let mut names = Vec::new();
    for s in desc
        .task
        .iter()
        .chain(std::iter::once(&desc.mesh))
        .chain(desc.fragment.iter())
    {
        names.push(
            CString::new(s.entry.as_str())
                .map_err(|_| Error::Shader("shader entry contains NUL".into()))?,
        );
    }
    let mut stages = Vec::new();
    let mut i = 0;
    if let Some(s) = &desc.task {
        stages.push(stage(
            unsafe { s.module.native_ref() },
            &names[i],
            vk::ShaderStageFlags::TASK_EXT,
        )?);
        i += 1;
    }
    stages.push(stage(
        unsafe { desc.mesh.module.native_ref() },
        &names[i],
        vk::ShaderStageFlags::MESH_EXT,
    )?);
    i += 1;
    if let Some(s) = &desc.fragment {
        stages.push(stage(
            unsafe { s.module.native_ref() },
            &names[i],
            vk::ShaderStageFlags::FRAGMENT,
        )?);
    }
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
        .targets
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
    let formats: Result<Vec<_>> =
        desc.targets
            .iter()
            .map(|target| match target {
                Some(t) if t.write_mask == 0xf => super::resource::format(t.format)
                    .ok_or(Error::Unsupported("Vulkan color format")),
                Some(_) => Err(Error::Unsupported("Vulkan color blend/write mask")),
                None => Ok(vk::Format::UNDEFINED),
            })
            .collect();
    let formats = formats?;
    let multiview_mask = None::<u32>;
    if multiview_mask == Some(0) {
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
    .map_err(|(partial, e)| {
        for pipeline in partial {
            if pipeline != vk::Pipeline::null() {
                // SAFETY: these partially created pipelines have not been published.
                unsafe {
                    device.0.device.destroy_pipeline(pipeline, None);
                }
            }
        }
        vk_error(e)
    })?[0];
    Ok(VulkanMeshPipeline(Arc::new(MeshPipeline {
        device: device.0.clone(),
        pipeline,
        layout,
        task: desc.task.is_some(),
        signature: (
            formats[0],
            depth.as_ref().map(|_| vk::Format::D32_SFLOAT),
            desc.multisample.count,
        ),
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
    matches!(state.stencil_front.compare, api::CompareFunction::Always)
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
    let expected = if flags == vk::ShaderStageFlags::TASK_EXT {
        api::ShaderStage::Task
    } else if flags == vk::ShaderStageFlags::MESH_EXT {
        api::ShaderStage::Mesh
    } else {
        api::ShaderStage::Fragment
    };
    if !shader
        .entries()
        .iter()
        .any(|s| s.stage == expected && s.name.as_bytes() == entry.to_bytes())
    {
        return Err(Error::Shader(
            "Vulkan mesh shader entry or stage mismatch".into(),
        ));
    }
    Ok(vk::PipelineShaderStageCreateInfo::default()
        .stage(flags)
        .module(shader.module())
        .name(entry))
}
