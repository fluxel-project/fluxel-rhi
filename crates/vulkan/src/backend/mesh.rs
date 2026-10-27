//! Vulkan EXT mesh pipelines and dispatch, retained by the recording encoder.
use super::pipeline::VulkanPipelineLayout;
use super::{
    DeviceState, Vulkan, VulkanBuffer, VulkanDevice, VulkanEncoder, VulkanShader, vk_error,
};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::{
    collections::HashMap,
    ffi::CString,
    sync::{Arc, Mutex},
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
    desc: api::MeshPipelineDesc<Vulkan>,
    variants: Mutex<HashMap<u32, VulkanMeshPipeline>>,
    layout: VulkanPipelineLayout,
    task: bool,
    depth_writes: bool,
    stencil_writes: bool,
    signature: (Vec<vk::Format>, Option<vk::Format>, u32),
}
impl Drop for MeshPipeline {
    fn drop(&mut self) {
        // SAFETY: encoders retain this owner until completion.
        unsafe { self.device.device.destroy_pipeline(self.pipeline, None) }
    }
}
pub struct Mesh;
impl fluxel_rhi_core::backend::MeshOps<Vulkan> for Mesh {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = super::cold_worker::ColdFuture<VulkanMeshPipeline>;
    fn create_pipeline<'a>(
        d: &'a VulkanDevice,
        desc: &'a api::MeshPipelineDesc<Vulkan>,
    ) -> Self::PipelineFuture<'a> {
        let device = d.clone();
        let desc = copy_desc(desc);
        super::cold_worker::submit(move || create(&device, &desc, 0))
    }
    fn bind_pipeline(e: &mut VulkanEncoder, p: &VulkanMeshPipeline) -> Result<()> {
        if !e.raster
            || e.secondary_signature.is_some()
            || (e.raster_depth_read_only && p.0.depth_writes)
            || (e.raster_stencil_read_only && p.0.stencil_writes)
            || !e.raster_signature.as_ref().is_some_and(|s| {
                s.0 == p.0.signature.0
                    && s.1 == p.0.signature.1
                    && (s.2 == 0 || s.2 == p.0.signature.2)
            })
        {
            return Err(Error::InvalidState);
        }
        let mask = e.raster_multiview_mask.unwrap_or(0);
        let selected = if mask == 0 {
            p.clone()
        } else {
            if !e.device.multiview_mesh_shader {
                return Err(Error::Unsupported("Vulkan multiview mesh shader"));
            }
            let mut variants = p.0.variants.lock().map_err(|_| Error::InvalidState)?;
            if let Some(variant) = variants.get(&mask) {
                variant.clone()
            } else {
                let variant = create(&VulkanDevice(e.device.clone()), &p.0.desc, mask)?;
                variants.insert(mask, variant.clone());
                variant
            }
        };
        let layout_state = selected.0.layout.clone();
        let bindings = super::raster::prepare_groups(&layout_state, &e.raster_bound_groups)?;
        // SAFETY: the compatible pipeline is retained by the encoder until completion.
        unsafe {
            e.device.device.cmd_bind_pipeline(
                e.command,
                vk::PipelineBindPoint::GRAPHICS,
                selected.0.pipeline,
            );
        }
        super::raster::bind_prepared(
            &e.device.device,
            e.command,
            layout_state.native(),
            &bindings,
        );
        e.raster_layout = Some(layout_state.native());
        e.raster_layout_state = Some(layout_state);
        e.raster_push_constants = Some(p.0.layout.push_constants());
        e.mesh_active = Some(p.0.task);
        e.mesh_pipelines.push(selected);
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
    fn dispatch_indirect_multi(
        e: &mut VulkanEncoder,
        b: &VulkanBuffer,
        offset: u64,
        count: u32,
    ) -> Result<()> {
        if e.mesh_active.is_none() || !e.device.multi_draw_indirect {
            return Err(Error::Unsupported("Vulkan mesh multi-indirect"));
        }
        validate_indirect_records(b, offset, count)?;
        let m = e
            .device
            .mesh
            .as_ref()
            .ok_or(Error::Unsupported("Vulkan mesh shader"))?;
        // SAFETY: mesh and multi-draw-indirect features are enabled; the
        // checked records use the fixed 12-byte VkDrawMeshTasksIndirectCommandEXT ABI.
        unsafe {
            m.loader
                .cmd_draw_mesh_tasks_indirect(e.command, b.0.buffer, offset, count, 12);
        }
        e.buffers.push(b.clone());
        Ok(())
    }
    fn dispatch_indirect_count(
        e: &mut VulkanEncoder,
        b: &VulkanBuffer,
        offset: u64,
        count_buffer: &VulkanBuffer,
        count_offset: u64,
        max_count: u32,
    ) -> Result<()> {
        if e.mesh_active.is_none() || !e.device.draw_indirect_count {
            return Err(Error::Unsupported("Vulkan mesh indirect count"));
        }
        let m = e
            .device
            .mesh
            .as_ref()
            .ok_or(Error::Unsupported("Vulkan mesh shader"))?;
        if max_count > e.device.max_draw_indirect_count
            || !count_buffer.0.usage.contains(api::BufferUsage::INDIRECT)
            || count_offset % 4 != 0
            || count_offset
                .checked_add(4)
                .is_none_or(|end| end > count_buffer.0.size)
        {
            return Err(Error::InvalidRange);
        }
        validate_indirect_records(b, offset, max_count)?;
        // SAFETY: mesh/draw-indirect-count features are enabled, all source
        // ranges are checked, and Vulkan clamps GPU count to `max_count`.
        unsafe {
            m.loader.cmd_draw_mesh_tasks_indirect_count(
                e.command,
                b.0.buffer,
                offset,
                count_buffer.0.buffer,
                count_offset,
                max_count,
                12,
            );
        }
        e.buffers.push(b.clone());
        e.buffers.push(count_buffer.clone());
        Ok(())
    }
}

fn validate_indirect_records(b: &VulkanBuffer, offset: u64, count: u32) -> Result<()> {
    if !b.0.usage.contains(api::BufferUsage::INDIRECT)
        || offset % 4 != 0
        || offset
            .checked_add(
                u64::from(count)
                    .checked_mul(12)
                    .ok_or(Error::InvalidRange)?,
            )
            .is_none_or(|end| end > b.0.size)
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
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
#[path = "../tests/backend/mesh/topology_tests.rs"]
mod topology_tests;

fn create(
    device: &VulkanDevice,
    desc: &api::MeshPipelineDesc<Vulkan>,
    multiview_mask: u32,
) -> Result<VulkanMeshPipeline> {
    let mesh = device
        .0
        .mesh
        .as_ref()
        .ok_or(Error::Unsupported("Vulkan mesh shader"))?;
    if desc.task.is_some() && !mesh.task {
        return Err(Error::Unsupported("Vulkan task shader"));
    }
    if desc.primitive.strip_index_format.is_some() {
        return Err(Error::InvalidRange);
    }
    super::pipeline::fixed::validate_primitive(
        device,
        &desc.primitive,
        desc.depth_stencil.as_ref(),
    )?;
    super::pipeline::fixed::validate_blends(device, &desc.targets)?;
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
    let stage_descs: Vec<_> = desc
        .task
        .iter()
        .map(|s| (s, api::ShaderStage::Task, vk::ShaderStageFlags::TASK_EXT))
        .chain(std::iter::once((
            &desc.mesh,
            api::ShaderStage::Mesh,
            vk::ShaderStageFlags::MESH_EXT,
        )))
        .chain(desc.fragment.iter().map(|s| {
            (
                s,
                api::ShaderStage::Fragment,
                vk::ShaderStageFlags::FRAGMENT,
            )
        }))
        .collect();
    // SAFETY: Core validated the layout and shader identities for this device.
    let base_layout = unsafe { desc.layout.native_ref() }.clone();
    let compile = |layout: &VulkanPipelineLayout| -> Result<Vec<_>> {
        stage_descs
            .iter()
            .map(|(s, kind, _)| {
                // SAFETY: Core validated each shader's device identity.
                super::shader::compile_variant(
                    device,
                    unsafe { s.module.native_ref() },
                    layout,
                    &s.entry,
                    *kind,
                    &s.constants,
                    layout.push_constants().base(*kind),
                )
            })
            .collect()
    };
    let analysis_shaders = compile(&base_layout)?;
    let shader_refs: Vec<_> = analysis_shaders
        .iter()
        .zip(&stage_descs)
        .map(|((shader, _), (stage, kind, _))| (shader, stage.entry.as_str(), *kind))
        .collect();
    let layout = base_layout.for_shaders(&shader_refs)?;
    let shaders = compile(&layout)?;
    let specs: Vec<_> = stage_descs
        .iter()
        .zip(&shaders)
        .map(|((s, _, _), (shader, consumed))| {
            super::pipeline::specialization::Specialization::new(
                shader,
                if *consumed { &[] } else { &s.constants },
            )
        })
        .collect::<Result<_>>()?;
    let spec_infos: Vec<_> = specs.iter().map(|s| s.info()).collect();
    let stages: Vec<_> = stage_descs
        .iter()
        .zip(&shaders)
        .zip(names.iter().zip(&spec_infos))
        .map(|(((_, _, flags), (shader, _)), (name, spec))| {
            stage(shader, name, *flags).map(|s| s.specialization_info(spec))
        })
        .collect::<Result<_>>()?;
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let dynamic_states = [
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::BLEND_CONSTANTS,
        vk::DynamicState::STENCIL_REFERENCE,
    ];
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
    let mut raster = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(desc.primitive.unclipped_depth)
        .polygon_mode(super::pipeline::fixed::polygon_mode(
            desc.primitive.polygon_mode,
        ))
        .cull_mode(cull)
        .front_face(front)
        .line_width(1.0)
        .depth_bias_enable(desc.depth_stencil.as_ref().is_some_and(|d| {
            d.depth_bias_constant != 0
                || d.depth_bias_slope_scale != 0.0
                || d.depth_bias_clamp != 0.0
        }))
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
        .targets
        .iter()
        .map(|target| match target {
            Some(target) if target.write_mask & !0xf == 0 => {
                let mut native = vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::from_raw(target.write_mask as u32));
                if let Some(blend) = target.blend {
                    native = native
                        .blend_enable(true)
                        .src_color_blend_factor(super::pipeline::fixed::blend_factor(
                            blend.color.src_factor,
                            false,
                        )?)
                        .dst_color_blend_factor(super::pipeline::fixed::blend_factor(
                            blend.color.dst_factor,
                            false,
                        )?)
                        .color_blend_op(super::pipeline::fixed::blend_op(blend.color.operation)?)
                        .src_alpha_blend_factor(super::pipeline::fixed::blend_factor(
                            blend.alpha.src_factor,
                            true,
                        )?)
                        .dst_alpha_blend_factor(super::pipeline::fixed::blend_factor(
                            blend.alpha.dst_factor,
                            true,
                        )?)
                        .alpha_blend_op(super::pipeline::fixed::blend_op(blend.alpha.operation)?);
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
            .depth_compare_op(super::pipeline::fixed::compare(d.depth_compare))
            .stencil_test_enable(!super::pipeline::fixed::stencil_disabled(d))
            .front(super::pipeline::fixed::stencil_face(
                d.stencil_front,
                d.stencil_read_mask,
                d.stencil_write_mask,
            ))
            .back(super::pipeline::fixed::stencil_face(
                d.stencil_back,
                d.stencil_read_mask,
                d.stencil_write_mask,
            ))
    });
    let formats: Result<Vec<_>> = desc
        .targets
        .iter()
        .map(|target| match target {
            Some(t) => super::resource::native_format(&device.0, t.format)
                .ok_or(Error::Unsupported("Vulkan color format")),
            None => Ok(vk::Format::UNDEFINED),
        })
        .collect();
    let formats = formats?;
    for (format, target) in formats.iter().zip(&desc.targets) {
        if let Some(target) = target {
            super::pipeline::fixed::validate_attachment_format(
                device,
                *format,
                false,
                target.blend.is_some(),
            )?;
        }
    }
    if let Some(depth) = &desc.depth_stencil {
        super::pipeline::fixed::validate_attachment_format(
            device,
            super::resource::native_format(&device.0, depth.format)
                .ok_or(Error::Unsupported("Vulkan depth format"))?,
            true,
            false,
        )?;
    }
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&formats)
        .view_mask(multiview_mask);
    if let Some(depth) = &desc.depth_stencil {
        let format = super::resource::native_format(&device.0, depth.format)
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
    let pipeline = super::pipeline::cache::with_cache(desc.cache.as_ref(), |cache| {
        unsafe {
            device
                .0
                .device
                .create_graphics_pipelines(cache, &[info], None)
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
        })
        .map(|pipelines| pipelines[0])
    })?;
    Ok(VulkanMeshPipeline(Arc::new(MeshPipeline {
        device: device.0.clone(),
        pipeline,
        desc: copy_desc(desc),
        variants: Mutex::new(HashMap::new()),
        layout,
        task: desc.task.is_some(),
        depth_writes: desc
            .depth_stencil
            .as_ref()
            .is_some_and(|d| d.depth_write_enabled),
        stencil_writes: desc
            .depth_stencil
            .as_ref()
            .is_some_and(super::pipeline::fixed::stencil_writes),
        signature: (
            formats,
            desc.depth_stencil
                .as_ref()
                .and_then(|d| super::resource::native_format(&device.0, d.format)),
            desc.multisample.count,
        ),
    })))
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

// Keep only immutable pipeline inputs so view-mask variants can be created on demand.
fn copy_desc(desc: &api::MeshPipelineDesc<Vulkan>) -> api::MeshPipelineDesc<Vulkan> {
    fn stage(s: &api::PipelineStage<Vulkan>) -> api::PipelineStage<Vulkan> {
        api::PipelineStage {
            module: s.module.clone(),
            entry: s.entry.clone(),
            constants: s.constants.clone(),
        }
    }
    api::MeshPipelineDesc {
        layout: desc.layout.clone(),
        task: desc.task.as_ref().map(stage),
        mesh: stage(&desc.mesh),
        fragment: desc.fragment.as_ref().map(stage),
        primitive: desc.primitive,
        targets: desc.targets.clone(),
        depth_stencil: desc.depth_stencil,
        multisample: desc.multisample,
        cache: desc.cache.clone(),
    }
}
