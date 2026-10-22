//! Bind render resources and stage-local push and runtime-size data.

#[cfg(target_vendor = "apple")]
use crate::backend::bindings::BoundResource;
use crate::backend::{MetalBindGroup, MetalCommandEncoder};
use fluxel_rhi_core::api;

#[cfg(target_vendor = "apple")]
mod constants;
#[cfg(target_vendor = "apple")]
pub(super) use constants::bind_raster_push;
#[cfg(target_vendor = "apple")]
use constants::bind_raster_sizes;
#[cfg(target_vendor = "apple")]
pub(in crate::backend::encoder) use constants::write_push;

pub(super) fn set_bind_group(
    encoder: &mut MetalCommandEncoder,
    group_slot: u32,
    group: &MetalBindGroup,
    offsets: &[u32],
) -> api::Result<()> {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::MTLRenderCommandEncoder as _;
        let dynamic_count = group
            .resources
            .iter()
            .filter(|resource| {
                matches!(
                    resource,
                    BoundResource::Buffer {
                        dynamic_offset: true,
                        ..
                    }
                ) || matches!(
                    resource,
                    BoundResource::Array {
                        ty: api::BindingType::UniformBuffer {
                            dynamic_offset: true,
                            ..
                        } | api::BindingType::StorageBuffer {
                            dynamic_offset: true,
                            ..
                        },
                        ..
                    }
                )
            })
            .count();
        if offsets.len() != dynamic_count {
            return Err(api::Error::InvalidRange);
        }
        // Validate the complete offset slice before changing native encoder
        // state, so a late bad offset cannot leave a partially rebound group.
        let mut dynamic_index = 0;
        for resource in &group.resources {
            if let BoundResource::Buffer {
                offset,
                size,
                allocation_size,
                dynamic_offset: true,
                offset_alignment,
                ..
            } = resource
            {
                crate::backend::bindings::apply_dynamic_offset(
                    *offset,
                    *size,
                    *allocation_size,
                    offsets[dynamic_index],
                    *offset_alignment,
                )?;
                dynamic_index += 1;
            }
            if let BoundResource::Array { ty, elements, .. } = resource {
                if matches!(
                    ty,
                    api::BindingType::UniformBuffer {
                        dynamic_offset: true,
                        ..
                    } | api::BindingType::StorageBuffer {
                        dynamic_offset: true,
                        ..
                    }
                ) {
                    crate::backend::bindings::argument_elements(
                        ty,
                        elements,
                        elements.len() as u32,
                        Some(offsets[dynamic_index]),
                    )?;
                    dynamic_index += 1;
                }
            }
        }
        if encoder.render.is_none() {
            return Err(api::Error::InvalidState);
        }
        if encoder.raster_pipeline_bound && group_slot as usize >= encoder.raster_group_count {
            return Err(api::Error::InvalidRange);
        }
        if encoder.raster_pipeline_bound {
            let mut dynamic_index = 0;
            for resource in &group.resources {
                let (binding, dynamic, buffers): (u32, bool, Vec<&BoundResource>) = match resource {
                    BoundResource::Buffer {
                        binding,
                        dynamic_offset,
                        ..
                    } => (*binding, *dynamic_offset, vec![resource]),
                    BoundResource::Array {
                        binding,
                        ty,
                        elements,
                        ..
                    } => (
                        *binding,
                        matches!(
                            ty,
                            api::BindingType::UniformBuffer {
                                dynamic_offset: true,
                                ..
                            } | api::BindingType::StorageBuffer {
                                dynamic_offset: true,
                                ..
                            }
                        ),
                        elements.iter().flatten().collect(),
                    ),
                    _ => continue,
                };
                let dynamic_offset = if dynamic {
                    let value = offsets[dynamic_index] as u64;
                    dynamic_index += 1;
                    value
                } else {
                    0
                };
                for stage in &encoder.raster_bindings {
                    if let Some((_, _, alignment)) = stage
                        .buffer_alignments
                        .iter()
                        .find(|(group, entry, _)| *group == group_slot && *entry == binding)
                    {
                        for buffer in &buffers {
                            if let BoundResource::Buffer { offset, .. } = buffer {
                                if offset
                                    .checked_add(dynamic_offset)
                                    .is_none_or(|value| value % alignment != 0)
                                {
                                    return Err(api::Error::InvalidRange);
                                }
                            }
                        }
                    }
                }
            }
        }
        encoder
            .raster_groups
            .insert(group_slot, (group.clone(), offsets.to_vec()));
        if !encoder.raster_pipeline_bound {
            return Ok(());
        }
        let render = encoder.render.as_ref().ok_or(api::Error::InvalidState)?;
        let mut dynamic_index = 0;
        for resource in &group.resources {
            match resource {
                BoundResource::Array {
                    binding,
                    visibility,
                    ty,
                    elements,
                } => {
                    use objc2_metal::{MTLCommandEncoder as _, MTLRenderStages, MTLResourceUsage};
                    let dynamic = matches!(
                        ty,
                        api::BindingType::UniformBuffer {
                            dynamic_offset: true,
                            ..
                        } | api::BindingType::StorageBuffer {
                            dynamic_offset: true,
                            ..
                        }
                    );
                    let offset = if dynamic {
                        let value = offsets[dynamic_index];
                        dynamic_index += 1;
                        Some(value)
                    } else {
                        None
                    };
                    let usage = match ty {
                        api::BindingType::StorageBuffer { access, .. }
                        | api::BindingType::StorageTexture { access, .. } => match access {
                            api::StorageAccess::Read => MTLResourceUsage::Read,
                            api::StorageAccess::Write => MTLResourceUsage::Write,
                            api::StorageAccess::ReadWrite => {
                                MTLResourceUsage::Read | MTLResourceUsage::Write
                            }
                        },
                        _ => MTLResourceUsage::Read,
                    };
                    for stage in &encoder.raster_bindings {
                        if !visibility.contains(&stage.stage) {
                            continue;
                        }
                        let Some((_, _, capacity)) = stage
                            .arrays
                            .iter()
                            .find(|(g, b, _)| *g == group_slot && *b == *binding)
                        else {
                            continue;
                        };
                        let function = stage.function.as_ref().ok_or(api::Error::InvalidState)?;
                        let slot = stage
                            .map
                            .slot(group_slot, *binding)
                            .ok_or(api::Error::InvalidState)?
                            .index as usize;
                        let native_elements = crate::backend::bindings::argument_elements(
                            ty, elements, *capacity, offset,
                        )?;
                        if let Some((_, _, metadata_slot)) =
                            stage.map.ray_metadata.iter().find(|(group, entry, _)| {
                                *group == group_slot && *entry == *binding
                            })
                        {
                            let structures = elements
                                .iter()
                                .map(|element| match element {
                                    Some(BoundResource::Acceleration { value, .. }) => {
                                        Ok(Some(value.clone()))
                                    }
                                    None => Ok(None),
                                    _ => Err(api::Error::InvalidState),
                                })
                                .collect::<api::Result<Vec<_>>>()?;
                            let metadata =
                                crate::backend::ray::MetalAcceleration::query_contribution_array(
                                    &render.device(),
                                    &structures,
                                    *capacity as usize,
                                    encoder.ray_owner,
                                )?;
                            // SAFETY: the shader metadata assigns this slot to the immutable snapshot;
                            // the command buffer retains the buffer through GPU execution.
                            unsafe {
                                match stage.stage {
                                    api::ShaderStage::Vertex => render
                                        .setVertexBuffer_offset_atIndex(
                                            Some(&metadata),
                                            0,
                                            *metadata_slot as usize,
                                        ),
                                    api::ShaderStage::Fragment => render
                                        .setFragmentBuffer_offset_atIndex(
                                            Some(&metadata),
                                            0,
                                            *metadata_slot as usize,
                                        ),
                                    _ => {
                                        return Err(api::Error::Unsupported(
                                            "Metal mesh ray query",
                                        ));
                                    }
                                }
                            }
                        }
                        let render_stage = match stage.stage {
                            api::ShaderStage::Vertex => MTLRenderStages::Vertex,
                            api::ShaderStage::Fragment => MTLRenderStages::Fragment,
                            api::ShaderStage::Task => MTLRenderStages::Object,
                            api::ShaderStage::Mesh => MTLRenderStages::Mesh,
                            _ => return Err(api::Error::InvalidState),
                        };
                        for item in elements.iter().flatten() {
                            if let BoundResource::Acceleration { value, .. } = item {
                                value.validate_binding(encoder.ray_owner)?;
                                value.declare_render_dependencies(
                                    render,
                                    encoder.ray_owner,
                                    render_stage,
                                )?;
                                encoder.accelerations.push(value.clone());
                            }
                        }
                        // SAFETY: translated shader metadata specifies this exact array and buffer slot.
                        let array = unsafe {
                            crate::backend::argument::materialize(
                                &render.device(),
                                function,
                                slot,
                                native_elements,
                            )?
                        };
                        for element in &array.elements {
                            if let Some(resource) = element.resource() {
                                // SAFETY: resources remain retained until GPU completion.
                                render.useResource_usage_stages(resource, usage, render_stage);
                            }
                        }
                        // SAFETY: slot belongs to this shader stage and the array remains alive through completion.
                        unsafe {
                            match stage.stage {
                                api::ShaderStage::Vertex => render.setVertexBuffer_offset_atIndex(
                                    Some(&array.buffer),
                                    0,
                                    slot,
                                ),
                                api::ShaderStage::Fragment => render
                                    .setFragmentBuffer_offset_atIndex(Some(&array.buffer), 0, slot),
                                api::ShaderStage::Task => render.setObjectBuffer_offset_atIndex(
                                    Some(&array.buffer),
                                    0,
                                    slot,
                                ),
                                api::ShaderStage::Mesh => render.setMeshBuffer_offset_atIndex(
                                    Some(&array.buffer),
                                    0,
                                    slot,
                                ),
                                _ => unreachable!(),
                            }
                        };
                        encoder.argument_arrays.push(std::sync::Arc::new(array));
                    }
                }
                BoundResource::Acceleration {
                    binding,
                    visibility,
                    value,
                } => {
                    use objc2_foundation::NSObjectProtocol as _;
                    use objc2_metal::MTLRenderStages;
                    value.validate_binding(encoder.ray_owner)?;
                    if [api::ShaderStage::Task, api::ShaderStage::Mesh]
                        .iter()
                        .any(|stage| {
                            visibility.contains(stage)
                                && raster_slot(
                                    &encoder.raster_bindings,
                                    *stage,
                                    group_slot,
                                    *binding,
                                )
                                .is_some()
                        })
                    {
                        return Err(api::Error::Unsupported("Metal mesh acceleration binding"));
                    }
                    let mut stages = MTLRenderStages::empty();
                    if let Some(index) = raster_slot(
                        &encoder.raster_bindings,
                        api::ShaderStage::Vertex,
                        group_slot,
                        *binding,
                    )
                    .filter(|_| visibility.contains(&api::ShaderStage::Vertex))
                    {
                        if !render.respondsToSelector(
                            objc2::sel!(setVertexAccelerationStructure:atBufferIndex:),
                        ) {
                            return Err(api::Error::Unsupported("Metal vertex ray tracing"));
                        }
                        // SAFETY: the queried selector exists, the slot is allocated by this pipeline, and the TLAS is retained through completion.
                        unsafe {
                            render.setVertexAccelerationStructure_atBufferIndex(
                                Some(&value.raw),
                                index,
                            )
                        };
                        stages |= MTLRenderStages::Vertex;
                    }
                    if let Some(index) = raster_slot(
                        &encoder.raster_bindings,
                        api::ShaderStage::Fragment,
                        group_slot,
                        *binding,
                    )
                    .filter(|_| visibility.contains(&api::ShaderStage::Fragment))
                    {
                        if !render.respondsToSelector(
                            objc2::sel!(setFragmentAccelerationStructure:atBufferIndex:),
                        ) {
                            return Err(api::Error::Unsupported("Metal fragment ray tracing"));
                        }
                        // SAFETY: the queried selector exists, the slot is allocated by this pipeline, and the TLAS is retained through completion.
                        unsafe {
                            render.setFragmentAccelerationStructure_atBufferIndex(
                                Some(&value.raw),
                                index,
                            )
                        };
                        stages |= MTLRenderStages::Fragment;
                    }
                    for stage in &encoder.raster_bindings {
                        if !visibility.contains(&stage.stage) {
                            continue;
                        }
                        let Some((_, _, slot)) =
                            stage.map.ray_metadata.iter().find(|(group, entry, _)| {
                                *group == group_slot && *entry == *binding
                            })
                        else {
                            continue;
                        };
                        let buffer = value.query_contributions(encoder.ray_owner)?;
                        // SAFETY: the stage map reserves this slot; the command buffer retains the immutable snapshot.
                        unsafe {
                            match stage.stage {
                                api::ShaderStage::Vertex => render.setVertexBuffer_offset_atIndex(
                                    Some(&buffer),
                                    0,
                                    *slot as usize,
                                ),
                                api::ShaderStage::Fragment => render
                                    .setFragmentBuffer_offset_atIndex(
                                        Some(&buffer),
                                        0,
                                        *slot as usize,
                                    ),
                                _ => return Err(api::Error::InvalidState),
                            }
                        };
                    }
                    if !stages.is_empty() {
                        value.declare_render_dependencies(render, encoder.ray_owner, stages)?;
                        encoder.accelerations.push(value.clone());
                    }
                }
                BoundResource::Buffer {
                    binding,
                    visibility,
                    raw,
                    offset,
                    size,
                    allocation_size,
                    dynamic_offset,
                    offset_alignment,
                    ..
                } => {
                    let effective_offset = if *dynamic_offset {
                        let dynamic = offsets[dynamic_index];
                        dynamic_index += 1;
                        crate::backend::bindings::apply_dynamic_offset(
                            *offset,
                            *size,
                            *allocation_size,
                            dynamic,
                            *offset_alignment,
                        )?
                    } else {
                        *offset
                    };
                    if let Some(index) = raster_slot(
                        &encoder.raster_bindings,
                        api::ShaderStage::Vertex,
                        group_slot,
                        *binding,
                    )
                    .filter(|_| visibility.contains(&api::ShaderStage::Vertex))
                    {
                        // SAFETY: index is within the fixed 28-slot ABI;
                        // the effective aligned range was validated and the
                        // retained resource remains alive through completion.
                        unsafe {
                            render.setVertexBuffer_offset_atIndex(
                                Some(raw),
                                effective_offset as usize,
                                index as usize,
                            )
                        };
                    }
                    if let Some(index) = raster_slot(
                        &encoder.raster_bindings,
                        api::ShaderStage::Task,
                        group_slot,
                        *binding,
                    )
                    .filter(|_| visibility.contains(&api::ShaderStage::Task))
                    {
                        // SAFETY: slot and retained buffer range have been validated above.
                        unsafe {
                            render.setObjectBuffer_offset_atIndex(
                                Some(raw),
                                effective_offset as usize,
                                index as usize,
                            )
                        };
                    }
                    if let Some(index) = raster_slot(
                        &encoder.raster_bindings,
                        api::ShaderStage::Mesh,
                        group_slot,
                        *binding,
                    )
                    .filter(|_| visibility.contains(&api::ShaderStage::Mesh))
                    {
                        // SAFETY: slot and retained buffer range have been validated above.
                        unsafe {
                            render.setMeshBuffer_offset_atIndex(
                                Some(raw),
                                effective_offset as usize,
                                index as usize,
                            )
                        };
                    }
                    if let Some(index) = raster_slot(
                        &encoder.raster_bindings,
                        api::ShaderStage::Fragment,
                        group_slot,
                        *binding,
                    )
                    .filter(|_| visibility.contains(&api::ShaderStage::Fragment))
                    {
                        // SAFETY: index is within the fixed 28-slot ABI;
                        // the effective aligned range was validated and the
                        // retained resource remains alive through completion.
                        unsafe {
                            render.setFragmentBuffer_offset_atIndex(
                                Some(raw),
                                effective_offset as usize,
                                index as usize,
                            )
                        };
                    }
                }
                BoundResource::Texture {
                    binding,
                    visibility,
                    raw,
                } => {
                    if let Some(index) = raster_slot(
                        &encoder.raster_bindings,
                        api::ShaderStage::Vertex,
                        group_slot,
                        *binding,
                    )
                    .filter(|_| visibility.contains(&api::ShaderStage::Vertex))
                    {
                        // SAFETY: index is within the fixed 28-slot ABI and
                        // the texture is retained until command completion.
                        unsafe { render.setVertexTexture_atIndex(Some(raw), index) };
                    }
                    if let Some(index) = raster_slot(
                        &encoder.raster_bindings,
                        api::ShaderStage::Task,
                        group_slot,
                        *binding,
                    )
                    .filter(|_| visibility.contains(&api::ShaderStage::Task))
                    {
                        // SAFETY: the fixed slot is valid and the resource is retained through completion.
                        unsafe { render.setObjectTexture_atIndex(Some(raw), index) };
                    }
                    if let Some(index) = raster_slot(
                        &encoder.raster_bindings,
                        api::ShaderStage::Mesh,
                        group_slot,
                        *binding,
                    )
                    .filter(|_| visibility.contains(&api::ShaderStage::Mesh))
                    {
                        // SAFETY: the fixed slot is valid and the resource is retained through completion.
                        unsafe { render.setMeshTexture_atIndex(Some(raw), index) };
                    }
                    if let Some(index) = raster_slot(
                        &encoder.raster_bindings,
                        api::ShaderStage::Fragment,
                        group_slot,
                        *binding,
                    )
                    .filter(|_| visibility.contains(&api::ShaderStage::Fragment))
                    {
                        // SAFETY: index is within the fixed 28-slot ABI and
                        // the texture is retained until command completion.
                        unsafe { render.setFragmentTexture_atIndex(Some(raw), index) };
                    }
                }
                BoundResource::Sampler {
                    binding,
                    visibility,
                    raw,
                } => {
                    if let Some(index) = raster_slot(
                        &encoder.raster_bindings,
                        api::ShaderStage::Vertex,
                        group_slot,
                        *binding,
                    )
                    .filter(|_| visibility.contains(&api::ShaderStage::Vertex))
                    {
                        // SAFETY: index is within the fixed 28-slot ABI and
                        // the sampler is retained until command completion.
                        unsafe { render.setVertexSamplerState_atIndex(Some(raw), index) };
                    }
                    if let Some(index) = raster_slot(
                        &encoder.raster_bindings,
                        api::ShaderStage::Task,
                        group_slot,
                        *binding,
                    )
                    .filter(|_| visibility.contains(&api::ShaderStage::Task))
                    {
                        // SAFETY: the fixed slot is valid and the resource is retained through completion.
                        unsafe { render.setObjectSamplerState_atIndex(Some(raw), index) };
                    }
                    if let Some(index) = raster_slot(
                        &encoder.raster_bindings,
                        api::ShaderStage::Mesh,
                        group_slot,
                        *binding,
                    )
                    .filter(|_| visibility.contains(&api::ShaderStage::Mesh))
                    {
                        // SAFETY: the fixed slot is valid and the resource is retained through completion.
                        unsafe { render.setMeshSamplerState_atIndex(Some(raw), index) };
                    }
                    if let Some(index) = raster_slot(
                        &encoder.raster_bindings,
                        api::ShaderStage::Fragment,
                        group_slot,
                        *binding,
                    )
                    .filter(|_| visibility.contains(&api::ShaderStage::Fragment))
                    {
                        // SAFETY: index is within the fixed 28-slot ABI and
                        // the sampler is retained until command completion.
                        unsafe { render.setFragmentSamplerState_atIndex(Some(raw), index) };
                    }
                }
            }
        }
        bind_raster_sizes(encoder)?;
        Ok(())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (encoder, group_slot, group, offsets);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

#[cfg(target_vendor = "apple")]
fn raster_slot(
    bindings: &[crate::backend::shader::variant::StageBindings],
    stage: api::ShaderStage,
    group: u32,
    binding: u32,
) -> Option<usize> {
    bindings
        .iter()
        .find(|entry| entry.stage == stage)?
        .map
        .slot(group, binding)
        .map(|slot| slot.index as usize)
}
