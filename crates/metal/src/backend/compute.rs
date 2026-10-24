//! Direct Metal compute pipeline creation and dispatch.
//!
//! RenderGraph provides the dispatch order and resource transitions. This
//! module encodes each operation on one native compute encoder and stores no
//! replay list or backend-side resource state.

use fluxel_rhi_core::api::{self, ComputeOps};
use std::{future::Future, pin::Pin};

use super::{Metal, MetalCommandEncoder, MetalComputePipeline, MetalDevice};

pub struct Compute;

impl ComputeOps<Metal> for Compute {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = Pin<Box<dyn Future<Output = api::Result<MetalComputePipeline>> + 'a>>;

    fn create_pipeline<'a>(
        device: &'a MetalDevice,
        desc: &'a api::ComputePipelineDesc<Metal>,
    ) -> Self::PipelineFuture<'a> {
        Box::pin(pipeline::create_pipeline(device, desc))
    }

    fn begin(encoder: &mut MetalCommandEncoder) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLCommandBuffer as _;
            if encoder.render.is_some() || encoder.compute.is_some() {
                return Err(api::Error::InvalidState);
            }
            encoder.compute_bindings.clear();
            encoder.compute_groups.clear();
            encoder.compute_layout = None;
            encoder.compute_push.clear();
            encoder.workgroup_size = None;
            encoder.compute = Some(if encoder.pending_statistics.is_some() {
                let descriptor = objc2_metal::MTLComputePassDescriptor::computePassDescriptor();
                super::query::configure_compute_statistics(encoder, &descriptor)?;
                encoder
                    .command
                    .computeCommandEncoderWithDescriptor(&descriptor)
                    .ok_or(api::Error::OutOfMemory)?
            } else {
                encoder
                    .command
                    .computeCommandEncoder()
                    .ok_or(api::Error::OutOfMemory)?
            });
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = encoder;
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn set_pipeline(
        encoder: &mut MetalCommandEncoder,
        pipeline: &MetalComputePipeline,
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLComputeCommandEncoder as _;
            encoder
                .compute
                .as_ref()
                .ok_or(api::Error::InvalidState)?
                .setComputePipelineState(&pipeline.state);
            encoder.workgroup_size = Some(pipeline.workgroup_size);
            encoder.compute_bindings = pipeline.bindings.clone();
            encoder.compute_layout = Some(pipeline.layout.clone());
            for (slot, (group, offsets)) in encoder.compute_groups.clone() {
                Self::set_bind_group(encoder, slot, &group, &offsets)?;
            }
            bind_compute_sizes(encoder)?;
            bind_compute_push(encoder)?;
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, pipeline);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn set_bind_group(
        encoder: &mut MetalCommandEncoder,
        group_slot: u32,
        group: &super::MetalBindGroup,
        offsets: &[u32],
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLComputeCommandEncoder as _;
            let dynamic_count = group
                .resources
                .iter()
                .filter(|resource| {
                    matches!(
                        resource,
                        super::bindings::BoundResource::Buffer {
                            dynamic_offset: true,
                            ..
                        }
                    ) || matches!(
                        resource,
                        super::bindings::BoundResource::Array {
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
                let (binding, elements, dynamic) = match resource {
                    super::bindings::BoundResource::Buffer {
                        binding,
                        dynamic_offset,
                        ..
                    } => (*binding, vec![Some(resource)], *dynamic_offset),
                    super::bindings::BoundResource::Array {
                        binding,
                        ty,
                        elements,
                        ..
                    } => (
                        *binding,
                        elements.iter().map(Option::as_ref).collect(),
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
                    ),
                    _ => (0, Vec::new(), false),
                };
                if let Some(alignment) = encoder
                    .compute_bindings
                    .iter()
                    .filter(|stage| stage.stage == api::ShaderStage::Compute)
                    .flat_map(|stage| &stage.buffer_alignments)
                    .find_map(|(group, b, alignment)| {
                        (*group == group_slot && *b == binding).then_some(*alignment)
                    })
                {
                    for element in elements.into_iter().flatten() {
                        if let super::bindings::BoundResource::Buffer { offset, .. } = element {
                            let start = offset
                                .checked_add(if dynamic {
                                    u64::from(offsets[dynamic_index])
                                } else {
                                    0
                                })
                                .ok_or(api::Error::InvalidRange)?;
                            if !start.is_multiple_of(alignment) {
                                return Err(api::Error::InvalidRange);
                            }
                        }
                    }
                }
                if let super::bindings::BoundResource::Acceleration {
                    visibility, value, ..
                } = resource
                {
                    if visibility.contains(&api::ShaderStage::Compute) {
                        value.validate_binding(encoder.ray_owner)?;
                    }
                }
                if let super::bindings::BoundResource::Buffer {
                    offset,
                    size,
                    allocation_size,
                    dynamic_offset: true,
                    offset_alignment,
                    ..
                } = resource
                {
                    super::bindings::apply_dynamic_offset(
                        *offset,
                        *size,
                        *allocation_size,
                        offsets[dynamic_index],
                        *offset_alignment,
                    )?;
                    dynamic_index += 1;
                }
                if let super::bindings::BoundResource::Array { ty, elements, .. } = resource {
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
                    if dynamic {
                        super::bindings::argument_elements(
                            ty,
                            elements,
                            elements.len() as u32,
                            Some(offsets[dynamic_index]),
                        )?;
                        dynamic_index += 1;
                    }
                }
            }
            if encoder.compute.is_none() {
                return Err(api::Error::InvalidState);
            }
            encoder
                .compute_groups
                .insert(group_slot, (group.clone(), offsets.to_vec()));
            if encoder.workgroup_size.is_none() {
                return Ok(());
            }
            let compute = encoder.compute.as_ref().ok_or(api::Error::InvalidState)?;
            let slot_for = |binding| {
                encoder
                    .compute_bindings
                    .iter()
                    .find(|item| item.stage == api::ShaderStage::Compute)
                    .and_then(|item| item.map.slot(group_slot, binding))
                    .map(|slot| slot.index as usize)
            };
            let mut dynamic_index = 0;
            for resource in &group.resources {
                match resource {
                    super::bindings::BoundResource::Array {
                        binding,
                        visibility,
                        ty,
                        elements,
                    } => {
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
                        if !visibility.contains(&api::ShaderStage::Compute) {
                            continue;
                        }
                        let Some(stage) = encoder
                            .compute_bindings
                            .iter()
                            .find(|item| item.stage == api::ShaderStage::Compute)
                        else {
                            continue;
                        };
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
                            .ok_or(api::Error::InvalidState)?;
                        let native_elements =
                            super::bindings::argument_elements(ty, elements, *capacity, offset)?;
                        if matches!(ty, api::BindingType::AccelerationStructure) {
                            if let Some((_, _, metadata_slot)) = stage
                                .map
                                .ray_metadata
                                .iter()
                                .find(|(g, b, _)| *g == group_slot && *b == *binding)
                            {
                                let structures = elements
                                    .iter()
                                    .map(|element| match element {
                                        Some(super::bindings::BoundResource::Acceleration {
                                            value,
                                            ..
                                        }) => Ok(Some(value.clone())),
                                        None => Ok(None),
                                        _ => Err(api::Error::InvalidState),
                                    })
                                    .collect::<api::Result<Vec<_>>>()?;
                                let metadata =
                                    super::ray::MetalAcceleration::query_contribution_array(
                                        &compute.device(),
                                        &structures,
                                        *capacity as usize,
                                        encoder.ray_owner,
                                    )?;
                                // SAFETY: the map reserves the slot; command buffer retains the snapshot.
                                unsafe {
                                    compute.setBuffer_offset_atIndex(
                                        Some(&metadata),
                                        0,
                                        *metadata_slot as usize,
                                    )
                                };
                            }
                        }
                        for item in elements.iter().flatten() {
                            if let super::bindings::BoundResource::Acceleration { value, .. } = item
                            {
                                value.validate_binding(encoder.ray_owner)?;
                                value.declare_dependencies(compute, encoder.ray_owner)?;
                                encoder.accelerations.push(value.clone());
                            }
                        }
                        use objc2_metal::{MTLCommandEncoder as _, MTLResourceUsage};
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
                        // SAFETY: the Naga metadata identifies a used binding-array argument at this slot.
                        let array = unsafe {
                            super::argument::materialize(
                                &compute.device(),
                                function,
                                slot.index as usize,
                                native_elements,
                            )?
                        };
                        for element in &array.elements {
                            if let Some(resource) = element.resource() {
                                unsafe { compute.useResource_usage(resource, usage) };
                            }
                        }
                        // SAFETY: slot and allocation are validated above; encoder retains the array through completion.
                        unsafe {
                            compute.setBuffer_offset_atIndex(
                                Some(&array.buffer),
                                0,
                                slot.index as usize,
                            );
                        }
                        encoder.argument_arrays.push(std::sync::Arc::new(array));
                    }
                    super::bindings::BoundResource::Acceleration {
                        binding,
                        visibility,
                        value,
                    } if visibility.contains(&api::ShaderStage::Compute) => {
                        let Some(index) = slot_for(*binding) else {
                            continue;
                        };
                        // SAFETY: the slot is bounded by the ABI and Metal's retained
                        // command buffer keeps the directly bound structure alive.
                        unsafe {
                            compute.setAccelerationStructure_atBufferIndex(Some(&value.raw), index)
                        };
                        if let Some((_, _, metadata_slot)) = encoder
                            .compute_bindings
                            .iter()
                            .find(|stage| stage.stage == api::ShaderStage::Compute)
                            .and_then(|stage| {
                                stage
                                    .map
                                    .ray_metadata
                                    .iter()
                                    .find(|(g, b, _)| *g == group_slot && *b == *binding)
                            })
                        {
                            let metadata = value.query_contributions(encoder.ray_owner)?;
                            // SAFETY: the slot is reserved for this acceleration structure;
                            // the retained command buffer keeps this snapshot alive.
                            unsafe {
                                compute.setBuffer_offset_atIndex(
                                    Some(&metadata),
                                    0,
                                    *metadata_slot as usize,
                                )
                            };
                        }
                        value.declare_dependencies(compute, encoder.ray_owner)?;
                        encoder.accelerations.push(value.clone());
                    }
                    super::bindings::BoundResource::Buffer {
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
                            super::bindings::apply_dynamic_offset(
                                *offset,
                                *size,
                                *allocation_size,
                                dynamic,
                                *offset_alignment,
                            )?
                        } else {
                            *offset
                        };
                        if visibility.contains(&api::ShaderStage::Compute) {
                            let Some(index) = slot_for(*binding) else {
                                continue;
                            };
                            // SAFETY: group/binding are bounded by the fixed 28-slot ABI;
                            // the effective aligned range was validated and the
                            // retained buffer remains live through command completion.
                            unsafe {
                                compute.setBuffer_offset_atIndex(
                                    Some(raw),
                                    effective_offset as usize,
                                    index,
                                )
                            };
                        }
                    }
                    super::bindings::BoundResource::Texture {
                        binding,
                        visibility,
                        raw,
                    } if visibility.contains(&api::ShaderStage::Compute) => {
                        let Some(index) = slot_for(*binding) else {
                            continue;
                        };
                        // SAFETY: group/binding are bounded by the fixed 28-slot ABI; the command buffer retains the texture.
                        unsafe { compute.setTexture_atIndex(Some(raw), index) };
                    }
                    super::bindings::BoundResource::Sampler {
                        binding,
                        visibility,
                        raw,
                    } if visibility.contains(&api::ShaderStage::Compute) => {
                        let Some(index) = slot_for(*binding) else {
                            continue;
                        };
                        // SAFETY: group/binding are bounded by the fixed 28-slot ABI; the command buffer retains the sampler.
                        unsafe { compute.setSamplerState_atIndex(Some(raw), index) };
                    }
                    _ => {}
                }
            }
            bind_compute_sizes(encoder)?;
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, group_slot, group, offsets);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn set_push_constants(
        encoder: &mut MetalCommandEncoder,
        offset: u32,
        bytes: &[u8],
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            let layout = encoder
                .compute_layout
                .as_ref()
                .ok_or(api::Error::InvalidState)?;
            let end = offset
                .checked_add(u32::try_from(bytes.len()).map_err(|_| api::Error::InvalidRange)?)
                .ok_or(api::Error::InvalidRange)?;
            if bytes.is_empty() || offset % 4 != 0 || bytes.len() % 4 != 0 {
                return Err(api::Error::InvalidRange);
            }
            let mut covered = offset;
            while covered < end {
                covered = layout
                    .push_constants
                    .iter()
                    .filter(|range| {
                        range.stages.contains(&api::ShaderStage::Compute)
                            && range.range.start <= covered
                            && range.range.end > covered
                    })
                    .map(|range| range.range.end)
                    .max()
                    .ok_or(api::Error::InvalidRange)?;
            }
            encoder
                .compute_push
                .resize(encoder.compute_push.len().max(end as usize), 0);
            encoder.compute_push[offset as usize..end as usize].copy_from_slice(bytes);
            bind_compute_push(encoder)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, offset, bytes);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn dispatch(encoder: &mut MetalCommandEncoder, groups: [u32; 3]) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLComputeCommandEncoder as _, MTLSize};
            let encoder_native = encoder.compute.as_ref().ok_or(api::Error::InvalidState)?;
            let [x, y, z] = encoder.workgroup_size.ok_or(api::Error::InvalidState)?;
            if groups.contains(&0) {
                return Err(api::Error::InvalidRange);
            }
            encoder_native.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups[0] as usize,
                    height: groups[1] as usize,
                    depth: groups[2] as usize,
                },
                MTLSize {
                    width: x as usize,
                    height: y as usize,
                    depth: z as usize,
                },
            );
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, groups);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn dispatch_indirect(
        encoder: &mut MetalCommandEncoder,
        arguments: &super::MetalBuffer,
        offset: u64,
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLComputeCommandEncoder as _, MTLSize};
            validate_indirect_dispatch_support(encoder.indirect_dispatch_supported)?;
            validate_indirect_dispatch(arguments.usage, arguments.size(), offset)?;
            let compute = encoder.compute.as_ref().ok_or(api::Error::InvalidState)?;
            let [x, y, z] = encoder.workgroup_size.ok_or(api::Error::InvalidState)?;
            let native_offset = usize::try_from(offset).map_err(|_| api::Error::InvalidRange)?;
            // SAFETY: validation proves INDIRECT usage, four-byte alignment,
            // and the complete 12-byte argument record is in range. The buffer
            // is retained in `indirect_arguments` and then by a command-buffer
            // completion handler because Metal permits this argument buffer
            // to be unretained by the encoded dispatch.
            unsafe {
                compute.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    &arguments.raw,
                    native_offset,
                    MTLSize {
                        width: x as usize,
                        height: y as usize,
                        depth: z as usize,
                    },
                );
            }
            encoder.indirect_arguments.push(arguments.raw.clone());
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, arguments, offset);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn end(encoder: &mut MetalCommandEncoder) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLCommandEncoder as _;
            encoder
                .compute
                .take()
                .ok_or(api::Error::InvalidState)?
                .endEncoding();
            encoder.workgroup_size = None;
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = encoder;
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn abort(encoder: &mut MetalCommandEncoder) {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(native) = encoder.compute.take() {
                use objc2_metal::MTLCommandEncoder as _;
                native.endEncoding();
            }
            encoder.workgroup_size = None;
        }
        #[cfg(not(target_vendor = "apple"))]
        let _ = encoder;
    }
}

#[cfg(target_vendor = "apple")]
fn bind_compute_sizes(encoder: &MetalCommandEncoder) -> api::Result<()> {
    use objc2_metal::{
        MTLCommandEncoder as _, MTLComputeCommandEncoder as _, MTLDevice as _, MTLResourceOptions,
    };
    let Some(stage) = encoder
        .compute_bindings
        .iter()
        .find(|stage| stage.stage == api::ShaderStage::Compute)
    else {
        return Ok(());
    };
    let Some(slot) = stage.map.sizes_slot else {
        return Ok(());
    };
    let mut sizes = Vec::new();
    for &(group, binding, count) in &stage.size_bindings {
        let resource = encoder.compute_groups.get(&group).and_then(|(group, _)| {
            group.resources.iter().find(|item| match item {
                super::bindings::BoundResource::Buffer { binding: b, .. }
                | super::bindings::BoundResource::Array { binding: b, .. } => *b == binding,
                _ => false,
            })
        });
        sizes
            .try_reserve(count as usize)
            .map_err(|_| api::Error::OutOfMemory)?;
        for index in 0..count as usize {
            let size = match resource {
                Some(super::bindings::BoundResource::Buffer { size, .. }) if index == 0 => *size,
                Some(super::bindings::BoundResource::Array { elements, .. }) => {
                    match elements.get(index).and_then(Option::as_ref) {
                        Some(super::bindings::BoundResource::Buffer { size, .. }) => *size,
                        _ => 0,
                    }
                }
                _ => 0,
            };
            sizes.push(u32::try_from(size).map_err(|_| api::Error::InvalidRange)?);
        }
    }
    if !sizes.is_empty() {
        let pointer = std::ptr::NonNull::new(sizes.as_mut_ptr().cast()).expect("nonempty sizes");
        let compute = encoder.compute.as_ref().ok_or(api::Error::InvalidState)?;
        let length = sizes.len().checked_mul(4).ok_or(api::Error::InvalidRange)?;
        if length <= 4096 {
            // SAFETY: the stage map reserves the slot and Metal copies the bytes synchronously.
            unsafe {
                compute.setBytes_length_atIndex(pointer, length, slot as usize);
            }
        } else {
            // SAFETY: the input remains valid during Metal's synchronous copy.
            let buffer = unsafe {
                compute.device().newBufferWithBytes_length_options(
                    pointer,
                    length,
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or(api::Error::OutOfMemory)?;
            // SAFETY: the command buffer retains this buffer and slot is reserved.
            unsafe {
                compute.setBuffer_offset_atIndex(Some(&buffer), 0, slot as usize);
            }
        }
    }
    Ok(())
}

#[cfg(target_vendor = "apple")]
fn bind_compute_push(encoder: &mut MetalCommandEncoder) -> api::Result<()> {
    use objc2_metal::MTLComputeCommandEncoder as _;
    let Some(stage) = encoder
        .compute_bindings
        .iter()
        .find(|stage| stage.stage == api::ShaderStage::Compute)
    else {
        return Ok(());
    };
    let Some(slot) = stage.map.push_slot else {
        return Ok(());
    };
    let layout = encoder
        .compute_layout
        .as_ref()
        .ok_or(api::Error::InvalidState)?;
    let length = layout
        .push_constants
        .iter()
        .filter(|range| range.stages.contains(&api::ShaderStage::Compute))
        .map(|range| range.range.end as usize)
        .max()
        .unwrap_or(0);
    if length == 0 {
        return Ok(());
    }
    encoder
        .compute_push
        .resize(encoder.compute_push.len().max(length), 0);
    let pointer = std::ptr::NonNull::new(encoder.compute_push.as_mut_ptr().cast())
        .expect("nonempty push data");
    let compute = encoder.compute.as_ref().ok_or(api::Error::InvalidState)?;
    if length <= 4096 {
        // SAFETY: the stage map reserves the slot and Metal copies the bytes synchronously.
        unsafe {
            compute.setBytes_length_atIndex(pointer, length, slot as usize);
        }
    } else {
        use objc2_metal::{MTLCommandBuffer as _, MTLDevice as _, MTLResourceOptions};
        // SAFETY: pointer refers to `length` initialized bytes and Metal copies them to the new buffer.
        let buffer = unsafe {
            encoder.command.device().newBufferWithBytes_length_options(
                pointer,
                length,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: slot is reserved and the command buffer retains the bound resource.
        unsafe {
            compute.setBuffer_offset_atIndex(Some(&buffer), 0, slot as usize);
        }
    }
    Ok(())
}

#[cfg(any(target_vendor = "apple", test))]
const INDIRECT_ARGUMENT_BYTES: u64 = 12;

#[cfg(target_vendor = "apple")]
pub(super) fn supports_indirect_dispatch(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> bool {
    use objc2_metal::MTLDevice as _;
    device.supportsFamily(objc2_metal::MTLGPUFamily::Apple3)
        || device.supportsFamily(objc2_metal::MTLGPUFamily::Mac2)
}

#[cfg(any(target_vendor = "apple", test))]
fn validate_indirect_dispatch(usage: api::BufferUsage, size: u64, offset: u64) -> api::Result<()> {
    if !usage.contains(api::BufferUsage::INDIRECT)
        || !offset.is_multiple_of(4)
        || offset
            .checked_add(INDIRECT_ARGUMENT_BYTES)
            .is_none_or(|end| end > size)
    {
        return Err(api::Error::InvalidRange);
    }
    Ok(())
}

#[cfg(any(target_vendor = "apple", test))]
fn validate_indirect_dispatch_support(capability: bool) -> api::Result<()> {
    if capability {
        Ok(())
    } else {
        Err(api::Error::Unsupported(
            "Metal indirect dispatch requires Apple3 or Mac2 GPU family",
        ))
    }
}

#[cfg(test)]
mod tests;

mod pipeline;
