//! Stage-local runtime sizes and push constant bindings.

use crate::backend::MetalCommandEncoder;
use crate::backend::bindings::BoundResource;
use fluxel_rhi_core::api;

#[cfg(target_vendor = "apple")]
pub(super) fn bind_raster_sizes(encoder: &MetalCommandEncoder) -> api::Result<()> {
    use objc2_metal::MTLRenderCommandEncoder as _;
    let render = encoder.render.as_ref().ok_or(api::Error::InvalidState)?;
    for stage in &encoder.raster_bindings {
        let Some(slot) = stage.map.sizes_slot else {
            continue;
        };
        let mut sizes = Vec::new();
        for &(group, binding, count) in &stage.size_bindings {
            let resource = encoder.raster_groups.get(&group).and_then(|(group, _)| {
                group.resources.iter().find(|item| match item {
                    BoundResource::Buffer { binding: b, .. }
                    | BoundResource::Array { binding: b, .. } => *b == binding,
                    _ => false,
                })
            });
            sizes
                .try_reserve(count as usize)
                .map_err(|_| api::Error::OutOfMemory)?;
            for index in 0..count as usize {
                let size = match resource {
                    Some(BoundResource::Buffer { size, .. }) if index == 0 => *size,
                    Some(BoundResource::Array { elements, .. }) => {
                        match elements.get(index).and_then(Option::as_ref) {
                            Some(BoundResource::Buffer { size, .. }) => *size,
                            _ => 0,
                        }
                    }
                    _ => 0,
                };
                sizes.push(u32::try_from(size).map_err(|_| api::Error::InvalidRange)?);
            }
        }
        if sizes.is_empty() {
            continue;
        }
        let pointer = std::ptr::NonNull::new(sizes.as_mut_ptr().cast()).expect("nonempty sizes");
        let length = sizes.len().checked_mul(4).ok_or(api::Error::InvalidRange)?;
        if length <= 4096 {
            // SAFETY: Metal copies these bytes and the stage map reserves this slot.
            unsafe {
                match stage.stage {
                    api::ShaderStage::Vertex => {
                        render.setVertexBytes_length_atIndex(pointer, length, slot as usize)
                    }
                    api::ShaderStage::Fragment => {
                        render.setFragmentBytes_length_atIndex(pointer, length, slot as usize)
                    }
                    api::ShaderStage::Task => {
                        render.setObjectBytes_length_atIndex(pointer, length, slot as usize)
                    }
                    api::ShaderStage::Mesh => {
                        render.setMeshBytes_length_atIndex(pointer, length, slot as usize)
                    }
                    _ => return Err(api::Error::InvalidState),
                }
            };
        } else {
            use objc2_metal::{MTLCommandEncoder as _, MTLDevice as _, MTLResourceOptions};
            // SAFETY: Metal copies the source bytes before the allocation call returns.
            let buffer = unsafe {
                render.device().newBufferWithBytes_length_options(
                    pointer,
                    length,
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or(api::Error::OutOfMemory)?;
            // SAFETY: the command buffer retains the buffer and the stage map reserves the slot.
            unsafe {
                match stage.stage {
                    api::ShaderStage::Vertex => {
                        render.setVertexBuffer_offset_atIndex(Some(&buffer), 0, slot as usize)
                    }
                    api::ShaderStage::Fragment => {
                        render.setFragmentBuffer_offset_atIndex(Some(&buffer), 0, slot as usize)
                    }
                    api::ShaderStage::Task => {
                        render.setObjectBuffer_offset_atIndex(Some(&buffer), 0, slot as usize)
                    }
                    api::ShaderStage::Mesh => {
                        render.setMeshBuffer_offset_atIndex(Some(&buffer), 0, slot as usize)
                    }
                    _ => return Err(api::Error::InvalidState),
                }
            };
        }
    }
    Ok(())
}

#[cfg(target_vendor = "apple")]
pub(in crate::backend::encoder) fn write_push(
    encoder: &mut MetalCommandEncoder,
    stages: api::StageMask,
    offset: u32,
    bytes: &[u8],
) -> api::Result<()> {
    let kinds = [
        (api::ShaderStage::Vertex, api::StageMask::VERTEX),
        (api::ShaderStage::Fragment, api::StageMask::FRAGMENT),
        (api::ShaderStage::Task, api::StageMask::TASK),
        (api::ShaderStage::Mesh, api::StageMask::MESH),
    ];
    let allowed = kinds.iter().fold(0, |mask, (_, bit)| mask | bit.0);
    let end = offset
        .checked_add(u32::try_from(bytes.len()).map_err(|_| api::Error::InvalidRange)?)
        .ok_or(api::Error::InvalidRange)?;
    if !encoder.raster_pipeline_bound
        || bytes.is_empty()
        || offset % 4 != 0
        || bytes.len() % 4 != 0
        || stages.0 == 0
        || stages.0 & !allowed != 0
    {
        return Err(api::Error::InvalidRange);
    }
    for (stage, bit) in kinds {
        if stages.0 & bit.0 == 0 {
            continue;
        }
        let mut covered = offset;
        while covered < end {
            let next = encoder
                .raster_push_ranges
                .iter()
                .filter(|r| {
                    r.stages.contains(&stage) && r.range.start <= covered && r.range.end > covered
                })
                .map(|r| r.range.end)
                .max()
                .ok_or(api::Error::InvalidRange)?;
            covered = next;
        }
    }
    for (index, (_, bit)) in kinds.iter().enumerate() {
        if stages.0 & bit.0 != 0 {
            let data = &mut encoder.raster_push[index];
            data.resize(data.len().max(end as usize), 0);
            data[offset as usize..end as usize].copy_from_slice(bytes);
        }
    }
    bind_raster_push(encoder)
}

#[cfg(target_vendor = "apple")]
pub(in crate::backend::encoder::raster) fn bind_raster_push(
    encoder: &mut MetalCommandEncoder,
) -> api::Result<()> {
    use objc2_metal::MTLRenderCommandEncoder as _;
    let render = encoder.render.as_ref().ok_or(api::Error::InvalidState)?;
    for stage in &encoder.raster_bindings {
        let Some(slot) = stage.map.push_slot else {
            continue;
        };
        let index = match stage.stage {
            api::ShaderStage::Vertex => 0,
            api::ShaderStage::Fragment => 1,
            api::ShaderStage::Task => 2,
            api::ShaderStage::Mesh => 3,
            _ => return Err(api::Error::InvalidState),
        };
        let required = encoder
            .raster_push_ranges
            .iter()
            .filter(|r| r.stages.contains(&stage.stage))
            .map(|r| r.range.end as usize)
            .max()
            .unwrap_or(0);
        if required == 0 {
            continue;
        }
        let data = &mut encoder.raster_push[index];
        data.resize(data.len().max(required), 0);
        let pointer =
            std::ptr::NonNull::new(data.as_mut_ptr().cast()).expect("nonempty push constants");
        if required > 4096 {
            use objc2_metal::{
                MTLCommandBuffer as _, MTLCommandQueue as _, MTLDevice as _, MTLResourceOptions,
            };
            let device = encoder.command.commandQueue().device();
            // SAFETY: pointer covers `required` initialized bytes; Metal copies them into the new buffer.
            let buffer = unsafe {
                device.newBufferWithBytes_length_options(
                    pointer,
                    required,
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or(api::Error::OutOfMemory)?;
            // SAFETY: stage map reserves this slot; the command buffer retains the bound buffer.
            unsafe {
                match stage.stage {
                    api::ShaderStage::Vertex => {
                        render.setVertexBuffer_offset_atIndex(Some(&buffer), 0, slot as usize)
                    }
                    api::ShaderStage::Fragment => {
                        render.setFragmentBuffer_offset_atIndex(Some(&buffer), 0, slot as usize)
                    }
                    api::ShaderStage::Task => {
                        render.setObjectBuffer_offset_atIndex(Some(&buffer), 0, slot as usize)
                    }
                    api::ShaderStage::Mesh => {
                        render.setMeshBuffer_offset_atIndex(Some(&buffer), 0, slot as usize)
                    }
                    _ => unreachable!(),
                }
            }
            continue;
        }
        // SAFETY: Metal copies the bytes; the pipeline stage map reserves this buffer slot.
        unsafe {
            match stage.stage {
                api::ShaderStage::Vertex => {
                    render.setVertexBytes_length_atIndex(pointer, required, slot as usize)
                }
                api::ShaderStage::Fragment => {
                    render.setFragmentBytes_length_atIndex(pointer, required, slot as usize)
                }
                api::ShaderStage::Task => {
                    render.setObjectBytes_length_atIndex(pointer, required, slot as usize)
                }
                api::ShaderStage::Mesh => {
                    render.setMeshBytes_length_atIndex(pointer, required, slot as usize)
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(())
}
