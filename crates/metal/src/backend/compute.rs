//! Direct Metal compute pipeline creation and dispatch.
//!
//! RenderGraph provides the dispatch order and resource transitions. This
//! module encodes each operation on one native compute encoder and stores no
//! replay list or backend-side resource state.

use std::future::{Ready, ready};

use fluxel_rhi_core::api::{self, ComputeOps};

use super::{Metal, MetalCommandEncoder, MetalComputePipeline, MetalDevice};

pub struct Compute;

impl ComputeOps<Metal> for Compute {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = Ready<api::Result<MetalComputePipeline>>;

    fn create_pipeline<'a>(
        device: &'a MetalDevice,
        desc: &'a api::ComputePipelineDesc<Metal>,
    ) -> Self::PipelineFuture<'a> {
        ready(create_pipeline(device, desc))
    }

    fn begin(encoder: &mut MetalCommandEncoder) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLCommandBuffer as _;
            if encoder.render.is_some() || encoder.compute.is_some() {
                return Err(api::Error::InvalidState);
            }
            encoder.compute = Some(
                encoder
                    .command
                    .computeCommandEncoder()
                    .ok_or(api::Error::OutOfMemory)?,
            );
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
            if group_slot >= 4 {
                return Err(api::Error::InvalidRange);
            }
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
                if let super::bindings::BoundResource::Buffer {
                    offset,
                    size,
                    allocation_size,
                    dynamic_offset: true,
                    ..
                } = resource
                {
                    super::bindings::apply_dynamic_uniform_offset(
                        *offset,
                        *size,
                        *allocation_size,
                        offsets[dynamic_index],
                    )?;
                    dynamic_index += 1;
                }
            }
            let compute = encoder.compute.as_ref().ok_or(api::Error::InvalidState)?;
            let mut dynamic_index = 0;
            for resource in &group.resources {
                match resource {
                    super::bindings::BoundResource::Buffer {
                        binding,
                        visibility,
                        raw,
                        offset,
                        size,
                        allocation_size,
                        dynamic_offset,
                    } => {
                        let effective_offset = if *dynamic_offset {
                            let dynamic = offsets[dynamic_index];
                            dynamic_index += 1;
                            super::bindings::apply_dynamic_uniform_offset(
                                *offset,
                                *size,
                                *allocation_size,
                                dynamic,
                            )?
                        } else {
                            *offset
                        };
                        if visibility.contains(&api::ShaderStage::Compute) {
                            let index = metal_slot(group_slot, *binding)? as usize;
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
                        let index = metal_slot(group_slot, *binding)? as usize;
                        // SAFETY: group/binding are bounded by the fixed 28-slot ABI; the command buffer retains the texture.
                        unsafe { compute.setTexture_atIndex(Some(raw), index) };
                    }
                    super::bindings::BoundResource::Sampler {
                        binding,
                        visibility,
                        raw,
                    } if visibility.contains(&api::ShaderStage::Compute) => {
                        let index = metal_slot(group_slot, *binding)? as usize;
                        // SAFETY: group/binding are bounded by the fixed 28-slot ABI; the command buffer retains the sampler.
                        unsafe { compute.setSamplerState_atIndex(Some(raw), index) };
                    }
                    _ => {}
                }
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, group_slot, group, offsets);
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

#[cfg(target_vendor = "apple")]
fn metal_slot(group: u32, binding: u32) -> api::Result<u32> {
    if group >= 4 || binding >= 7 {
        return Err(api::Error::InvalidRange);
    }
    Ok(group * 7 + binding)
}

fn create_pipeline(
    device: &MetalDevice,
    desc: &api::ComputePipelineDesc<Metal>,
) -> api::Result<MetalComputePipeline> {
    if !desc.compute.constants.is_empty()
        || desc.required_subgroup_size.is_some()
        || desc.cache.is_some()
    {
        return Err(api::Error::Unsupported("Metal compute pipeline descriptor"));
    }
    #[cfg(target_vendor = "apple")]
    {
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLComputePipelineState as _, MTLDevice as _, MTLFunction as _, MTLFunctionType,
            MTLLibrary as _,
        };
        // SAFETY: Core checks shader/layout device identity before invoking ComputeOps.
        let shader = unsafe { desc.compute.module.native_ref() };
        let entry = shader
            .entries
            .iter()
            .find(|entry| {
                entry.name == desc.compute.entry && entry.stage == api::ShaderStage::Compute
            })
            .ok_or_else(|| {
                api::Error::Shader("Metal compute entry lacks declared workgroup size".into())
            })?;
        let workgroup_size = entry
            .workgroup_size
            .ok_or_else(|| api::Error::Shader("Metal compute workgroup size is required".into()))?;
        if workgroup_size.contains(&0) {
            return Err(api::Error::InvalidRange);
        }
        let function = shader
            ._library
            .newFunctionWithName(&NSString::from_str(&desc.compute.entry))
            .ok_or_else(|| api::Error::Shader("Metal compute entry not found".into()))?;
        if function.functionType() != MTLFunctionType::Kernel {
            return Err(api::Error::Shader(
                "Metal entry is not a compute kernel".into(),
            ));
        }
        let state = device
            ._state
            ._device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| api::Error::Backend(format!("Metal compute pipeline: {error:?}")))?;
        let threads = workgroup_size
            .into_iter()
            .try_fold(1_usize, |count, axis| {
                count
                    .checked_mul(axis as usize)
                    .ok_or(api::Error::InvalidRange)
            })?;
        if threads > state.maxTotalThreadsPerThreadgroup() {
            return Err(api::Error::Unsupported("Metal compute workgroup size"));
        }
        Ok(MetalComputePipeline {
            state,
            workgroup_size,
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, desc);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

#[cfg(test)]
mod tests {
    //! Range and usage checks for Metal's three-`u32` indirect dispatch record.

    use super::{
        INDIRECT_ARGUMENT_BYTES, validate_indirect_dispatch, validate_indirect_dispatch_support,
    };
    use fluxel_rhi_core::api::{BufferUsage, Error};

    #[test]
    fn indirect_dispatch_requires_aligned_complete_arguments_and_usage() {
        assert!(validate_indirect_dispatch(BufferUsage::INDIRECT, 24, 4).is_ok());
        assert_eq!(
            validate_indirect_dispatch(BufferUsage::INDIRECT, 24, 2),
            Err(Error::InvalidRange)
        );
        assert_eq!(
            validate_indirect_dispatch(BufferUsage::INDIRECT, 15, 4),
            Err(Error::InvalidRange)
        );
        assert_eq!(
            validate_indirect_dispatch(BufferUsage::COPY_SRC, 24, 4),
            Err(Error::InvalidRange)
        );
        assert_eq!(
            validate_indirect_dispatch(
                BufferUsage::INDIRECT,
                u64::MAX,
                u64::MAX - (INDIRECT_ARGUMENT_BYTES - 2),
            ),
            Err(Error::InvalidRange)
        );
    }

    #[test]
    fn indirect_dispatch_support_is_required_only_for_the_indirect_verb() {
        assert_eq!(
            validate_indirect_dispatch_support(false),
            Err(Error::Unsupported(
                "Metal indirect dispatch requires Apple3 or Mac2 GPU family"
            ))
        );
        assert_eq!(validate_indirect_dispatch_support(true), Ok(()));
    }
}
