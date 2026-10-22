//! Compute pipeline creation and reflection.

use super::*;

pub(super) fn create_pipeline(
    device: &MetalDevice,
    desc: &api::ComputePipelineDesc<Metal>,
) -> api::Result<MetalComputePipeline> {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::{
            MTLComputePipelineState as _, MTLDevice as _, MTLFunction as _, MTLFunctionType,
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
        // SAFETY: Core checked layout provenance before invoking ComputeOps.
        let layout = unsafe { desc.layout.native_ref().clone() };
        let (function, mut binding, wgsl_size) = super::super::shader::variant::native_function(
            device,
            shader,
            &desc.compute.entry,
            api::ShaderStage::Compute,
            &desc.compute.constants,
            &layout,
            0,
            false,
        )?;
        let workgroup_size = wgsl_size
            .or(entry.workgroup_size)
            .ok_or_else(|| api::Error::Shader("Metal compute workgroup size is required".into()))?;
        if workgroup_size.contains(&0) {
            return Err(api::Error::InvalidRange);
        }
        if function.functionType() != MTLFunctionType::Kernel {
            return Err(api::Error::Shader(
                "Metal entry is not a compute kernel".into(),
            ));
        }
        let descriptor = objc2_metal::MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        let mut reflection = None;
        let mut create = || {
            device
                ._state
                ._device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &descriptor,
                    objc2_metal::MTLPipelineOption::ArgumentInfo
                        | objc2_metal::MTLPipelineOption::BufferTypeInfo,
                    Some(&mut reflection),
                )
                .map_err(|error| api::Error::Backend(format!("Metal compute pipeline: {error:?}")))
        };
        let state = if let Some(cache) = &desc.cache {
            // SAFETY: Core verified cache provenance before pipeline creation.
            unsafe { cache.native_ref() }.with_compute(&descriptor, create)?
        } else {
            create()?
        };
        let reflection = reflection.ok_or(api::Error::InvalidState)?;
        use objc2::runtime::NSObjectProtocol as _;
        if reflection.respondsToSelector(objc2::sel!(bindings)) {
            super::super::shader::variant::apply_reflected_alignment(
                &mut binding,
                &layout,
                Some(&reflection.bindings()),
                None,
            )?;
        } else {
            #[allow(deprecated)]
            super::super::shader::variant::apply_reflected_alignment(
                &mut binding,
                &layout,
                None,
                Some(&reflection.arguments()),
            )?;
        }
        if desc.required_subgroup_size.is_some_and(|required| {
            usize::try_from(required).ok() != Some(state.threadExecutionWidth())
        }) {
            return Err(api::Error::Unsupported("Metal compute subgroup size"));
        }
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
        let bindings = vec![binding];
        Ok(MetalComputePipeline {
            state,
            bindings,
            layout,
            workgroup_size,
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, desc);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}
