//! Compute pipeline creation and reflection.

use super::*;

pub(super) async fn create_pipeline(
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
        let (function, mut binding, wgsl_size) =
            super::super::shader::variant::native_function_async(
                device,
                shader,
                desc.compute.entry.clone(),
                api::ShaderStage::Compute,
                desc.compute.constants.clone(),
                layout.clone(),
                0,
                false,
            )
            .await?;
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
        let archive_lease = if let Some(cache) = &desc.cache {
            // SAFETY: Core verified cache provenance before pipeline creation.
            Some(
                unsafe { cache.native_ref() }
                    .prepare_compute(&descriptor)
                    .await?,
            )
        } else {
            None
        };
        let (future, completer) = super::super::cold::pending();
        let callback_descriptor = descriptor.clone();
        let callback_function = function.clone();
        let retained_inputs = std::cell::RefCell::new(Some((
            archive_lease,
            callback_descriptor,
            callback_function,
        )));
        let callback = block2::RcBlock::new(
            move |state: *mut objc2::runtime::ProtocolObject<
                dyn objc2_metal::MTLComputePipelineState,
            >,
                  reflection: *mut objc2_metal::MTLComputePipelineReflection,
                  error: *mut objc2_foundation::NSError| {
                // Metal only lends these callback objects. Retain them before the
                // block returns; the future may be polled after autorelease pools drain.
                let result = if !error.is_null() {
                    // SAFETY: Metal supplies a valid NSError for a failed compile.
                    Err(api::Error::Backend(format!(
                        "Metal compute pipeline: {:?}",
                        unsafe { &*error }
                    )))
                } else {
                    // SAFETY: successful completion supplies both objects for the
                    // requested reflection option; retain their callback borrows.
                    let state = unsafe { objc2::rc::Retained::retain(state) }
                        .ok_or(api::Error::InvalidState);
                    let reflection = unsafe { objc2::rc::Retained::retain(reflection) }
                        .ok_or(api::Error::InvalidState);
                    state.and_then(|state| reflection.map(|reflection| (state, reflection)))
                };
                completer.finish(result);
                // Keep the descriptor, function, and archive read lease alive until
                // after Metal has completed consuming the native descriptor.
                // Completion handlers are invoked once. `take` nevertheless keeps
                // this block safe if a broken implementation invokes it again.
                drop(retained_inputs.borrow_mut().take());
            },
        );
        // SAFETY: Metal copies the block. The callback retains its output before
        // completing the cold future and keeps all descriptor inputs alive.
        unsafe {
            device
                ._state
                ._device
                .newComputePipelineStateWithDescriptor_options_completionHandler(
                    &descriptor,
                    objc2_metal::MTLPipelineOption::ArgumentInfo
                        | objc2_metal::MTLPipelineOption::BufferTypeInfo,
                    block2::RcBlock::as_ptr(&callback).cast(),
                );
        }
        let (state, reflection) = future.await?;
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
