//! Native Metal function construction and reflection for shader variants.
//!
//! This boundary owns Objective-C library/function lifetimes and completion
//! handlers. The parent variant module remains CPU-only Naga translation.

use super::*;

/// Shader-visible resource assignment retained by a native pipeline stage.
#[derive(Clone)]
pub(crate) struct StageBindings {
    pub stage: api::ShaderStage,
    pub map: StageBindingMap,
    pub size_bindings: Vec<(u32, u32, u32)>,
    pub arrays: Vec<(u32, u32, u32)>,
    pub buffer_alignments: Vec<(u32, u32, u64)>,
    #[cfg(target_vendor = "apple")]
    pub function:
        Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLFunction>>>,
}

#[cfg(target_vendor = "apple")]
pub(crate) fn apply_reflected_alignment(
    stage: &mut StageBindings,
    layout: &MetalPipelineLayout,
    reflection: Option<
        &objc2_foundation::NSArray<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBinding>>,
    >,
    legacy: Option<&objc2_foundation::NSArray<objc2_metal::MTLArgument>>,
) -> api::Result<()> {
    let mut buffer_slots = Vec::new();
    let mut array_slots = Vec::new();
    let (active_buffers, active_arrays) = if let Some(reflection) = reflection {
        (
            super::super::super::native_array_reflection::active_buffer_slots(reflection),
            super::super::super::native_array_reflection::argument_buffer_slots(reflection),
        )
    } else if let Some(legacy) = legacy {
        (
            super::super::super::native_array_reflection::legacy_active_buffer_slots(legacy),
            super::super::super::native_array_reflection::legacy_argument_buffer_slots(legacy),
        )
    } else {
        return Err(Error::Unsupported("Metal pipeline buffer reflection"));
    };
    let mut retained_arrays = Vec::new();
    for &(group, binding, count) in &stage.arrays {
        let entry = layout
            .groups
            .get(group as usize)
            .and_then(|g| g.entries.iter().find(|e| e.binding == binding))
            .ok_or(Error::InvalidState)?;
        let slot = stage
            .map
            .slot(group, binding)
            .ok_or(Error::InvalidState)?
            .index as usize;
        if !active_buffers.contains(&slot) {
            continue;
        }
        if !active_arrays.contains(&slot) {
            return Err(Error::Shader(
                "Metal active buffer does not match the argument-array ABI".into(),
            ));
        }
        use super::super::super::native_array_reflection::{NativeArrayElement as E, inspect_slot};
        let kind = match entry.ty {
            api::BindingType::UniformBuffer { .. } | api::BindingType::StorageBuffer { .. } => {
                E::Buffer
            }
            api::BindingType::SampledTexture { .. } | api::BindingType::StorageTexture { .. } => {
                E::Texture
            }
            api::BindingType::Sampler { .. } => E::Sampler,
            api::BindingType::AccelerationStructure => E::AccelerationStructure,
            api::BindingType::VideoFrame => return Err(Error::InvalidState),
        };
        if !inspect_slot(
            stage.function.as_ref().ok_or(Error::InvalidState)?,
            slot,
            kind,
        )? {
            return Err(Error::Shader(
                "Metal reflected argument buffer is unavailable".into(),
            ));
        }
        retained_arrays.push((group, binding, count));
    }
    stage.arrays = retained_arrays;
    for &(group, binding, slot) in &stage.map.entries {
        let Some(entry) = layout
            .groups
            .get(group as usize)
            .and_then(|g| g.entries.iter().find(|e| e.binding == binding))
        else {
            return Err(Error::InvalidState);
        };
        if !matches!(
            entry.ty,
            api::BindingType::UniformBuffer { .. } | api::BindingType::StorageBuffer { .. }
        ) {
            continue;
        }
        buffer_slots.push((group, binding, slot.index as usize));
        if entry.count != api::BindingCount::One {
            array_slots.push(slot.index as usize);
        }
    }
    let reflected = if let Some(reflection) = reflection {
        super::super::super::native_array_reflection::buffer_alignments(reflection, &array_slots)?
    } else if let Some(legacy) = legacy {
        super::super::super::native_array_reflection::legacy_buffer_alignments(
            legacy,
            &array_slots,
        )?
    } else {
        return Err(Error::Unsupported("Metal pipeline buffer reflection"));
    };
    for (slot, alignment) in reflected {
        if let Some(&(group, binding, _)) = buffer_slots.iter().find(|(_, _, index)| *index == slot)
        {
            if let Some((_, _, existing)) = stage
                .buffer_alignments
                .iter_mut()
                .find(|(g, b, _)| *g == group && *b == binding)
            {
                *existing = (*existing).max(alignment);
            } else {
                stage.buffer_alignments.push((group, binding, alignment));
            }
        }
    }
    Ok(())
}

#[cfg(target_vendor = "apple")]
pub(crate) async fn native_function_async(
    device: &super::super::MetalDevice,
    shader: &super::super::MetalShader,
    entry: String,
    stage: api::ShaderStage,
    constants: Vec<(String, api::SpecializationValue)>,
    layout: MetalPipelineLayout,
    vertex_buffers: usize,
    multiview: bool,
) -> api::Result<(
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLFunction>>,
    StageBindings,
    Option<[u32; 3]>,
)> {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLDevice as _, MTLGPUFamily, MTLLibrary as _};

    let native = &device._state._device;
    super::super::super::bindings::validate_array_limits(native, &layout, stage)?;
    let texture_limit = if native.supportsFamily(MTLGPUFamily::Apple6)
        || native.supportsFamily(MTLGPUFamily::Mac2)
    {
        128
    } else if native.supportsFamily(MTLGPUFamily::Apple4) {
        96
    } else {
        31
    };
    if let Some(wgsl) = &shader.wgsl_source {
        // The worker receives only owned Rust layout/source data. It never sees
        // an Objective-C object or runs on a Metal callback queue.
        let variant = super::super::super::cold_cpu::spawn_cpu({
            let bytes = wgsl.clone();
            move || {
                translate(
                    &bytes,
                    &entry,
                    stage,
                    &constants,
                    &layout,
                    if stage == api::ShaderStage::Vertex {
                        vertex_buffers
                    } else {
                        0
                    },
                    texture_limit,
                )
            }
        })
        .await?;
        if stage != api::ShaderStage::Compute && !variant.map.ray_metadata.is_empty() {
            use objc2::runtime::NSObjectProtocol as _;
            if !native.respondsToSelector(objc2::sel!(supportsRaytracingFromRender))
                || !native.supportsRaytracingFromRender()
            {
                return Err(Error::Unsupported("Metal ray query from render stage"));
            }
        }
        let ray_query = !variant.map.ray_metadata.is_empty();
        let translated_name = variant.name.clone();
        let compiled = super::super::compile_library_async(
            device,
            std::sync::Arc::from(variant.source.into_bytes()),
            Vec::new(),
            Vec::new(),
            ray_query,
        )
        .await?;
        let library = compiled._library.ok_or(Error::InvalidState)?;
        let function = library
            .newFunctionWithName(&NSString::from_str(&translated_name))
            .ok_or_else(|| Error::Shader("Metal shader variant entry missing".into()))?;
        let binding = StageBindings {
            stage,
            map: variant.map,
            size_bindings: variant.size_bindings,
            arrays: variant.arrays,
            buffer_alignments: variant.buffer_alignments,
            function: Some(function.clone()),
        };
        return Ok((function, binding, Some(variant.workgroup_size)));
    }
    let library = shader._library.as_ref().ok_or(Error::InvalidState)?;
    let function_name = NSString::from_str(shader.native_entry_name(&entry));
    let function = if constants.is_empty() {
        library
            .newFunctionWithName(&function_name)
            .ok_or_else(|| Error::Shader("Metal shader entry missing".into()))?
    } else {
        specialize_function_async(library, function_name, constants).await?
    };
    let map = binding_abi::stage_map(
        &layout.groups,
        &layout.push_constants,
        stage,
        if stage == api::ShaderStage::Vertex {
            vertex_buffers
        } else {
            0
        },
        false,
        multiview,
        &[],
        texture_limit,
    )?;
    let arrays = layout
        .groups
        .iter()
        .enumerate()
        .flat_map(|(group, layout_group)| {
            layout_group.entries.iter().filter_map(move |item| {
                item.visibility
                    .contains(&stage)
                    .then(|| match item.count {
                        api::BindingCount::One => None,
                        api::BindingCount::Fixed(n) => Some((group as u32, item.binding, n)),
                        api::BindingCount::RuntimeSized { max_count } => {
                            Some((group as u32, item.binding, max_count))
                        }
                    })
                    .flatten()
            })
        })
        .collect();
    Ok((
        function.clone(),
        StageBindings {
            stage,
            map,
            size_bindings: Vec::new(),
            arrays,
            buffer_alignments: Vec::new(),
            function: Some(function),
        },
        None,
    ))
}

#[cfg(target_vendor = "apple")]
async fn specialize_function_async(
    library: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLLibrary>,
    name: objc2::rc::Retained<objc2_foundation::NSString>,
    constants: Vec<(String, api::SpecializationValue)>,
) -> api::Result<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLFunction>>>
{
    use objc2::Message as _;
    use objc2::rc::Retained;
    use objc2_metal::{MTLDataType as T, MTLFunctionConstantValues, MTLLibrary as _};
    fn put<V>(values: &MTLFunctionConstantValues, name: &str, ty: T, value: &V) {
        let pointer = std::ptr::NonNull::from(value).cast();
        // SAFETY: Metal copies the matching scalar during this setter call.
        unsafe {
            if let Ok(i) = name.parse() {
                values.setConstantValue_type_atIndex(pointer, ty, i)
            } else {
                values.setConstantValue_type_withName(
                    pointer,
                    ty,
                    &objc2_foundation::NSString::from_str(name),
                )
            }
        }
    }
    let values = MTLFunctionConstantValues::new();
    let mut seen = std::collections::HashSet::new();
    for (key, value) in &constants {
        if !seen.insert(key) {
            return Err(Error::InvalidState);
        }
        match value {
            api::SpecializationValue::Bool(v) => put(&values, key, T::Bool, v),
            api::SpecializationValue::U8(v) => put(&values, key, T::UChar, v),
            api::SpecializationValue::I8(v) => put(&values, key, T::Char, v),
            api::SpecializationValue::U16(v) => put(&values, key, T::UShort, v),
            api::SpecializationValue::I16(v) => put(&values, key, T::Short, v),
            api::SpecializationValue::U32(v) => put(&values, key, T::UInt, v),
            api::SpecializationValue::I32(v) => put(&values, key, T::Int, v),
            api::SpecializationValue::U64(v) => put(&values, key, T::ULong, v),
            api::SpecializationValue::I64(v) => put(&values, key, T::Long, v),
            api::SpecializationValue::F16Bits(v) => put(&values, key, T::Half, v),
            api::SpecializationValue::F32(v) => put(&values, key, T::Float, v),
            api::SpecializationValue::F64(_) => {
                return Err(Error::Unsupported(
                    "Metal double precision function constants",
                ));
            }
        }
    }
    let (future, completer) = super::super::super::cold::pending();
    let retained_library = library.retain();
    let retained_name = name.clone();
    let retained_values = values.clone();
    let callback = block2::RcBlock::new(move |function, error| {
        let _keep = (&retained_library, &retained_name, &retained_values);
        let result = unsafe { Retained::retain(function) }.ok_or_else(|| {
            let error = unsafe { Retained::retain(error) };
            Error::Shader(format!("Metal function specialization: {error:?}"))
        });
        completer.finish(result);
    });
    // SAFETY: Metal copies the block; its captures retain library/name/values until callback entry.
    unsafe {
        library.newFunctionWithName_constantValues_completionHandler(&name, &values, &callback);
    }
    future.await
}
