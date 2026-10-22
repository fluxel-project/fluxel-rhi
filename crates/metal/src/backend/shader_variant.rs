//! Per-pipeline WGSL specialization and Metal binding assignment.
//!
//! Naga's size table follows the post-specialization global declaration order.
//! The encoder uses `size_bindings` to write exactly that table for this stage.

use super::super::{
    binding_abi::{self, SlotKind, StageBindingMap},
    bindings::MetalPipelineLayout,
};
use fluxel_rhi_core::api::{self, Error};

pub(crate) struct Variant {
    pub source: String,
    pub name: String,
    pub map: StageBindingMap,
    pub size_bindings: Vec<(u32, u32, u32)>,
    pub workgroup_size: [u32; 3],
    pub arrays: Vec<(u32, u32, u32)>,
    pub buffer_alignments: Vec<(u32, u32, u64)>,
}

fn runtime_array(module: &naga::Module, ty: naga::Handle<naga::Type>) -> bool {
    match &module.types[ty].inner {
        naga::TypeInner::Array {
            size: naga::ArraySize::Dynamic,
            ..
        } => true,
        naga::TypeInner::Struct { members, .. } => members.last().is_some_and(|member| {
            matches!(
                module.types[member.ty].inner,
                naga::TypeInner::Array {
                    size: naga::ArraySize::Dynamic,
                    ..
                }
            )
        }),
        naga::TypeInner::BindingArray { base, .. } => runtime_array(module, *base),
        _ => false,
    }
}

/// Map Metal's dense amplification ID to the pass's actual view index.
/// Naga emits one selected
/// entry point; only its generated function body is rewritten.
fn lower_view_index(mut source: String, entry: &str, slot: u8) -> api::Result<String> {
    let signature = source
        .find(&format!("{entry}("))
        .ok_or_else(|| Error::Shader("Metal view-index entry signature missing".into()))?;
    let attr = source[signature..]
        .find("[[amplification_id]]")
        .map(|n| n + signature)
        .ok_or_else(|| Error::Shader("Metal view-index argument missing".into()))?;
    let bytes = source.as_bytes();
    let mut end = attr;
    while end > signature && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    let mut start = end;
    while start > signature
        && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_')
    {
        start -= 1;
    }
    if start == end {
        return Err(Error::Shader("invalid Metal view-index argument".into()));
    }
    let argument = source[start..end].to_string();
    let open = source[signature..].find('(').unwrap() + signature;
    let mut depth = 0usize;
    let mut close = None;
    for (i, ch) in source[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close.ok_or_else(|| Error::Shader("invalid Metal entry signature".into()))?;
    if attr > close {
        return Err(Error::Shader(
            "Metal view-index input is not an entry argument".into(),
        ));
    }
    let body = source[close..]
        .find('{')
        .map(|n| n + close)
        .ok_or_else(|| Error::Shader("Metal entry body missing".into()))?;
    let mut unique = 0usize;
    let (map_name, value_name) = loop {
        let map = format!("_fluxel_view_map_{unique}");
        let value = format!("_fluxel_view_value_{unique}");
        if !source.contains(&map) && !source.contains(&value) {
            break (map, value);
        }
        unique += 1;
    };
    let mut brace_depth = 0usize;
    let mut body_end = None;
    for (i, ch) in source[body..].char_indices() {
        match ch {
            '{' => brace_depth += 1,
            '}' => {
                brace_depth -= 1;
                if brace_depth == 0 {
                    body_end = Some(body + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let body_end = body_end.ok_or_else(|| Error::Shader("invalid Metal entry body".into()))?;
    let tail = &source[body + 1..body_end];
    let mut rewritten = String::with_capacity(tail.len());
    let mut i = 0;
    while i < tail.len() {
        let ch = tail.as_bytes()[i];
        if ch.is_ascii_alphabetic() || ch == b'_' {
            let begin = i;
            i += 1;
            while i < tail.len()
                && (tail.as_bytes()[i].is_ascii_alphanumeric() || tail.as_bytes()[i] == b'_')
            {
                i += 1;
            }
            let token = &tail[begin..i];
            let previous = tail[..begin].trim_end().as_bytes().last().copied();
            rewritten.push_str(
                if token == argument && !matches!(previous, Some(b'.' | b':')) {
                    &value_name
                } else {
                    token
                },
            );
        } else {
            let ch = tail[i..].chars().next().ok_or(Error::InvalidState)?;
            rewritten.push(ch);
            i += ch.len_utf8();
        }
    }
    let prologue = format!("\n    const uint {value_name} = {map_name}[{argument}];\n");
    source.replace_range(body + 1..body_end, &(prologue + &rewritten));
    source.replace_range(
        close..close,
        &format!(", constant uint* {map_name} [[buffer({slot})]]"),
    );
    Ok(source)
}

pub(crate) fn translate(
    bytes: &[u8],
    entry: &str,
    stage: api::ShaderStage,
    constants: &[(String, api::SpecializationValue)],
    layout: &MetalPipelineLayout,
    vertex_buffers: usize,
    texture_limit: usize,
) -> api::Result<Variant> {
    let source = std::str::from_utf8(bytes).map_err(|e| Error::Shader(e.to_string()))?;
    let normalized = super::normalize_push_address_space(source);
    let mut module = naga::front::wgsl::parse_str(&normalized)
        .map_err(|e| Error::Shader(e.emit_to_string(&normalized)))?;
    // The public runtime array chooses a capacity in its layout. Specialize
    // Naga's unsized declaration to that capacity, keeping the supplied
    // bind-group length as the caller's valid-access bound.
    let dynamic_arrays: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(handle, global)| {
            let naga::TypeInner::BindingArray {
                base,
                size: naga::ArraySize::Dynamic,
            } = module.types[global.ty].inner
            else {
                return None;
            };
            Some((handle, global.binding.clone(), base))
        })
        .collect();
    for (handle, binding, base) in dynamic_arrays {
        let binding = binding.ok_or(Error::InvalidState)?;
        let item = layout
            .groups
            .get(binding.group as usize)
            .and_then(|group| {
                group
                    .entries
                    .iter()
                    .find(|item| item.binding == binding.binding)
            })
            .ok_or(Error::InvalidState)?;
        let max = match item.count {
            api::BindingCount::RuntimeSized { max_count } => max_count,
            _ => return Err(Error::InvalidState),
        };
        let size = std::num::NonZeroU32::new(max).ok_or(Error::InvalidRange)?;
        let ty = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::BindingArray {
                    base,
                    size: naga::ArraySize::Constant(size),
                },
            },
            naga::Span::UNDEFINED,
        );
        module.global_variables[handle].ty = ty;
    }
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| Error::Shader(e.to_string()))?;
    let naga_stage = match stage {
        api::ShaderStage::Vertex => naga::ShaderStage::Vertex,
        api::ShaderStage::Fragment => naga::ShaderStage::Fragment,
        api::ShaderStage::Compute => naga::ShaderStage::Compute,
        _ => return Err(Error::Unsupported("Metal WGSL shader stage")),
    };
    if !module
        .entry_points
        .iter()
        .any(|ep| ep.name == entry && ep.stage == naga_stage)
    {
        return Err(Error::Shader("Metal shader entry point mismatch".into()));
    }
    let mut values = naga::back::PipelineConstants::default();
    let mut seen = std::collections::HashSet::new();
    for (name, value) in constants {
        let item = module
            .overrides
            .iter()
            .find(|(_, item)| {
                item.id.map(|id| id.to_string()).as_deref() == Some(name.as_str())
                    || (item.id.is_none() && item.name.as_deref() == Some(name.as_str()))
            })
            .map(|(_, item)| item)
            .ok_or_else(|| Error::Shader(format!("unknown override {name}")))?;
        let (kind, value) = match value {
            api::SpecializationValue::Bool(v) => {
                (naga::ScalarKind::Bool, if *v { 1.0 } else { 0.0 })
            }
            api::SpecializationValue::U32(v) => (naga::ScalarKind::Uint, f64::from(*v)),
            api::SpecializationValue::I32(v) => (naga::ScalarKind::Sint, f64::from(*v)),
            api::SpecializationValue::F32(v) if v.is_finite() => {
                (naga::ScalarKind::Float, f64::from(*v))
            }
            _ => return Err(Error::Shader(format!("invalid WGSL override {name}"))),
        };
        if !seen.insert(name)
            || !matches!(module.types[item.ty].inner, naga::TypeInner::Scalar(s) if s.kind == kind && (kind == naga::ScalarKind::Bool || s.width == 4))
        {
            return Err(Error::Shader(format!(
                "invalid or duplicate WGSL override {name}"
            )));
        }
        values.insert(name.clone(), value);
    }
    let (module, info) = naga::back::pipeline_constants::process_overrides(
        &module,
        &info,
        Some((naga_stage, entry)),
        &values,
    )
    .map_err(|e| Error::Shader(e.to_string()))?;
    let entry_index = module
        .entry_points
        .iter()
        .position(|ep| ep.name == entry && ep.stage == naga_stage)
        .ok_or_else(|| Error::Shader("shader entry missing".into()))?;
    let has_view_index = module.entry_points[entry_index].function.arguments.iter().any(|arg| {
        matches!(arg.binding, Some(naga::Binding::BuiltIn(naga::BuiltIn::ViewIndex)))
            || matches!(&module.types[arg.ty].inner, naga::TypeInner::Struct { members, .. }
                if members.iter().any(|member| matches!(member.binding, Some(naga::Binding::BuiltIn(naga::BuiltIn::ViewIndex)))))
    });
    let mut sizes = Vec::new();
    let mut arrays = Vec::new();
    let mut layouter = naga::proc::Layouter::default();
    layouter
        .update(module.to_ctx())
        .map_err(|error| Error::Shader(error.to_string()))?;
    let mut buffer_alignments = Vec::new();
    for (handle, global) in module.global_variables.iter() {
        if !info.get_entry_point(entry_index)[handle].is_empty() {
            if matches!(
                global.space,
                naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. }
            ) {
                let binding = global.binding.as_ref().ok_or(Error::InvalidState)?;
                let ty = match module.types[global.ty].inner {
                    naga::TypeInner::BindingArray { base, .. } => base,
                    _ => global.ty,
                };
                buffer_alignments.push((
                    binding.group,
                    binding.binding,
                    u64::from(layouter[ty].alignment * 1),
                ));
            }
            if let naga::TypeInner::BindingArray { base, size } = module.types[global.ty].inner {
                let binding = global.binding.as_ref().ok_or(Error::InvalidState)?;
                if let naga::ArraySize::Constant(count) = size {
                    let item = layout
                        .groups
                        .get(binding.group as usize)
                        .and_then(|group| {
                            group
                                .entries
                                .iter()
                                .find(|item| item.binding == binding.binding)
                        })
                        .ok_or(Error::InvalidState)?;
                    let type_matches = match (&item.ty, &module.types[base].inner, global.space) {
                        (
                            api::BindingType::UniformBuffer { .. },
                            _,
                            naga::AddressSpace::Uniform,
                        ) => true,
                        (
                            api::BindingType::StorageBuffer { .. },
                            _,
                            naga::AddressSpace::Storage { .. },
                        ) => true,
                        (
                            api::BindingType::SampledTexture { .. },
                            naga::TypeInner::Image {
                                class:
                                    naga::ImageClass::Sampled { .. } | naga::ImageClass::Depth { .. },
                                ..
                            },
                            _,
                        ) => true,
                        (
                            api::BindingType::StorageTexture { .. },
                            naga::TypeInner::Image {
                                class: naga::ImageClass::Storage { .. },
                                ..
                            },
                            _,
                        ) => true,
                        (
                            api::BindingType::Sampler { comparison, .. },
                            naga::TypeInner::Sampler {
                                comparison: shader_comparison,
                            },
                            _,
                        ) => comparison == shader_comparison,
                        (
                            api::BindingType::AccelerationStructure,
                            naga::TypeInner::AccelerationStructure { .. },
                            _,
                        ) => true,
                        _ => false,
                    };
                    let capacity = match item.count {
                        api::BindingCount::Fixed(n) => n,
                        api::BindingCount::RuntimeSized { max_count } => max_count,
                        api::BindingCount::One => 0,
                    };
                    if !type_matches || capacity < count.get() {
                        return Err(Error::InvalidState);
                    }
                    arrays.push((binding.group, binding.binding, count.get()));
                }
            }
        }
        if runtime_array(&module, global.ty) {
            let binding = global
                .binding
                .as_ref()
                .ok_or_else(|| Error::Shader("runtime array without resource binding".into()))?;
            let count = if let naga::TypeInner::BindingArray {
                size: naga::ArraySize::Constant(n),
                ..
            } = module.types[global.ty].inner
            {
                let item = layout.groups.get(binding.group as usize).and_then(|group| {
                    group
                        .entries
                        .iter()
                        .find(|item| item.binding == binding.binding)
                });
                let capacity = match item.map(|item| item.count) {
                    Some(api::BindingCount::Fixed(n)) => n,
                    Some(api::BindingCount::RuntimeSized { max_count }) => max_count,
                    Some(api::BindingCount::One) => return Err(Error::InvalidState),
                    None => 0,
                };
                n.get().max(capacity)
            } else {
                1
            };
            sizes.push((binding.group, binding.binding, count));
        }
    }
    let query_bindings: Vec<_> = if module
        .types
        .iter()
        .any(|(_, ty)| matches!(ty.inner, naga::TypeInner::RayQuery { .. }))
    {
        module
            .global_variables
            .iter()
            .filter_map(|(handle, global)| {
                if info.get_entry_point(entry_index)[handle].is_empty() {
                    return None;
                }
                global
                    .binding
                    .as_ref()
                    .map(|binding| (binding.group, binding.binding))
            })
            .collect()
    } else {
        Vec::new()
    };
    let map = binding_abi::stage_map(
        &layout.groups,
        &layout.push_constants,
        stage,
        vertex_buffers,
        !sizes.is_empty(),
        has_view_index,
        &query_bindings,
        texture_limit,
    )?;
    let mut resources = naga::back::msl::BindingMap::new();
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = &global.binding else {
            continue;
        };
        let Some(slot) = map.slot(binding.group, binding.binding) else {
            continue;
        };
        let mut target = naga::back::msl::BindTarget::default();
        match slot.kind {
            SlotKind::Buffer | SlotKind::ArgumentBuffer => target.buffer = Some(slot.index),
            SlotKind::Texture => target.texture = Some(slot.index),
            SlotKind::Sampler => {
                target.sampler = Some(naga::back::msl::BindSamplerTarget::Resource(slot.index))
            }
        }
        resources.insert(binding.clone(), target);
    }
    let mut options = naga::back::msl::Options {
        fake_missing_bindings: false,
        ray_query_initialization_tracking: true,
        lang_version: if module
            .types
            .iter()
            .any(|(_, ty)| matches!(ty.inner, naga::TypeInner::RayQuery { .. }))
        {
            (2, 4)
        } else if has_view_index {
            (2, 3)
        } else {
            (2, 0)
        },
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite,
            buffer: naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite,
            image_load: naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite,
            binding_array: naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite,
        },
        ..Default::default()
    };
    options.per_entry_point_map.insert(
        entry.to_string(),
        naga::back::msl::EntryPointResources {
            resources,
            immediates_buffer: map.push_slot,
            sizes_buffer: map.sizes_slot,
        },
    );
    let (mut source, result) = naga::back::msl::write_string(
        &module,
        &info,
        &options,
        &naga::back::msl::PipelineOptions {
            entry_point: Some((naga_stage, entry.to_string())),
            binding_array_length_map: layout
                .groups
                .iter()
                .enumerate()
                .flat_map(|(group, layout)| {
                    layout.entries.iter().filter_map(move |entry| {
                        let count = match entry.count {
                            api::BindingCount::One => return None,
                            api::BindingCount::Fixed(n) => n,
                            api::BindingCount::RuntimeSized { max_count } => max_count,
                        };
                        Some((
                            naga::ResourceBinding {
                                group: group as u32,
                                binding: entry.binding,
                            },
                            count,
                        ))
                    })
                })
                .collect(),
            ..Default::default()
        },
    )
    .map_err(|e| Error::Shader(e.to_string()))?;
    let name = result
        .entry_point_names
        .into_iter()
        .next()
        .ok_or_else(|| Error::Shader("missing translated entry".into()))?
        .map_err(|e| Error::Shader(e.to_string()))?;
    if has_view_index {
        source = lower_view_index(source, &name, map.view_slot.ok_or(Error::InvalidState)?)?;
    }
    if module
        .types
        .iter()
        .any(|(_, ty)| matches!(ty.inner, naga::TypeInner::RayQuery { .. }))
    {
        let metadata_slots = map
            .ray_metadata
            .iter()
            .map(|(group, binding, slot)| {
                Ok((
                    map.slot(*group, *binding).ok_or(Error::InvalidState)?.index,
                    *slot,
                ))
            })
            .collect::<api::Result<Vec<_>>>()?;
        source = super::super::shader_ray_query::lower(source, &metadata_slots)?;
    }
    Ok(Variant {
        source,
        name,
        map,
        size_bindings: sizes,
        workgroup_size: module.entry_points[entry_index].workgroup_size,
        arrays,
        buffer_alignments,
    })
}

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
            super::super::native_array_reflection::active_buffer_slots(reflection),
            super::super::native_array_reflection::argument_buffer_slots(reflection),
        )
    } else if let Some(legacy) = legacy {
        (
            super::super::native_array_reflection::legacy_active_buffer_slots(legacy),
            super::super::native_array_reflection::legacy_argument_buffer_slots(legacy),
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
        use super::super::native_array_reflection::{NativeArrayElement as E, inspect_slot};
        let kind = match entry.ty {
            api::BindingType::UniformBuffer { .. } | api::BindingType::StorageBuffer { .. } => {
                E::Buffer
            }
            api::BindingType::SampledTexture { .. } | api::BindingType::StorageTexture { .. } => {
                E::Texture
            }
            api::BindingType::Sampler { .. } => E::Sampler,
            api::BindingType::AccelerationStructure => E::AccelerationStructure,
            api::BindingType::ExternalTexture => return Err(Error::InvalidState),
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
        super::super::native_array_reflection::buffer_alignments(reflection, &array_slots)?
    } else if let Some(legacy) = legacy {
        super::super::native_array_reflection::legacy_buffer_alignments(legacy, &array_slots)?
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
pub(crate) fn native_function(
    device: &super::super::MetalDevice,
    shader: &super::super::MetalShader,
    entry: &str,
    stage: api::ShaderStage,
    constants: &[(String, api::SpecializationValue)],
    layout: &MetalPipelineLayout,
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
    super::super::bindings::validate_array_limits(native, layout, stage)?;
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
        let variant = translate(
            wgsl,
            entry,
            stage,
            constants,
            layout,
            if stage == api::ShaderStage::Vertex {
                vertex_buffers
            } else {
                0
            },
            texture_limit,
        )?;
        if stage != api::ShaderStage::Compute && !variant.map.ray_metadata.is_empty() {
            use objc2::runtime::NSObjectProtocol as _;
            if !native.respondsToSelector(objc2::sel!(supportsRaytracingFromRender))
                || !native.supportsRaytracingFromRender()
            {
                return Err(Error::Unsupported("Metal ray query from render stage"));
            }
        }
        let library = device
            ._state
            ._device
            .newLibraryWithSource_options_error(&NSString::from_str(&variant.source), None)
            .map_err(|e| Error::Shader(format!("Metal shader variant: {e:?}")))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(&variant.name))
            .ok_or_else(|| Error::Shader("Metal shader variant entry missing".into()))?;
        return Ok((
            function.clone(),
            StageBindings {
                stage,
                map: variant.map,
                size_bindings: variant.size_bindings,
                arrays: variant.arrays,
                buffer_alignments: variant.buffer_alignments,
                function: Some(function.clone()),
            },
            Some(variant.workgroup_size),
        ));
    }
    let library = shader._library.as_ref().ok_or(Error::InvalidState)?;
    let function_name = NSString::from_str(shader.native_entry_name(entry));
    let function = if constants.is_empty() {
        library
            .newFunctionWithName(&function_name)
            .ok_or_else(|| Error::Shader("Metal shader entry missing".into()))?
    } else {
        use objc2_metal::{MTLDataType as T, MTLFunctionConstantValues};
        fn put<TValue>(values: &MTLFunctionConstantValues, name: &str, ty: T, value: &TValue) {
            let pointer = std::ptr::NonNull::from(value).cast();
            // SAFETY: pointer refers to a live value of the matching Metal scalar
            // type, and Metal copies it during this synchronous setter call.
            unsafe {
                if let Ok(index) = name.parse::<usize>() {
                    values.setConstantValue_type_atIndex(pointer, ty, index);
                } else {
                    values.setConstantValue_type_withName(pointer, ty, &NSString::from_str(name));
                }
            }
        }
        let values = MTLFunctionConstantValues::new();
        let mut seen = std::collections::HashSet::new();
        for (name, value) in constants {
            if !seen.insert(name) {
                return Err(Error::InvalidState);
            }
            match value {
                api::SpecializationValue::Bool(v) => put(&values, name, T::Bool, v),
                api::SpecializationValue::U8(v) => put(&values, name, T::UChar, v),
                api::SpecializationValue::I8(v) => put(&values, name, T::Char, v),
                api::SpecializationValue::U16(v) => put(&values, name, T::UShort, v),
                api::SpecializationValue::I16(v) => put(&values, name, T::Short, v),
                api::SpecializationValue::U32(v) => put(&values, name, T::UInt, v),
                api::SpecializationValue::I32(v) => put(&values, name, T::Int, v),
                api::SpecializationValue::U64(v) => put(&values, name, T::ULong, v),
                api::SpecializationValue::I64(v) => put(&values, name, T::Long, v),
                api::SpecializationValue::F16Bits(v) => put(&values, name, T::Half, v),
                api::SpecializationValue::F32(v) => put(&values, name, T::Float, v),
                api::SpecializationValue::F64(_) => {
                    return Err(Error::Unsupported(
                        "Metal double precision function constants",
                    ));
                }
            }
        }
        library
            .newFunctionWithName_constantValues_error(&function_name, &values)
            .map_err(|e| Error::Shader(format!("Metal function specialization: {e:?}")))?
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
    let mut arrays = Vec::new();
    for (group, layout_group) in layout.groups.iter().enumerate() {
        for entry in &layout_group.entries {
            if !entry.visibility.contains(&stage) {
                continue;
            }
            let count = match entry.count {
                api::BindingCount::One => continue,
                api::BindingCount::Fixed(n) => n,
                api::BindingCount::RuntimeSized { max_count } => max_count,
            };
            let group = u32::try_from(group).map_err(|_| Error::InvalidRange)?;
            arrays.push((group, entry.binding, count));
        }
    }
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

#[cfg(test)]
#[path = "shader_variant/tests.rs"]
mod tests;
