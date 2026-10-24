//! Per-entry WGSL lowering to the existing DX12 raw-buffer ABI.
//!
//! Storage buffer view types must be validated at binding time.
//! It accepts vertex, fragment and compute with uniform and raw storage buffers,
//! plus supported textures and scalar samplers.
#[path = "shader_naga/binding.rs"]
mod binding;
#[path = "shader_naga/dxc.rs"]
mod dxc;
#[path = "shader_naga/validation.rs"]
mod validation;
#[path = "shader_naga/wave_size.rs"]
mod wave_size;
use binding::{validate_binding_type, wgsl_binding_shape};
use fluxel_rhi_core::api::{self, Error, Result};
use validation::validation_capabilities;
pub(crate) use wave_size::{inject_direct_hlsl, wave_ops_supported};

pub(crate) fn shader_model_supported(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
) -> bool {
    use windows::Win32::Graphics::Direct3D12::*;
    let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: D3D_SHADER_MODEL_6_0,
    };
    // SAFETY: model is valid writable storage for this synchronous query.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            (&mut model as *mut D3D12_FEATURE_DATA_SHADER_MODEL).cast(),
            std::mem::size_of_val(&model) as u32,
        )
    }
    .is_ok()
        && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_0.0
}

/// Checks the explicitly requested native HLSL shader model. `Hlsl(66)` means
/// shader model 6.6; DXC receives the corresponding `*_6_6` profile.
pub(crate) fn hlsl_shader_model_supported(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    model: u16,
) -> bool {
    use windows::Win32::Graphics::Direct3D12::*;
    let Some(requested) = native_shader_model(model) else {
        return false;
    };
    let mut reported = D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: requested,
    };
    // SAFETY: `reported` is valid writable storage for this synchronous query.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            (&mut reported as *mut D3D12_FEATURE_DATA_SHADER_MODEL).cast(),
            std::mem::size_of_val(&reported) as u32,
        )
    }
    .is_ok()
        && reported.HighestShaderModel.0 >= requested.0
}

fn native_shader_model(
    model: u16,
) -> Option<windows::Win32::Graphics::Direct3D12::D3D_SHADER_MODEL> {
    // D3D12 encodes 6.0 through 6.9 as 0x60 through 0x69. Keeping this
    // conversion here makes Core's compact `major * 10 + minor` spelling
    // explicit and rejects future/other-major values until DXC support is
    // deliberately evaluated.
    (60..=69)
        .contains(&model)
        .then_some(windows::Win32::Graphics::Direct3D12::D3D_SHADER_MODEL(
            i32::from(0x60 + model - 60),
        ))
}

pub(crate) fn hlsl_profile(stage: api::ShaderStage, model: u16) -> Result<String> {
    if !(60..=69).contains(&model) {
        return Err(Error::Unsupported("DX12 HLSL shader model"));
    }
    let stage = match stage {
        api::ShaderStage::Vertex => "vs",
        api::ShaderStage::Fragment => "ps",
        api::ShaderStage::Compute => "cs",
        _ => return Err(Error::Unsupported("DX12 HLSL shader stage")),
    };
    Ok(format!("{stage}_{}_{}", model / 10, model % 10))
}

pub(crate) fn compile_direct_hlsl(
    source: &str,
    entry: &str,
    stage: api::ShaderStage,
    model: u16,
    constants: &[(String, api::SpecializationValue)],
) -> Result<Vec<u8>> {
    dxc::compile_hlsl(source, entry, &hlsl_profile(stage, model)?, constants)
}

pub(crate) fn preprocess_direct_hlsl(
    source: &str,
    constants: &[(String, api::SpecializationValue)],
) -> Result<String> {
    dxc::preprocess_hlsl(source, constants)
}

pub(crate) fn validate_source(desc: &api::ShaderDesc) -> Result<()> {
    let source =
        std::str::from_utf8(&desc.bytes).map_err(|_| Error::Shader("WGSL is not UTF-8".into()))?;
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| Error::Shader(e.emit_to_string(source)))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        validation_capabilities(desc),
    )
    .validate(&module)
    .map_err(|e| Error::Shader(e.emit_to_string(source)))?;
    for entry in &desc.entries {
        let stage = match entry.stage {
            api::ShaderStage::Compute => naga::ShaderStage::Compute,
            api::ShaderStage::Vertex => naga::ShaderStage::Vertex,
            api::ShaderStage::Fragment => naga::ShaderStage::Fragment,
            _ => return Err(Error::Unsupported("DX12 WGSL shader stage")),
        };
        if !module
            .entry_points
            .iter()
            .any(|ep| ep.name == entry.name && ep.stage == stage)
        {
            return Err(Error::Shader(format!(
                "missing shader entry {}",
                entry.name
            )));
        }
    }
    Ok(())
}

pub struct CompiledEntry {
    pub bytes: Vec<u8>,
    pub hlsl: String,
    pub name: String,
    pub stage: api::ShaderStage,
    pub translated_name: String,
    pub vertex_index_constants: bool,
    pub num_workgroups_constants: bool,
    /// Active storage bindings require Raw views for Naga's ByteAddressBuffer ABI.
    pub raw_storage_bindings: Vec<(u32, u32)>,
}
pub struct CompiledShader {
    pub entries: Vec<CompiledEntry>,
}

pub fn compile(desc: &api::ShaderDesc) -> Result<CompiledShader> {
    compile_inner(desc, None, &[], None, None, None, false)
}

/// Compile one pipeline variant. The caller must include the entry, stage,
/// constants and ABI version in its cache key and retain the WGSL source.
pub fn compile_variant(
    desc: &api::ShaderDesc,
    entry: &api::ShaderEntryPoint,
    constants: &[(String, api::SpecializationValue)],
) -> Result<CompiledEntry> {
    compile_variant_impl(desc, entry, constants, &[], None, None, None, false)
}

pub fn compile_variant_with_typed_map(
    desc: &api::ShaderDesc,
    entry: &api::ShaderEntryPoint,
    constants: &[(String, api::SpecializationValue)],
    register_map: &[(u32, u32, u32, u8, u32)],
    binding_types: &[(u32, u32, api::BindingType)],
    runtime_binding_capacities: &[(u32, u32, u32)],
    required_wave_size: Option<u32>,
) -> Result<CompiledEntry> {
    compile_variant_impl(
        desc,
        entry,
        constants,
        register_map,
        Some(binding_types),
        Some(runtime_binding_capacities),
        required_wave_size,
        true,
    )
}

#[cfg(test)]
pub fn compile_variant_with_map(
    desc: &api::ShaderDesc,
    entry: &api::ShaderEntryPoint,
    constants: &[(String, api::SpecializationValue)],
    register_map: &[(u32, u32, u32, u8, u32)],
) -> Result<CompiledEntry> {
    compile_variant_impl(desc, entry, constants, register_map, None, None, None, true)
}

fn compile_variant_impl(
    desc: &api::ShaderDesc,
    entry: &api::ShaderEntryPoint,
    constants: &[(String, api::SpecializationValue)],
    register_map: &[(u32, u32, u32, u8, u32)],
    binding_types: Option<&[(u32, u32, api::BindingType)]>,
    runtime_binding_capacities: Option<&[(u32, u32, u32)]>,
    required_wave_size: Option<u32>,
    require_layout: bool,
) -> Result<CompiledEntry> {
    if !desc
        .entries
        .iter()
        .any(|declared| declared.name == entry.name && declared.stage == entry.stage)
    {
        return Err(Error::Shader(
            "pipeline entry was not declared by the shader".into(),
        ));
    }
    let mut selected = desc.clone();
    selected.entries = vec![entry.clone()];
    let mut compiled = compile_inner(
        &selected,
        Some(constants),
        register_map,
        binding_types,
        runtime_binding_capacities,
        required_wave_size,
        require_layout,
    )?;
    Ok(compiled.entries.remove(0))
}

fn compile_inner(
    desc: &api::ShaderDesc,
    constants: Option<&[(String, api::SpecializationValue)]>,
    register_map: &[(u32, u32, u32, u8, u32)],
    binding_types: Option<&[(u32, u32, api::BindingType)]>,
    runtime_binding_capacities: Option<&[(u32, u32, u32)]>,
    required_wave_size: Option<u32>,
    require_layout: bool,
) -> Result<CompiledShader> {
    if desc.format != api::ShaderFormat::Wgsl
        || desc
            .required_features
            .iter()
            .any(|x| x != api::SHADER_FEATURE_RAY_QUERY && x != "shaderClipDistance")
    {
        return Err(Error::Unsupported("DX12 WGSL feature requirement"));
    }
    if desc.entries.is_empty() {
        return Err(Error::Unsupported("DX12 WGSL requires entries"));
    }
    let source =
        std::str::from_utf8(&desc.bytes).map_err(|_| Error::Shader("WGSL is not UTF-8".into()))?;
    let original = naga::front::wgsl::parse_str(source)
        .map_err(|e| Error::Shader(e.emit_to_string(source)))?;
    if constants.is_none() && !original.overrides.is_empty() {
        return Err(Error::Unsupported("DX12 WGSL overrides"));
    }
    let original_info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        validation_capabilities(desc),
    )
    .validate(&original)
    .map_err(|e| Error::Shader(e.emit_to_string(source)))?;
    let mut values = naga::back::PipelineConstants::default();
    if let Some(constants) = constants {
        for (name, value) in constants {
            let item = original
                .overrides
                .iter()
                .find(|(_, item)| {
                    item.id.map(|id| id.to_string()) == Some(name.clone())
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
                _ => return Err(Error::Unsupported("DX12 WGSL specialization type")),
            };
            if !matches!(original.types[item.ty].inner, naga::TypeInner::Scalar(s) if s.kind == kind && (kind == naga::ScalarKind::Bool || s.width == 4))
            {
                return Err(Error::Shader(format!("override {name} type mismatch")));
            }
            if values.insert(name.clone(), value).is_some() {
                return Err(Error::Shader(format!("duplicate override {name}")));
            }
        }
    }
    let (module, _info) = if constants.is_some() {
        let entry = &desc.entries[0];
        let stage = match entry.stage {
            api::ShaderStage::Vertex => naga::ShaderStage::Vertex,
            api::ShaderStage::Fragment => naga::ShaderStage::Fragment,
            api::ShaderStage::Compute => naga::ShaderStage::Compute,
            _ => return Err(Error::Unsupported("DX12 WGSL shader stage")),
        };
        naga::back::pipeline_constants::process_overrides(
            &original,
            &original_info,
            Some((stage, &entry.name)),
            &values,
        )
        .map_err(|e| Error::Shader(e.to_string()))?
    } else {
        (
            std::borrow::Cow::Borrowed(&original),
            std::borrow::Cow::Borrowed(&original_info),
        )
    };
    let mut module = module.into_owned();
    binding::specialize_runtime_arrays(&mut module, runtime_binding_capacities, require_layout)?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        validation_capabilities(desc),
    )
    .validate(&module)
    .map_err(|e| Error::Shader(e.emit_to_string(source)))?;
    let array_handles: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(handle, global)| match module.types[global.ty].inner {
            naga::TypeInner::BindingArray {
                base,
                size: naga::ArraySize::Constant(size),
            } if matches!(module.types[base].inner, naga::TypeInner::Sampler { .. }) => {
                Some((handle, base, size.get()))
            }
            _ => None,
        })
        .collect();
    let mut array_samplers = Vec::new();
    for (index, (handle, base, size)) in array_handles.into_iter().enumerate() {
        let global = &mut module.global_variables[handle];
        let binding = global
            .binding
            .as_ref()
            .ok_or(Error::Unsupported("DX12 WGSL unbound sampler array"))?;
        let comparison = matches!(
            module.types[base].inner,
            naga::TypeInner::Sampler { comparison: true }
        );
        let mut marker = format!("fluxel_internal_sampler_array_{index}_");
        while source.contains(&marker) {
            marker.push('_');
        }
        global.name = Some(marker.clone());
        let register = register_map
            .iter()
            .find(|&&(g, b, _, _, _)| g == binding.group && b == binding.binding)
            .map_or(binding.binding, |&(_, _, r, _, _)| r);
        array_samplers.push((binding.group, register, comparison, size, marker));
    }
    let mut options = naga::back::hlsl::Options {
        shader_model: if required_wave_size.is_some() {
            naga::back::hlsl::ShaderModel::V6_6
        } else if desc
            .required_features
            .iter()
            .any(|x| x == api::SHADER_FEATURE_RAY_QUERY)
        {
            naga::back::hlsl::ShaderModel::V6_5
        } else {
            naga::back::hlsl::ShaderModel::V6_0
        },
        fake_missing_bindings: false,
        ..Default::default()
    };
    let mut samplers = Vec::new();
    for (_, global) in module.global_variables.iter() {
        match global.space {
            naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. } => {}
            naga::AddressSpace::Handle
                if matches!(
                    module.types[global.ty].inner,
                    naga::TypeInner::Image { .. } | naga::TypeInner::AccelerationStructure { .. }
                ) => {}
            naga::AddressSpace::Handle
                if matches!(module.types[global.ty].inner,
                    naga::TypeInner::BindingArray { base, size: naga::ArraySize::Constant(_) }
                    if matches!(module.types[base].inner, naga::TypeInner::Image { .. })) => {}
            naga::AddressSpace::Handle
                if matches!(
                    module.types[global.ty].inner,
                    naga::TypeInner::Sampler { .. }
                ) =>
            {
                let binding = global
                    .binding
                    .as_ref()
                    .ok_or(Error::Unsupported("DX12 WGSL unbound sampler"))?;
                let comparison = matches!(
                    module.types[global.ty].inner,
                    naga::TypeInner::Sampler { comparison: true }
                );
                let register = register_map
                    .iter()
                    .find(|&&(g, b, _, _, _)| g == binding.group && b == binding.binding)
                    .map_or(binding.binding, |&(_, _, r, _, _)| r);
                samplers.push((binding.group, register, comparison));
                options.sampler_buffer_binding_map.insert(
                    naga::back::hlsl::SamplerIndexBufferKey {
                        group: binding.group,
                    },
                    naga::back::hlsl::BindTarget {
                        space: 255,
                        register: binding.group,
                        ..Default::default()
                    },
                );
            }
            naga::AddressSpace::Handle
                if matches!(module.types[global.ty].inner,
                    naga::TypeInner::BindingArray { base, size: naga::ArraySize::Constant(_) }
                    if matches!(module.types[base].inner, naga::TypeInner::Sampler { .. })) =>
            {
                let binding = global
                    .binding
                    .as_ref()
                    .ok_or(Error::Unsupported("DX12 WGSL unbound sampler array"))?;
                options.sampler_buffer_binding_map.insert(
                    naga::back::hlsl::SamplerIndexBufferKey {
                        group: binding.group,
                    },
                    naga::back::hlsl::BindTarget {
                        space: 255,
                        register: binding.group,
                        ..Default::default()
                    },
                );
            }
            naga::AddressSpace::Private | naga::AddressSpace::WorkGroup => continue,
            _ => return Err(Error::Unsupported("DX12 WGSL resource or address space")),
        }
        let binding = global
            .binding
            .as_ref()
            .ok_or(Error::Unsupported("DX12 WGSL unbound resource"))?;
        let space = u8::try_from(binding.group)
            .map_err(|_| Error::Unsupported("DX12 WGSL register space"))?;
        options.binding_map.insert(
            binding.clone(),
            naga::back::hlsl::BindTarget {
                space,
                register: register_map
                    .iter()
                    .find(|&&(g, b, _, _, _)| g == binding.group && b == binding.binding)
                    .map_or(binding.binding, |&(_, _, r, _, _)| r),
                ..Default::default()
            },
        );
    }
    let mut entries = Vec::new();
    for entry in &desc.entries {
        let (stage, profile) = match entry.stage {
            api::ShaderStage::Vertex => (
                naga::ShaderStage::Vertex,
                if desc
                    .required_features
                    .iter()
                    .any(|x| x == api::SHADER_FEATURE_RAY_QUERY)
                {
                    "vs_6_5"
                } else {
                    "vs_6_0"
                },
            ),
            api::ShaderStage::Fragment => (
                naga::ShaderStage::Fragment,
                if desc
                    .required_features
                    .iter()
                    .any(|x| x == api::SHADER_FEATURE_RAY_QUERY)
                {
                    "ps_6_5"
                } else {
                    "ps_6_0"
                },
            ),
            api::ShaderStage::Compute => (
                naga::ShaderStage::Compute,
                if required_wave_size.is_some() {
                    "cs_6_6"
                } else if desc
                    .required_features
                    .iter()
                    .any(|x| x == api::SHADER_FEATURE_RAY_QUERY)
                {
                    "cs_6_5"
                } else {
                    "cs_6_0"
                },
            ),
            _ => return Err(Error::Unsupported("DX12 WGSL shader stage")),
        };
        if entries.iter().any(|existing: &CompiledEntry| {
            existing.name == entry.name && existing.stage == entry.stage
        }) {
            return Err(Error::Shader("duplicate WGSL entry".into()));
        }
        let index = module
            .entry_points
            .iter()
            .position(|e| e.name == entry.name && e.stage == stage)
            .ok_or_else(|| Error::Shader("WGSL entry not found".into()))?;
        if require_layout {
            for (handle, global) in module.global_variables.iter() {
                if info.get_entry_point(index)[handle].is_empty() {
                    continue;
                }
                let Some(binding) = &global.binding else {
                    continue;
                };
                let (_, _, _, actual_class, actual_count) = register_map
                    .iter()
                    .find(|&&(group, slot, _, _, _)| {
                        group == binding.group && slot == binding.binding
                    })
                    .ok_or_else(|| {
                        Error::Shader("WGSL resource missing from pipeline layout".into())
                    })?;
                if let Some(types) = binding_types {
                    let ty = types
                        .iter()
                        .find(|(group, slot, _)| {
                            *group == binding.group && *slot == binding.binding
                        })
                        .map(|(_, _, ty)| ty)
                        .ok_or_else(|| {
                            Error::Shader("WGSL binding type missing from pipeline layout".into())
                        })?;
                    validate_binding_type(&module, global, ty)?;
                }
                let (class, count) = wgsl_binding_shape(&module, global)?;
                if *actual_class != class || *actual_count < count {
                    return Err(Error::Shader(
                        "WGSL resource type or count does not match pipeline layout".into(),
                    ));
                }
            }
        }
        let num_workgroups_constants = stage == naga::ShaderStage::Compute
            && module.entry_points[index]
                .function
                .arguments
                .iter()
                .any(|argument| {
                    uses_num_workgroups(&module, argument.ty, argument.binding.as_ref())
                });
        if entry
            .workgroup_size
            .is_some_and(|size| size != module.entry_points[index].workgroup_size)
        {
            return Err(Error::Shader("WGSL workgroup size mismatch".into()));
        }
        let vertex_index_constants = stage == naga::ShaderStage::Vertex
            && module.entry_points[index]
                .function
                .arguments
                .iter()
                .any(|argument| uses_base_index(&module, argument.ty, argument.binding.as_ref()));
        if (vertex_index_constants || num_workgroups_constants)
            && module.global_variables.iter().any(|(_, global)| {
                global
                    .binding
                    .as_ref()
                    .is_some_and(|binding| binding.group == 255)
            })
        {
            return Err(Error::Unsupported(
                "DX12 WGSL vertex constants register space",
            ));
        }
        options.special_constants_binding = (vertex_index_constants || num_workgroups_constants)
            .then_some(naga::back::hlsl::BindTarget {
                space: 255,
                register: 0,
                ..Default::default()
            });
        let pipeline = naga::back::hlsl::PipelineOptions {
            entry_point: Some((stage, entry.name.clone())),
        };
        let mut hlsl = String::new();
        let reflection = naga::back::hlsl::Writer::new(&mut hlsl, &options, &pipeline)
            .write(&module, &info, None)
            .map_err(|e| Error::Shader(e.to_string()))?;
        let name = reflection
            .entry_point_names
            .first()
            .ok_or_else(|| Error::Shader("missing HLSL entry reflection".into()))?
            .as_ref()
            .map_err(|e| Error::Shader(format!("{e:?}")))?
            .clone();
        if !array_samplers.is_empty() {
            hlsl = lower_sampler_arrays(&hlsl, &array_samplers)?;
        }
        if !samplers.is_empty() || !array_samplers.is_empty() {
            let groups: Vec<u32> = samplers
                .iter()
                .map(|s| s.0)
                .chain(array_samplers.iter().map(|s| s.0))
                .collect();
            hlsl = lower_scalar_samplers(&hlsl, &samplers, &groups)?;
        }
        if let Some(size) = required_wave_size {
            if stage != naga::ShaderStage::Compute {
                return Err(Error::InvalidState);
            }
            hlsl = wave_size::inject(&hlsl, &name, size)?;
        }
        let bytes = dxc::compile_hlsl(&hlsl, &name, profile, &[])?;
        let raw_storage_bindings = module
            .global_variables
            .iter()
            .filter_map(|(handle, global)| {
                if matches!(global.space, naga::AddressSpace::Storage { .. })
                    && !info.get_entry_point(index)[handle].is_empty()
                {
                    global
                        .binding
                        .as_ref()
                        .map(|binding| (binding.group, binding.binding))
                } else {
                    None
                }
            })
            .collect();
        entries.push(CompiledEntry {
            bytes,
            hlsl,
            name: entry.name.clone(),
            stage: entry.stage,
            translated_name: name,
            vertex_index_constants,
            num_workgroups_constants,
            raw_storage_bindings,
        });
    }
    Ok(CompiledShader { entries })
}

/// Replace only Naga 27's exact sampler-array indirection with descriptor-table
/// arrays. Unexpected generated syntax is rejected rather than miscompiled.
fn lower_sampler_arrays(source: &str, arrays: &[(u32, u32, bool, u32, String)]) -> Result<String> {
    let mut output = source.to_owned();
    for (group, binding, comparison, count, marker) in arrays {
        let prefix = format!("static const uint {marker}");
        let line = output
            .lines()
            .find(|line| line.starts_with(&prefix))
            .ok_or(Error::Unsupported("DX12 WGSL sampler array declaration"))?;
        let (left, value) = line
            .split_once(" = ")
            .ok_or(Error::Unsupported("DX12 WGSL sampler array declaration"))?;
        if value != format!("{binding};") {
            return Err(Error::Unsupported("DX12 WGSL sampler array binding"));
        }
        let name = left
            .strip_prefix("static const uint ")
            .ok_or(Error::InvalidState)?
            .to_owned();
        let replacement = format!(
            "{} {name}[{count}] : register(s{binding}, space{group});",
            if *comparison {
                "SamplerComparisonState"
            } else {
                "SamplerState"
            }
        );
        output = output.replacen(line, &replacement, 1);
        let heap = if *comparison {
            "nagaComparisonSamplerHeap"
        } else {
            "nagaSamplerHeap"
        };
        for nonuniform in [false, true] {
            let prefix = if nonuniform {
                format!(
                    "{heap}[NonUniformResourceIndex(nagaGroup{group}SamplerIndexArray[{name} + "
                )
            } else {
                format!("{heap}[nagaGroup{group}SamplerIndexArray[{name} + ")
            };
            while let Some(start) = output.find(&prefix) {
                let expr_start = start + prefix.len();
                let mut depth = 0usize;
                let mut end = None;
                for (offset, ch) in output[expr_start..].char_indices() {
                    match ch {
                        '[' => depth += 1,
                        ']' if depth == 0 => {
                            end = Some(expr_start + offset);
                            break;
                        }
                        ']' => depth -= 1,
                        _ => {}
                    }
                }
                let end = end.ok_or(Error::Unsupported("DX12 WGSL sampler array expression"))?;
                let suffix = if nonuniform { "])]" } else { "]]" };
                if !output[end..].starts_with(suffix) {
                    return Err(Error::Unsupported("DX12 WGSL sampler array expression"));
                }
                let expr = &output[expr_start..end];
                let replacement = if nonuniform {
                    format!("{name}[NonUniformResourceIndex({expr})]")
                } else {
                    format!("{name}[{expr}]")
                };
                output.replace_range(start..end + suffix.len(), &replacement);
            }
        }
    }
    Ok(output)
}

/// Naga 27 emits a sampler heap plus an index buffer even for scalar samplers.
/// Rewrite only its exact, pinned scalar declaration form into root-table slots;
/// reject any unexpected generated form instead of silently changing the ABI.
fn lower_scalar_samplers(
    source: &str,
    samplers: &[(u32, u32, bool)],
    groups: &[u32],
) -> Result<String> {
    let mut output = String::new();
    let mut seen = Vec::new();
    for line in source.lines() {
        if line == "SamplerState nagaSamplerHeap[2048]: register(s0, space0);"
            || line
                == "SamplerComparisonState nagaComparisonSamplerHeap[2048]: register(s0, space1);"
        {
            continue;
        }
        let mut index_decl = false;
        for &group in groups {
            if line
                == format!(
                    "StructuredBuffer<uint> nagaGroup{group}SamplerIndexArray : register(t{group}, space255);"
                )
            {
                index_decl = true;
            }
        }
        if index_decl {
            continue;
        }
        let declaration = line
            .strip_prefix("static const SamplerState ")
            .map(|rest| (rest, false))
            .or_else(|| {
                line.strip_prefix("static const SamplerComparisonState ")
                    .map(|rest| (rest, true))
            });
        if let Some((rest, comparison)) = declaration {
            let (name, expr) = rest
                .split_once(" = ")
                .ok_or(Error::Unsupported("DX12 WGSL sampler lowering"))?;
            let mut matched = None;
            for &(group, binding, expected_comparison) in samplers {
                let heap = if comparison {
                    "nagaComparisonSamplerHeap"
                } else {
                    "nagaSamplerHeap"
                };
                if comparison == expected_comparison
                    && expr == format!("{heap}[nagaGroup{group}SamplerIndexArray[{binding}]];")
                {
                    matched = Some((group, binding));
                    break;
                }
            }
            let (group, binding) =
                matched.ok_or(Error::Unsupported("DX12 WGSL sampler lowering"))?;
            output.push_str(&format!(
                "{} {name} : register(s{binding}, space{group});\n",
                if comparison {
                    "SamplerComparisonState"
                } else {
                    "SamplerState"
                }
            ));
            seen.push((group, binding, comparison));
            continue;
        }
        output.push_str(line);
        output.push('\n');
    }
    if (source.contains("nagaSamplerHeap") || source.contains("nagaComparisonSamplerHeap"))
        && (samplers.iter().any(|item| !seen.contains(item))
            || output.contains("nagaSamplerHeap")
            || output.contains("nagaComparisonSamplerHeap")
            || output.contains("SamplerIndexArray"))
    {
        return Err(Error::Unsupported("DX12 WGSL sampler lowering"));
    }
    Ok(output)
}

fn uses_base_index(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    binding: Option<&naga::Binding>,
) -> bool {
    if matches!(
        binding,
        Some(naga::Binding::BuiltIn(
            naga::BuiltIn::VertexIndex | naga::BuiltIn::InstanceIndex
        ))
    ) {
        return true;
    }
    if let naga::TypeInner::Struct { members, .. } = &module.types[ty].inner {
        return members
            .iter()
            .any(|member| uses_base_index(module, member.ty, member.binding.as_ref()));
    }
    false
}

fn uses_num_workgroups(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    binding: Option<&naga::Binding>,
) -> bool {
    if matches!(
        binding,
        Some(naga::Binding::BuiltIn(naga::BuiltIn::NumWorkGroups))
    ) {
        return true;
    }
    if let naga::TypeInner::Struct { members, .. } = &module.types[ty].inner {
        return members
            .iter()
            .any(|member| uses_num_workgroups(module, member.ty, member.binding.as_ref()));
    }
    false
}
