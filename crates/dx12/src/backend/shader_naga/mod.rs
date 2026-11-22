//! Per-entry WGSL lowering to the existing DX12 raw-buffer ABI.
//!
//! Storage buffer view types must be validated at binding time.
//! It accepts vertex, fragment and compute with uniform and raw storage buffers,
//! plus supported textures and scalar samplers.
#[path = "binding.rs"]
mod binding;
mod buffer_array;
mod diagnostics;
mod dxil_container;
mod dxil_library;
mod dxil_visibility;
mod immediates;
mod layout_reflection;
mod native_layout_reflection;
mod sampler_lowering;
pub(super) use dxc::compile_ray_library;
pub(super) use dxil_container::{
    validate_atomic_dxil, validate_dual_source_outputs, validate_typed_int64_atomics,
};
pub(super) use dxil_library::validate_export as validate_dxil_library_export;
pub(super) use dxil_visibility::{
    validate as validate_dxil_visibility, validate_r32_atomic_bindings, validate_r64uint_bindings,
};
pub(super) use layout_reflection::reflect;
pub(super) use native_layout_reflection::validate_visibility as validate_native_visibility;
use sampler_lowering::{lower_sampler_arrays, lower_scalar_samplers};
#[path = "dxc.rs"]
mod dxc;
#[path = "validation.rs"]
mod validation;
#[path = "wave_size.rs"]
mod wave_size;
use binding::{validate_binding_type, wgsl_binding_shape};
use fluxel_rhi_core::api::{self, Error, Result};
use validation::{requires_f16, texture_atomic_only, validation_capabilities};
pub(crate) use wave_size::inject_direct_hlsl;

pub(super) fn requires_native_16bit(desc: &api::ShaderDesc) -> bool {
    desc.required_features.iter().any(|feature| {
        matches!(
            feature.as_str(),
            api::SHADER_FEATURE_F16 | api::SHADER_FEATURE_I16
        )
    })
}

#[cfg(test)]
#[path = "../../tests/shader_naga/compiler.rs"]
mod compiler_tests;

#[cfg(test)]
#[path = "../../tests/shader_naga/contracts.rs"]
mod contract_tests;

#[cfg(all(test, feature = "hardware-tests"))]
#[path = "../../tests/shader_naga/hardware.rs"]
mod hardware_tests;

#[cfg(test)]
#[path = "../../tests/shader_naga/sampled_texture_nonuniform.rs"]
mod sampled_texture_nonuniform_tests;

#[cfg(test)]
#[path = "../../tests/shader_naga/storage_buffer_nonuniform.rs"]
mod storage_buffer_nonuniform_tests;

#[cfg(test)]
#[path = "../../tests/shader_naga/uniform_buffer_nonuniform.rs"]
mod uniform_buffer_nonuniform_tests;

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

pub(crate) fn compile_hlsl_with_report(
    source: &str,
    entry: &str,
    stage: api::ShaderStage,
    model: u16,
    constants: &[(String, api::SpecializationValue)],
    origin: api::ShaderDiagnosticSource,
    native_16bit: bool,
) -> Result<dxc::CompiledHlsl> {
    dxc::compile_hlsl_with_origin(
        source,
        entry,
        &hlsl_profile(stage, model)?,
        constants,
        origin,
        native_16bit,
    )
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
    let standard_immediate = immediates::required(desc);
    let module = diagnostics::parse_wgsl(source, standard_immediate)?;
    buffer_array::validate(&module, &desc.entries)?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        validation_capabilities(desc),
    )
    .validate(&module)
    .map_err(|e| diagnostics::wgsl_validation(source, e))?;
    validation::validate_uniform_sizes(&module)?;
    immediates::blocks(&module, standard_immediate)?;
    for entry in &desc.entries {
        let stage = match entry.stage {
            api::ShaderStage::Compute => naga::ShaderStage::Compute,
            api::ShaderStage::Vertex => naga::ShaderStage::Vertex,
            api::ShaderStage::Fragment => naga::ShaderStage::Fragment,
            _ => return Err(Error::Unsupported("DX12 WGSL shader stage")),
        };
        let selected = module
            .entry_points
            .iter()
            .find(|ep| ep.name == entry.name && ep.stage == stage)
            .ok_or_else(|| Error::Shader(format!("missing shader entry {}", entry.name)))?;
        validation::validate_entry(&module, selected)?;
    }
    Ok(())
}

pub struct CompiledEntry {
    pub bytes: Vec<u8>,
    pub report: Option<api::ShaderCompilationReport>,
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
    compile_inner(desc, None, &[], None, None, WgslTarget::default(), false)
}

/// Compile one pipeline variant. The caller must include the entry, stage,
/// constants and ABI version in its cache key and retain the WGSL source.
pub fn compile_variant(
    desc: &api::ShaderDesc,
    entry: &api::ShaderEntryPoint,
    constants: &[(String, api::SpecializationValue)],
) -> Result<CompiledEntry> {
    compile_variant_impl(
        desc,
        entry,
        constants,
        &[],
        None,
        None,
        WgslTarget::default(),
        false,
    )
}

#[derive(Clone, Copy, Default)]
pub(super) struct WgslTarget<'a> {
    pub(super) shader_model: u16,
    pub(super) required_wave_size: Option<u32>,
    pub(super) push_constants: Option<&'a super::pipeline::PushConstantsRoot>,
}

#[cfg(test)]
pub fn compile_variant_with_typed_map(
    desc: &api::ShaderDesc,
    entry: &api::ShaderEntryPoint,
    constants: &[(String, api::SpecializationValue)],
    register_map: &[(u32, u32, u32, u8, u32)],
    binding_types: &[(u32, u32, api::BindingType)],
    runtime_binding_capacities: &[(u32, u32, u32)],
    buffer_entries: &[(u32, api::BindGroupLayoutEntry)],
    required_wave_size: Option<u32>,
) -> Result<CompiledEntry> {
    let binding_entries = binding_types
        .iter()
        .map(|(group, binding, ty)| {
            buffer_entries
                .iter()
                .find(|(other_group, layout)| group == other_group && *binding == layout.binding)
                .cloned()
                .unwrap_or_else(|| {
                    (
                        *group,
                        api::BindGroupLayoutEntry {
                            binding: *binding,
                            ty: ty.clone(),
                            visibility: vec![entry.stage],
                            count: api::BindingCount::One,
                            partially_bound: false,
                            nonuniform_indexing: false,
                        },
                    )
                })
        })
        .collect::<Vec<_>>();
    compile_pipeline_variant(
        desc,
        entry,
        constants,
        register_map,
        runtime_binding_capacities,
        &binding_entries,
        WgslTarget {
            required_wave_size,
            ..Default::default()
        },
    )
}

pub(super) fn compile_pipeline_variant(
    desc: &api::ShaderDesc,
    entry: &api::ShaderEntryPoint,
    constants: &[(String, api::SpecializationValue)],
    register_map: &[(u32, u32, u32, u8, u32)],
    runtime_binding_capacities: &[(u32, u32, u32)],
    binding_entries: &[(u32, api::BindGroupLayoutEntry)],
    target: WgslTarget<'_>,
) -> Result<CompiledEntry> {
    compile_variant_impl(
        desc,
        entry,
        constants,
        register_map,
        Some(binding_entries),
        Some(runtime_binding_capacities),
        target,
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
    compile_variant_impl(
        desc,
        entry,
        constants,
        register_map,
        None,
        None,
        WgslTarget::default(),
        true,
    )
}

fn compile_variant_impl(
    desc: &api::ShaderDesc,
    entry: &api::ShaderEntryPoint,
    constants: &[(String, api::SpecializationValue)],
    register_map: &[(u32, u32, u32, u8, u32)],
    binding_entries: Option<&[(u32, api::BindGroupLayoutEntry)]>,
    runtime_binding_capacities: Option<&[(u32, u32, u32)]>,
    target: WgslTarget<'_>,
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
        binding_entries,
        runtime_binding_capacities,
        target,
        require_layout,
    )?;
    Ok(compiled.entries.remove(0))
}

fn compile_inner(
    desc: &api::ShaderDesc,
    constants: Option<&[(String, api::SpecializationValue)]>,
    register_map: &[(u32, u32, u32, u8, u32)],
    binding_entries: Option<&[(u32, api::BindGroupLayoutEntry)]>,
    runtime_binding_capacities: Option<&[(u32, u32, u32)]>,
    target: WgslTarget<'_>,
    require_layout: bool,
) -> Result<CompiledShader> {
    if desc.format != api::ShaderFormat::Wgsl
        || desc.required_features.iter().any(|x| {
            x != api::SHADER_FEATURE_RAY_QUERY
                && x != api::SHADER_FEATURE_PUSH_CONSTANTS
                && x != api::SHADER_FEATURE_IMMEDIATE_DATA
                && x != api::SHADER_FEATURE_VIEW_INDEX
                && x != api::SHADER_FEATURE_POINTER_COMPOSITE_ACCESS
                && x != api::SHADER_FEATURE_F16
                && x != api::SHADER_FEATURE_PRIMITIVE_INDEX
                && x != api::SHADER_FEATURE_EARLY_DEPTH_TEST
                && x != super::SUBGROUP_FEATURE
                && x != "shaderClipDistance"
                && x != api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_ADD
                && x != api::SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_ADD
                && x != api::SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MIN
                && x != api::SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MAX
                && x != api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MIN
                && x != api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MAX
                && x != api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_XOR
                && x != api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_AND
                && x != api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_OR
        })
    {
        return Err(Error::Unsupported("DX12 WGSL feature requirement"));
    }
    if desc.entries.is_empty() {
        return Err(Error::Unsupported("DX12 WGSL requires entries"));
    }
    let source =
        std::str::from_utf8(&desc.bytes).map_err(|_| Error::Shader("WGSL is not UTF-8".into()))?;
    let standard_immediate = immediates::required(desc);
    let original = diagnostics::parse_wgsl(source, standard_immediate)?;
    buffer_array::validate(&original, &desc.entries)?;
    let texture_atomic_route = match desc
        .required_features
        .iter()
        .filter_map(|feature| match feature.as_str() {
            api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_ADD => {
                Some((naga::AtomicFunction::Add, naga::StorageFormat::R32Uint))
            }
            api::SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_ADD => {
                Some((naga::AtomicFunction::Add, naga::StorageFormat::R32Sint))
            }
            api::SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MIN => {
                Some((naga::AtomicFunction::Min, naga::StorageFormat::R32Sint))
            }
            api::SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MAX => {
                Some((naga::AtomicFunction::Max, naga::StorageFormat::R32Sint))
            }
            api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MIN => {
                Some((naga::AtomicFunction::Min, naga::StorageFormat::R32Uint))
            }
            api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MAX => {
                Some((naga::AtomicFunction::Max, naga::StorageFormat::R32Uint))
            }
            api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_XOR => Some((
                naga::AtomicFunction::ExclusiveOr,
                naga::StorageFormat::R32Uint,
            )),
            api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_AND => {
                Some((naga::AtomicFunction::And, naga::StorageFormat::R32Uint))
            }
            api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_OR => Some((
                naga::AtomicFunction::InclusiveOr,
                naga::StorageFormat::R32Uint,
            )),
            _ => None,
        })
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] => None,
        [operation] => Some(*operation),
        _ => {
            return Err(Error::Unsupported(
                "DX12 shader declares conflicting texture-atomic features",
            ));
        }
    };
    if let Some((operation, format)) = texture_atomic_route {
        if !texture_atomic_only(&original, operation, format) {
            return Err(Error::Unsupported(
                "DX12 WGSL texture atomic operation or storage format is outside its declared capability",
            ));
        }
    }
    if constants.is_none() && !original.overrides.is_empty() {
        return Err(Error::Unsupported("DX12 WGSL overrides"));
    }
    let original_info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        validation_capabilities(desc),
    )
    .validate(&original)
    .map_err(|e| diagnostics::wgsl_validation(source, e))?;
    validation::validate_uniform_sizes(&original)?;
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
            let (scalar, value) = match value {
                api::SpecializationValue::Bool(v) => {
                    (naga::Scalar::BOOL, if *v { 1.0 } else { 0.0 })
                }
                api::SpecializationValue::U32(v) => (naga::Scalar::U32, f64::from(*v)),
                api::SpecializationValue::I32(v) => (naga::Scalar::I32, f64::from(*v)),
                api::SpecializationValue::F32(v) if v.is_finite() => {
                    (naga::Scalar::F32, f64::from(*v))
                }
                api::SpecializationValue::F16Bits(bits) => {
                    let value = dxc::f16_to_f32(*bits);
                    if !value.is_finite() {
                        return Err(Error::Unsupported(
                            "DX12 WGSL non-finite f16 specialization",
                        ));
                    }
                    (naga::Scalar::F16, f64::from(value))
                }
                _ => return Err(Error::Unsupported("DX12 WGSL specialization type")),
            };
            if !matches!(original.types[item.ty].inner, naga::TypeInner::Scalar(s) if s == scalar) {
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
    if require_layout {
        naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
    }
    binding::specialize_runtime_arrays(&mut module, runtime_binding_capacities, require_layout)?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        validation_capabilities(desc),
    )
    .validate(&module)
    .map_err(|e| diagnostics::wgsl_validation(source, e))?;
    if require_layout {
        buffer_array::validate_bindings(&module, &info, binding_entries.unwrap_or_default())?;
    }
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
    let required_wave_size = target.required_wave_size;
    let native_f16 = requires_f16(desc);
    let immediates = immediates::blocks(&module, standard_immediate)?;
    let mut options = naga::back::hlsl::Options {
        shader_model: if target.required_wave_size.is_some() {
            naga::back::hlsl::ShaderModel::V6_6
        } else if desc
            .required_features
            .iter()
            .any(|x| x == api::SHADER_FEATURE_RAY_QUERY)
        {
            naga::back::hlsl::ShaderModel::V6_5
        } else if target.shader_model >= 64 && validation::uses_packed_dot(&module) {
            naga::back::hlsl::ShaderModel::V6_4
        } else if native_f16 {
            naga::back::hlsl::ShaderModel::V6_2
        } else if validation::requires_view_index(desc) {
            naga::back::hlsl::ShaderModel::V6_1
        } else {
            naga::back::hlsl::ShaderModel::V6_0
        },
        fake_missing_bindings: false,
        wgsl_immediate_data: standard_immediate,
        wgsl_f16_uniforms: native_f16,
        push_constants_target: (!immediates.is_empty()).then_some(naga::back::hlsl::BindTarget {
            space: super::pipeline::IMMEDIATE_SPACE,
            register: super::pipeline::IMMEDIATE_REGISTER,
            ..Default::default()
        }),
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
            naga::AddressSpace::Private
            | naga::AddressSpace::WorkGroup
            | naga::AddressSpace::PushConstant => continue,
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
    let default_shader_model = options.shader_model;
    for entry in &desc.entries {
        let stage = match entry.stage {
            api::ShaderStage::Vertex => naga::ShaderStage::Vertex,
            api::ShaderStage::Fragment => naga::ShaderStage::Fragment,
            api::ShaderStage::Compute => naga::ShaderStage::Compute,
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
        validation::validate_entry(&module, &module.entry_points[index])?;
        if require_layout {
            for &(handle, slots) in &immediates {
                if !info.get_entry_point(index)[handle].is_empty() {
                    immediates::validate_range(target.push_constants, entry.stage, slots)?;
                }
            }
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
                if let Some(layout_entries) = binding_entries {
                    let layout_entry = layout_entries
                        .iter()
                        .find(|(group, layout_entry)| {
                            *group == binding.group && layout_entry.binding == binding.binding
                        })
                        .map(|(_, layout_entry)| layout_entry)
                        .ok_or_else(|| {
                            Error::Shader(
                                "WGSL binding metadata missing from pipeline layout".into(),
                            )
                        })?;
                    if !layout_entry.visibility.contains(&entry.stage) {
                        return Err(Error::Shader(format!(
                            "WGSL binding {}:{} is not visible to {:?}",
                            binding.group, binding.binding, entry.stage
                        )));
                    }
                    validate_binding_type(&module, global, &layout_entry.ty)?;
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
            && validation::uses_entry_builtin(
                &module,
                &module.entry_points[index],
                &[naga::BuiltIn::NumWorkGroups],
            );
        if stage == naga::ShaderStage::Compute {
            super::compute::validate_workgroup_size(module.entry_points[index].workgroup_size)?;
            let entry_usage = info.get_entry_point(index);
            let bytes = module
                .global_variables
                .iter()
                .filter(|(handle, var)| {
                    var.space == naga::AddressSpace::WorkGroup && !entry_usage[*handle].is_empty()
                })
                // WGSL accounts for each live workgroup variable in 16-byte units.
                .map(|(_, var)| {
                    u64::from(module.types[var.ty].inner.size(module.to_ctx())).next_multiple_of(16)
                })
                .fold(0, u64::saturating_add);
            if bytes > u64::from(super::compute::MAX_WORKGROUP_STORAGE_SIZE) {
                return Err(Error::InvalidRange);
            }
        }
        if entry
            .workgroup_size
            .is_some_and(|size| size != module.entry_points[index].workgroup_size)
        {
            return Err(Error::Shader("WGSL workgroup size mismatch".into()));
        }
        let vertex_indices = stage == naga::ShaderStage::Vertex
            && validation::uses_entry_builtin(
                &module,
                &module.entry_points[index],
                &[naga::BuiltIn::VertexIndex, naga::BuiltIn::InstanceIndex],
            );
        options.wgsl_native_vertex_offsets = vertex_indices && target.shader_model >= 68;
        options.shader_model = if options.wgsl_native_vertex_offsets {
            naga::back::hlsl::ShaderModel::V6_8
        } else {
            default_shader_model
        };
        let profile = format!("{}_{}", stage.to_hlsl_str(), options.shader_model.to_str());
        let vertex_index_constants = vertex_indices && !options.wgsl_native_vertex_offsets;
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
        let compiled = dxc::compile_hlsl_with_origin(
            &hlsl,
            &name,
            &profile,
            &[],
            api::ShaderDiagnosticSource::GeneratedHlsl,
            native_f16,
        )?;
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
            bytes: compiled.bytes,
            report: compiled.report,
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
