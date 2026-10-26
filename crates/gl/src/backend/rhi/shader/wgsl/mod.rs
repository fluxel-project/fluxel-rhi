//! WGSL parsing, profile-specific GLSL generation, and reflected binding metadata.
//!
//! Naga lowering adapts clip coordinates to the Core framebuffer convention.

use super::*;

mod constants;
use constants::constant_value;
mod multiview;
mod profile;
pub(super) use profile::{compute_supported, gl_stage};
use profile::{gl_dialect, naga_stage, naga_version, uses_gather};
mod push_ranges;
mod reflection;
pub(super) use multiview::lower_multiview;
use push_ranges::active_push_ranges;
use reflection::{collect_naga_bindings, is_storage_image, naga_storage_image_units};

/// Parses shared WGSL and emits one profile-appropriate GLSL entry point.
///
/// `WriterFlags` stay empty; the backend explicitly flips clip Y and converts
/// clip Z for offscreen textures whose first row is the Core top row.
pub(super) fn compile_wgsl(device: &GlDevice, desc: &ShaderDesc) -> Result<GlShader> {
    let source =
        std::str::from_utf8(&desc.bytes).map_err(|_| Error::Shader("WGSL is not UTF-8".into()))?;
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| Error::Shader(error.emit_to_string(source)))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| Error::Shader(error.emit_to_string(source)))?;
    for entry in &desc.entries {
        let stage = naga_stage(entry.stage)?;
        if !module
            .entry_points
            .iter()
            .any(|ep| ep.name == entry.name && ep.stage == stage)
        {
            return Err(Error::Shader(format!(
                "WGSL entry point {} not found",
                entry.name
            )));
        }
    }
    Ok(GlShader::from_inner(GlShaderInner {
        state: device.0.clone(),
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        id: None,
        source: None,
        wgsl: Some(desc.clone()),
        naga_bindings: Vec::new(),
        push_uniforms: Vec::new(),
        mip_uniforms: Vec::new(),
    }))
}

pub(super) fn lower_wgsl(
    profile: crate::backend::profile::GlProfile,
    desc: &ShaderDesc,
) -> Result<(
    crate::backend::api::GlShaderSource,
    Vec<GlNagaBindingName>,
    Vec<super::super::push::PushUniform>,
)> {
    lower_wgsl_with_constants(profile, desc, &[])
}

pub(super) fn lower_wgsl_with_constants(
    profile: crate::backend::profile::GlProfile,
    desc: &ShaderDesc,
    constants: &[(String, api::SpecializationValue)],
) -> Result<(
    crate::backend::api::GlShaderSource,
    Vec<GlNagaBindingName>,
    Vec<super::super::push::PushUniform>,
)> {
    lower_wgsl_with_parameters(profile, desc, constants, super::DrawParameters::None)
}

pub(super) fn lower_wgsl_with_parameters(
    profile: crate::backend::profile::GlProfile,
    desc: &ShaderDesc,
    constants: &[(String, api::SpecializationValue)],
    draw_parameters: super::DrawParameters,
) -> Result<(
    crate::backend::api::GlShaderSource,
    Vec<GlNagaBindingName>,
    Vec<super::super::push::PushUniform>,
)> {
    lower_wgsl_with_multiview(profile, desc, constants, draw_parameters, None)
}

pub(super) fn lower_wgsl_with_multiview(
    profile: crate::backend::profile::GlProfile,
    desc: &ShaderDesc,
    constants: &[(String, api::SpecializationValue)],
    draw_parameters: super::DrawParameters,
    multiview: Option<std::num::NonZeroU32>,
) -> Result<(
    crate::backend::api::GlShaderSource,
    Vec<GlNagaBindingName>,
    Vec<super::super::push::PushUniform>,
)> {
    lower_wgsl_with_limits(
        profile,
        desc,
        constants,
        draw_parameters,
        multiview,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        None,
    )
    .map(|(source, bindings, push, _)| (source, bindings, push))
}

pub(super) fn lower_wgsl_with_limits(
    profile: crate::backend::profile::GlProfile,
    desc: &ShaderDesc,
    constants: &[(String, api::SpecializationValue)],
    draw_parameters: super::DrawParameters,
    multiview: Option<std::num::NonZeroU32>,
    image_limit: u32,
    storage_buffer_limit: u32,
    sampled_limit: u32,
    uniform_buffer_limit: u32,
    layout: Option<&super::super::types::GlPipelineLayout>,
) -> Result<(
    crate::backend::api::GlShaderSource,
    Vec<GlNagaBindingName>,
    Vec<super::super::push::PushUniform>,
    Vec<super::super::types::GlMipUniform>,
)> {
    let [entry] = desc.entries.as_slice() else {
        return Err(Error::Unsupported("one WGSL GL entry point"));
    };
    let source =
        std::str::from_utf8(&desc.bytes).map_err(|_| Error::Shader("WGSL is not UTF-8".into()))?;
    let mut module = naga::front::wgsl::parse_str(source)
        .map_err(|error| Error::Shader(error.emit_to_string(source)))?;
    // Validate before walking the call graph: invalid WGSL can contain
    // recursive calls, but specialization assumes the validated DAG.
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| Error::Shader(error.emit_to_string(source)))?;
    let mut values = naga::back::PipelineConstants::default();
    for (key, value) in constants {
        let item = module
            .overrides
            .iter()
            .find_map(|(_, item)| {
                let name_match = item.name.as_deref() == Some(key.as_str());
                let id_match = item.id.is_some_and(|id| key.parse::<u16>() == Ok(id));
                (name_match || id_match).then_some(item)
            })
            .ok_or_else(|| Error::Shader(format!("unknown override key {key}")))?;
        let canonical = item
            .id
            .map(|id| id.to_string())
            .or_else(|| item.name.clone())
            .ok_or(Error::InvalidState)?;
        let expected = match value {
            api::SpecializationValue::Bool(_) => (naga::ScalarKind::Bool, 1),
            api::SpecializationValue::U8(_) => (naga::ScalarKind::Uint, 1),
            api::SpecializationValue::I8(_) => (naga::ScalarKind::Sint, 1),
            api::SpecializationValue::U16(_) => (naga::ScalarKind::Uint, 2),
            api::SpecializationValue::I16(_) => (naga::ScalarKind::Sint, 2),
            api::SpecializationValue::U32(_) => (naga::ScalarKind::Uint, 4),
            api::SpecializationValue::I32(_) => (naga::ScalarKind::Sint, 4),
            api::SpecializationValue::U64(_) => (naga::ScalarKind::Uint, 8),
            api::SpecializationValue::I64(_) => (naga::ScalarKind::Sint, 8),
            api::SpecializationValue::F16Bits(_) => (naga::ScalarKind::Float, 2),
            api::SpecializationValue::F32(_) => (naga::ScalarKind::Float, 4),
            api::SpecializationValue::F64(_) => (naga::ScalarKind::Float, 8),
        };
        if !matches!(module.types[item.ty].inner, naga::TypeInner::Scalar(scalar) if (scalar.kind, scalar.width) == expected)
        {
            return Err(Error::Shader(format!(
                "override {key} has a different type"
            )));
        }
        if values.insert(canonical, constant_value(*value)?).is_some() {
            return Err(Error::Shader(format!("duplicate override key {key}")));
        }
    }
    let (specialized, _) = naga::back::pipeline_constants::process_overrides(
        &module,
        &info,
        Some((naga_stage(entry.stage)?, &entry.name)),
        &values,
    )
    .map_err(|error| Error::Shader(error.to_string()))?;
    module = specialized.into_owned();
    // Lower only the selected entry and its reachable functions. Unrelated
    // entries must not consume resource units or constrain specialization.
    let selected_stage = naga_stage(entry.stage)?;
    module
        .entry_points
        .retain(|candidate| candidate.name == entry.name && candidate.stage == selected_stage);
    naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
    super::runtime_binding_arrays::specialize(&mut module, layout)?;
    let mut array_elements = std::collections::HashMap::new();
    super::binding_array::split_static(&mut module, &mut array_elements)?;
    super::binding_array_helper::lower(&mut module)?;
    super::binding_array_dynamic_sampled_query::lower(
        &mut module,
        &mut array_elements,
        sampled_limit,
    )?;
    super::binding_array_dynamic_sampled_load::lower(
        &mut module,
        &mut array_elements,
        sampled_limit,
    )?;
    super::binding_array_dynamic_sample::lower(&mut module, &mut array_elements, sampled_limit)?;
    super::binding_array_dynamic_storage_query::lower(
        &mut module,
        &mut array_elements,
        image_limit,
    )?;
    super::binding_array_dynamic::lower(&mut module, &mut array_elements, image_limit)?;
    super::binding_array_dynamic_load::lower(&mut module, &mut array_elements, image_limit)?;
    super::binding_array_dynamic_buffer::lower(
        &mut module,
        &mut array_elements,
        storage_buffer_limit,
    )?;
    super::binding_array_dynamic_buffer::lower_uniform(
        &mut module,
        &mut array_elements,
        uniform_buffer_limit,
    )?;
    let stage = match entry.stage {
        api::ShaderStage::Vertex => "vertex",
        api::ShaderStage::Fragment => "fragment",
        api::ShaderStage::Compute => "compute",
        _ => return Err(Error::InvalidState),
    };
    let mip_globals = if matches!(
        profile,
        crate::backend::profile::GlProfile::Embedded { .. }
            | crate::backend::profile::GlProfile::WebGl2
    ) {
        super::texture_num_levels::lower(&mut module, stage)?
    } else {
        Vec::new()
    };
    // Sample count is immutable allocation metadata and is also available on
    // profiles where GLSL lacks textureSamples or requires a newer version.
    let sample_globals = super::texture_num_samples::lower(&mut module, stage)?;
    let mip_uniforms = mip_globals
        .iter()
        .map(|(name, global)| (name, global, false))
        .chain(
            sample_globals
                .iter()
                .map(|(name, global)| (name, global, true)),
        )
        .map(|(name, global, samples)| {
            let element = array_elements.get(global);
            let logical = element.map_or(*global, |value| value.original);
            let binding = module.global_variables[logical]
                .binding
                .ok_or(Error::InvalidState)?;
            Ok(super::super::types::GlMipUniform {
                group: binding.group,
                binding: binding.binding,
                element: element.map(|value| value.index),
                name: name.clone(),
                samples,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    super::specialize::specialize(&mut module)?;
    let image_aliases = super::pairs::split(&mut module)?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| Error::Shader(error.emit_to_string(source)))?;
    let expected_stage = naga_stage(entry.stage)?;
    let entry_index = module
        .entry_points
        .iter()
        .position(|candidate| candidate.name == entry.name && candidate.stage == expected_stage)
        .ok_or(Error::Unsupported("WGSL GL entry point"))?;
    let entry_info = info.get_entry_point(entry_index);
    if module.global_variables.iter().any(|(handle, global)| {
        !entry_info[handle].is_empty()
            && matches!(
                module.types[global.ty].inner,
                naga::TypeInner::BindingArray { .. }
            )
    }) {
        return Err(Error::Unsupported(
            "GL shader binding array operation is not lowered",
        ));
    }
    let active_storage_buffers = module
        .global_variables
        .iter()
        .filter(|(handle, global)| {
            !entry_info[*handle].is_empty()
                && matches!(global.space, naga::AddressSpace::Storage { .. })
        })
        .count();
    if active_storage_buffers > storage_buffer_limit as usize {
        return Err(Error::Unsupported("GL shader storage buffer limit"));
    }
    let active_uniform_buffers = module
        .global_variables
        .iter()
        .filter(|(handle, global)| {
            !entry_info[*handle].is_empty() && matches!(global.space, naga::AddressSpace::Uniform)
        })
        .count();
    if active_uniform_buffers > uniform_buffer_limit as usize {
        return Err(Error::Unsupported("GL shader uniform buffer limit"));
    }
    let storage = module.global_variables.iter().any(|(handle, global)| {
        !entry_info[handle].is_empty()
            && (matches!(global.space, naga::AddressSpace::Storage { .. })
                || is_storage_image(&module, handle))
    });
    let gather = uses_gather(&module, &module.entry_points[entry_index].function);
    let texture_levels = module
        .functions
        .iter()
        .map(|(_, function)| function)
        .chain(module.entry_points.iter().map(|entry| &entry.function))
        .any(|function| {
            function.expressions.iter().any(|(_, expression)| {
                matches!(
                    expression,
                    naga::Expression::ImageQuery {
                        query: naga::ImageQuery::NumLevels,
                        ..
                    }
                )
            })
        });
    let multisampled_array = module.global_variables.iter().any(|(handle, global)| {
        !entry_info[handle].is_empty()
            && matches!(
                module.types[global.ty].inner,
                naga::TypeInner::Image {
                    dim: naga::ImageDimension::D2,
                    arrayed: true,
                    class: naga::ImageClass::Sampled { multi: true, .. }
                        | naga::ImageClass::Depth { multi: true },
                }
            )
    });
    let mut version = naga_version(profile, entry.stage, storage, gather, multisampled_array)?;
    // textureQueryLevels is core in GLSL 4.30. Core drivers are not
    // required to advertise the older ARB extension string.
    if texture_levels
        && matches!(profile, crate::backend::profile::GlProfile::Desktop { major, minor } if major > 4 || major == 4 && minor >= 3)
    {
        version = naga::back::glsl::Version::Desktop(430);
    }
    // Naga treats non-WebGL multiview as EXT_multiview and requires ESSL
    // 3.10. OVR_multiview2 itself works with ESSL 3.00; generate against
    // WebGL's identical OVR language path on ES 3.0 contexts.
    if multiview.is_some()
        && matches!(
            version,
            naga::back::glsl::Version::Embedded {
                version: 300,
                is_webgl: false
            }
        )
    {
        version = naga::back::glsl::Version::Embedded {
            version: 300,
            is_webgl: true,
        };
    }
    let image_units = naga_storage_image_units(&module, info.get_entry_point(entry_index))?;
    if image_units.len() > image_limit as usize {
        return Err(Error::Unsupported("GL shader image unit limit"));
    }
    let options = naga::back::glsl::Options {
        version,
        writer_flags: naga::back::glsl::WriterFlags::empty(),
        binding_map: image_units.clone(),
        zero_initialize_workgroup_memory: true,
    };
    let pipeline = naga::back::glsl::PipelineOptions {
        shader_stage: naga_stage(entry.stage)?,
        entry_point: entry.name.clone(),
        multiview,
    };
    let mut text = String::new();
    let mut writer = naga::back::glsl::Writer::new(
        &mut text,
        &module,
        &info,
        &options,
        &pipeline,
        naga::proc::BoundsCheckPolicies::default(),
    )
    .map_err(|error| Error::Shader(error.to_string()))?;
    let reflection = writer
        .write()
        .map_err(|error| Error::Shader(error.to_string()))?;
    for uniform in &mip_uniforms {
        let declaration = format!("uint {} = 0u;", uniform.name);
        if text.matches(&declaration).count() != 1 {
            return Err(Error::Shader(
                "private mip uniform declaration mismatch".into(),
            ));
        }
        text = text.replacen(&declaration, &format!("uniform uint {};", uniform.name), 1);
    }
    // Naga 27 incorrectly supplies an LOD to imageSize for storage-image
    // NumLayers. Unlike textureSize, GLSL imageSize takes only the image.
    for (&global, name) in &reflection.uniforms {
        if is_storage_image(&module, global) {
            text = text.replace(
                &format!("imageSize({name}, 0)"),
                &format!("imageSize({name})"),
            );
        }
    }
    for (name, mapping) in &reflection.texture_mapping {
        if is_storage_image(&module, mapping.texture) {
            text = text.replace(
                &format!("imageSize({name}, 0)"),
                &format!("imageSize({name})"),
            );
        }
    }
    // Storage images need not appear in Naga's sampler reflection. Its writer
    // emits a bare uniform identifier here, so handle those declarations too.
    let mut offset = 0;
    while let Some(found) = text[offset..].find("imageSize(") {
        let start = offset + found + "imageSize(".len();
        let Some(end) = text[start..].find(')') else {
            break;
        };
        let end = start + end;
        let arg = &text[start..end];
        if let Some(name) = arg.strip_suffix(", 0") {
            if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                let name_len = name.len();
                text.replace_range(start + name_len..end, "");
                offset = start + name_len;
                continue;
            }
        }
        offset = end + 1;
    }
    if reflection.texture_mapping.len() > sampled_limit as usize {
        return Err(Error::Unsupported("GL shader texture unit limit"));
    }
    // All GL texture storage uses Core's row order. Sampling and image
    // operations therefore use Naga's coordinates unchanged, including Cube
    // directions and cross-face filtering.
    // Multisample arrays are core in ESSL 3.20. Naga emits the OES
    // directive even for that version, but a core implementation need not
    // advertise the extension string.
    let text = if multisampled_array
        && matches!(
            version,
            naga::back::glsl::Version::Embedded {
                version: 320,
                is_webgl: false
            }
        ) {
        text.replace(
            "#extension GL_OES_texture_storage_multisample_2d_array : require\n",
            "",
        )
    } else {
        text
    };
    let text = lower_multiview(text, entry.stage, multiview)?;
    let used_push = active_push_ranges(&module, entry_index);
    let push_uniforms =
        super::super::push::reflect(&module, &reflection.push_constant_items, entry.stage)?
            .into_iter()
            .filter(|item| used_push.iter().any(|range| range.contains(&item.offset)))
            .collect();
    let bindings = collect_naga_bindings(
        &module,
        reflection,
        &image_units,
        &image_aliases,
        &array_elements,
    );
    let text = if entry.stage == api::ShaderStage::Vertex && text.contains("naga_vs_first_instance")
    {
        match draw_parameters {
            super::DrawParameters::Core => {
                version = naga::back::glsl::Version::Desktop(460);
                text.replacen("#version 400 core", "#version 460 core", 1)
                    .replacen("#version 430 core", "#version 460 core", 1)
                    .replace(
                        "uint(gl_InstanceID) + naga_vs_first_instance",
                        "uint(gl_InstanceID) + uint(gl_BaseInstance) + naga_vs_first_instance",
                    )
            }
            super::DrawParameters::Arb => {
                let text = text.replacen(
                    '\n',
                    "\n#extension GL_ARB_shader_draw_parameters : require\n",
                    1,
                );
                text.replace(
                    "uint(gl_InstanceID) + naga_vs_first_instance",
                    "uint(gl_InstanceID) + uint(gl_BaseInstanceARB) + naga_vs_first_instance",
                )
            }
            super::DrawParameters::None => text,
        }
    } else {
        text
    };
    let mut lowered = crate::backend::api::GlShaderSource {
        stage: gl_stage(entry.stage)?,
        dialect: gl_dialect(version),
        entry_point: entry.name.clone(),
        source_hash: crate::backend::api::GlShaderKey(source_key(text.as_bytes(), &entry.name)),
        text,
        spirv: None,
        debug_name: desc.label.clone(),
    };
    apply_embedded_precision_defaults(&mut lowered);
    apply_naga_clip_depth_abi(&mut lowered)?;
    lowered.source_hash = crate::backend::api::GlShaderKey(source_key(
        lowered.text.as_bytes(),
        &lowered.entry_point,
    ));
    lowered
        .validate_for(profile)
        .map_err(|_| Error::Shader("Naga emitted an incompatible GL shader".into()))?;
    Ok((lowered, bindings, push_uniforms, mip_uniforms))
}

/// Naga terminates generated entry points with `return;`. Insert before that
/// statement so the public WebGPU depth-to-GL clip conversion is reachable.
fn apply_naga_clip_depth_abi(source: &mut crate::backend::api::GlShaderSource) -> Result<()> {
    if source.stage != crate::backend::api::GlShaderStage::Vertex {
        return Ok(());
    }
    let return_at = source.text.rfind("return;").ok_or(Error::Shader(
        "Naga GLSL vertex entry has no final return".into(),
    ))?;
    source.text.insert_str(
        return_at,
        "gl_Position.y = -gl_Position.y;\n    gl_Position.z = 2.0 * gl_Position.z - gl_Position.w;\n    ",
    );
    Ok(())
}
