//! WGSL parsing, profile-specific GLSL generation, and reflected binding metadata.
//!
//! Naga lowering adapts clip coordinates to the Core framebuffer convention.

use super::*;

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
    profile: crate::backend::gl::profile::GlProfile,
    desc: &ShaderDesc,
) -> Result<(
    crate::backend::gl::api::GlShaderSource,
    Vec<GlNagaBindingName>,
    Vec<super::super::push::PushUniform>,
)> {
    lower_wgsl_with_constants(profile, desc, &[])
}

pub(super) fn lower_wgsl_with_constants(
    profile: crate::backend::gl::profile::GlProfile,
    desc: &ShaderDesc,
    constants: &[(String, api::SpecializationValue)],
) -> Result<(
    crate::backend::gl::api::GlShaderSource,
    Vec<GlNagaBindingName>,
    Vec<super::super::push::PushUniform>,
)> {
    lower_wgsl_with_parameters(profile, desc, constants, super::DrawParameters::None)
}

pub(super) fn lower_wgsl_with_parameters(
    profile: crate::backend::gl::profile::GlProfile,
    desc: &ShaderDesc,
    constants: &[(String, api::SpecializationValue)],
    draw_parameters: super::DrawParameters,
) -> Result<(
    crate::backend::gl::api::GlShaderSource,
    Vec<GlNagaBindingName>,
    Vec<super::super::push::PushUniform>,
)> {
    lower_wgsl_with_multiview(profile, desc, constants, draw_parameters, None)
}

pub(super) fn lower_wgsl_with_multiview(
    profile: crate::backend::gl::profile::GlProfile,
    desc: &ShaderDesc,
    constants: &[(String, api::SpecializationValue)],
    draw_parameters: super::DrawParameters,
    multiview: Option<std::num::NonZeroU32>,
) -> Result<(
    crate::backend::gl::api::GlShaderSource,
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
    )
    .map(|(source, bindings, push, _)| (source, bindings, push))
}

pub(super) fn lower_wgsl_with_limits(
    profile: crate::backend::gl::profile::GlProfile,
    desc: &ShaderDesc,
    constants: &[(String, api::SpecializationValue)],
    draw_parameters: super::DrawParameters,
    multiview: Option<std::num::NonZeroU32>,
    image_limit: u32,
    storage_buffer_limit: u32,
    sampled_limit: u32,
    uniform_buffer_limit: u32,
) -> Result<(
    crate::backend::gl::api::GlShaderSource,
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
        crate::backend::gl::profile::GlProfile::Embedded { .. }
            | crate::backend::gl::profile::GlProfile::WebGl2
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
        && matches!(profile, crate::backend::gl::profile::GlProfile::Desktop { major, minor } if major > 4 || major == 4 && minor >= 3)
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
    let mut lowered = crate::backend::gl::api::GlShaderSource {
        stage: gl_stage(entry.stage)?,
        dialect: gl_dialect(version),
        entry_point: entry.name.clone(),
        source_hash: crate::backend::gl::api::GlShaderKey(source_key(text.as_bytes(), &entry.name)),
        text,
        debug_name: desc.label.clone(),
    };
    apply_embedded_precision_defaults(&mut lowered);
    apply_naga_clip_depth_abi(&mut lowered)?;
    lowered.source_hash = crate::backend::gl::api::GlShaderKey(source_key(
        lowered.text.as_bytes(),
        &lowered.entry_point,
    ));
    lowered
        .validate_for(profile)
        .map_err(|_| Error::Shader("Naga emitted an incompatible GL shader".into()))?;
    Ok((lowered, bindings, push_uniforms, mip_uniforms))
}

fn uses_gather(module: &naga::Module, function: &naga::Function) -> bool {
    if function.expressions.iter().any(|(_, expr)| {
        matches!(
            expr,
            naga::Expression::ImageSample {
                gather: Some(_),
                ..
            }
        )
    }) {
        return true;
    }
    fn block(module: &naga::Module, statements: &naga::Block) -> bool {
        statements.iter().any(|statement| match statement {
            naga::Statement::Call { function, .. } => {
                uses_gather(module, &module.functions[*function])
            }
            naga::Statement::Block(inner) => block(module, inner),
            naga::Statement::If { accept, reject, .. } => {
                block(module, accept) || block(module, reject)
            }
            naga::Statement::Switch { cases, .. } => {
                cases.iter().any(|case| block(module, &case.body))
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => block(module, body) || block(module, continuing),
            _ => false,
        })
    }
    block(module, &function.body)
}

fn is_storage_image(module: &naga::Module, global: naga::Handle<naga::GlobalVariable>) -> bool {
    matches!(
        module.types[module.global_variables[global].ty].inner,
        naga::TypeInner::Image {
            class: naga::ImageClass::Storage { .. },
            ..
        }
    )
}

fn collect_naga_bindings(
    module: &naga::Module,
    reflection: naga::back::glsl::ReflectionInfo,
    image_units: &naga::back::glsl::BindingMap,
    image_aliases: &std::collections::HashMap<
        naga::Handle<naga::GlobalVariable>,
        naga::Handle<naga::GlobalVariable>,
    >,
    array_elements: &std::collections::HashMap<
        naga::Handle<naga::GlobalVariable>,
        super::binding_array::Element,
    >,
) -> Vec<GlNagaBindingName> {
    let mut bindings = Vec::new();
    for (global, name) in reflection.uniforms {
        push_binding(
            &mut bindings,
            module,
            global,
            name,
            image_units,
            array_elements,
        );
    }
    for (name, mapping) in reflection.texture_mapping {
        // GLSL has a combined texture/sampler uniform. Both WebGPU bindings
        // must resolve to that one reflected name for layout validation.
        push_binding(
            &mut bindings,
            module,
            image_aliases
                .get(&mapping.texture)
                .copied()
                .unwrap_or(mapping.texture),
            name.clone(),
            image_units,
            array_elements,
        );
        if let Some(sampler) = mapping.sampler {
            push_binding(
                &mut bindings,
                module,
                sampler,
                name,
                image_units,
                array_elements,
            );
        }
    }
    bindings.sort_by(|left, right| {
        (left.group, left.binding, left.element, &left.name).cmp(&(
            right.group,
            right.binding,
            right.element,
            &right.name,
        ))
    });
    bindings.dedup_by(|left, right| {
        left.group == right.group
            && left.binding == right.binding
            && left.element == right.element
            && left.name == right.name
    });
    bindings
}

fn push_binding(
    output: &mut Vec<GlNagaBindingName>,
    module: &naga::Module,
    global: naga::Handle<naga::GlobalVariable>,
    name: String,
    image_units: &naga::back::glsl::BindingMap,
    array_elements: &std::collections::HashMap<
        naga::Handle<naga::GlobalVariable>,
        super::binding_array::Element,
    >,
) {
    let native_binding = module.global_variables[global].binding;
    let element = array_elements.get(&global);
    let logical_global = element.map_or(global, |element| element.original);
    if let Some(binding) = module.global_variables[logical_global].binding {
        output.push(GlNagaBindingName {
            group: binding.group,
            binding: binding.binding,
            element: element.map(|value| value.index),
            name,
            image_unit: native_binding
                .and_then(|binding| image_units.get(&binding).copied())
                .map(u32::from),
            resource_type: naga_resource_type(module, global),
        });
    }
}

fn naga_resource_type(
    module: &naga::Module,
    global: naga::Handle<naga::GlobalVariable>,
) -> Option<super::super::types::GlNagaResourceType> {
    use super::super::types::GlNagaResourceType as R;
    use api::{TextureSampleType as S, TextureViewDimension as D};
    match module.types[module.global_variables[global].ty].inner {
        naga::TypeInner::Sampler { comparison } => Some(R::Sampler { comparison }),
        naga::TypeInner::Image {
            dim,
            arrayed,
            class,
        } => {
            let dimension = match (dim, arrayed) {
                (naga::ImageDimension::D1, false) => D::D1,
                (naga::ImageDimension::D2, false) => D::D2,
                (naga::ImageDimension::D2, true) => D::D2Array,
                (naga::ImageDimension::D3, false) => D::D3,
                (naga::ImageDimension::Cube, false) => D::Cube,
                (naga::ImageDimension::Cube, true) => D::CubeArray,
                _ => return None,
            };
            let (sample, multisampled) = match class {
                naga::ImageClass::Sampled { kind, multi } => (
                    match kind {
                        naga::ScalarKind::Float => S::Float,
                        naga::ScalarKind::Sint => S::Sint,
                        naga::ScalarKind::Uint => S::Uint,
                        _ => return None,
                    },
                    multi,
                ),
                naga::ImageClass::Depth { multi } => (S::Depth, multi),
                naga::ImageClass::Storage { format, access } => {
                    let format = naga_storage_format(format)?;
                    let read = access.contains(naga::StorageAccess::LOAD);
                    let write = access.contains(naga::StorageAccess::STORE);
                    let access = match (read, write) {
                        (true, true) => api::StorageAccess::ReadWrite,
                        (true, false) => api::StorageAccess::Read,
                        (false, true) => api::StorageAccess::Write,
                        _ => return None,
                    };
                    return Some(R::StorageTexture {
                        dimension,
                        format,
                        access,
                    });
                }
                naga::ImageClass::External => return None,
            };
            Some(R::SampledTexture {
                dimension,
                sample,
                multisampled,
            })
        }
        _ => None,
    }
}

fn naga_storage_format(format: naga::StorageFormat) -> Option<api::TextureFormat> {
    use api::TextureFormat as T;
    use naga::StorageFormat as N;
    Some(match format {
        N::R8Unorm => T::R8Unorm,
        N::R8Snorm => T::R8Snorm,
        N::R8Uint => T::R8Uint,
        N::R8Sint => T::R8Sint,
        N::R16Uint => T::R16Uint,
        N::R16Sint => T::R16Sint,
        N::R16Float => T::R16Float,
        N::Rg8Unorm => T::Rg8Unorm,
        N::Rg8Snorm => T::Rg8Snorm,
        N::Rg8Uint => T::Rg8Uint,
        N::Rg8Sint => T::Rg8Sint,
        N::R32Uint => T::R32Uint,
        N::R32Sint => T::R32Sint,
        N::R32Float => T::R32Float,
        N::Rg16Uint => T::Rg16Uint,
        N::Rg16Sint => T::Rg16Sint,
        N::Rg16Float => T::Rg16Float,
        N::Rgba8Unorm => T::Rgba8Unorm,
        N::Rgba8Snorm => T::Rgba8Snorm,
        N::Rgba8Uint => T::Rgba8Uint,
        N::Rgba8Sint => T::Rgba8Sint,
        N::Bgra8Unorm => T::Bgra8Unorm,
        N::Rgb10a2Uint => T::Rgb10a2Uint,
        N::Rgb10a2Unorm => T::Rgb10a2Unorm,
        N::Rg11b10Ufloat => T::Rg11b10Ufloat,
        N::R64Uint => T::R64Uint,
        N::Rg32Uint => T::Rg32Uint,
        N::Rg32Sint => T::Rg32Sint,
        N::Rg32Float => T::Rg32Float,
        N::Rgba16Uint => T::Rgba16Uint,
        N::Rgba16Sint => T::Rgba16Sint,
        N::Rgba16Float => T::Rgba16Float,
        N::Rgba32Uint => T::Rgba32Uint,
        N::Rgba32Sint => T::Rgba32Sint,
        N::Rgba32Float => T::Rgba32Float,
        N::R16Unorm => T::R16Unorm,
        N::R16Snorm => T::R16Snorm,
        N::Rg16Unorm => T::Rg16Unorm,
        N::Rg16Snorm => T::Rg16Snorm,
        N::Rgba16Unorm => T::Rgba16Unorm,
        N::Rgba16Snorm => T::Rgba16Snorm,
    })
}

/// Assigns active WGSL storage images dense GLSL image units in logical
/// group/binding order. GLES image uniforms cannot be updated with
/// `glUniform1i`; the same unit is emitted as a GLSL binding qualifier and
/// carried into bind-group execution.
fn naga_storage_image_units(
    module: &naga::Module,
    entry: &naga::valid::FunctionInfo,
) -> Result<naga::back::glsl::BindingMap> {
    let mut resources = module
        .global_variables
        .iter()
        .filter_map(|(handle, global)| {
            let binding = global.binding.as_ref()?;
            (!entry[handle].is_empty() && is_storage_image(module, handle))
                .then_some(binding.clone())
        })
        .collect::<Vec<_>>();
    resources.sort_by_key(|binding| (binding.group, binding.binding));
    resources.dedup();
    resources
        .into_iter()
        .enumerate()
        .map(|(unit, binding)| {
            let unit =
                u8::try_from(unit).map_err(|_| Error::Unsupported("too many storage images"))?;
            Ok((binding, unit))
        })
        .collect()
}

fn naga_version(
    profile: crate::backend::gl::profile::GlProfile,
    stage: api::ShaderStage,
    storage: bool,
    gather: bool,
    multisampled_array: bool,
) -> Result<naga::back::glsl::Version> {
    let compute = stage == api::ShaderStage::Compute || storage;
    Ok(match profile {
        crate::backend::gl::profile::GlProfile::Embedded { major, minor }
            if multisampled_array && (major > 3 || major == 3 && minor >= 2) =>
        {
            naga::back::glsl::Version::Embedded {
                version: 320,
                is_webgl: false,
            }
        }
        crate::backend::gl::profile::GlProfile::Desktop { minor, .. } if compute && minor >= 3 => {
            naga::back::glsl::Version::Desktop(430)
        }
        crate::backend::gl::profile::GlProfile::Desktop { .. } if !compute => {
            naga::back::glsl::Version::Desktop(400)
        }
        crate::backend::gl::profile::GlProfile::Embedded { minor, .. } if compute && minor >= 1 => {
            naga::back::glsl::Version::Embedded {
                version: 310,
                is_webgl: false,
            }
        }
        crate::backend::gl::profile::GlProfile::Embedded { minor, .. }
            if (gather || multisampled_array) && !compute && minor >= 1 =>
        {
            naga::back::glsl::Version::Embedded {
                version: 310,
                is_webgl: false,
            }
        }
        crate::backend::gl::profile::GlProfile::Embedded { .. }
            if !compute && !gather && !multisampled_array =>
        {
            naga::back::glsl::Version::Embedded {
                version: 300,
                is_webgl: false,
            }
        }
        crate::backend::gl::profile::GlProfile::WebGl2
            if !compute && !gather && !multisampled_array =>
        {
            naga::back::glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            }
        }
        _ => return Err(Error::Unsupported("GL shader stage/profile")),
    })
}

#[cfg(test)]
#[test]
fn multisampled_array_selects_gles_310_only_when_available() {
    use crate::backend::gl::profile::GlProfile;
    use naga::back::glsl::Version;
    let stage = api::ShaderStage::Fragment;
    assert_eq!(
        naga_version(
            GlProfile::Embedded { major: 3, minor: 1 },
            stage,
            false,
            false,
            true
        )
        .unwrap(),
        Version::Embedded {
            version: 310,
            is_webgl: false
        },
    );
    assert_eq!(
        naga_version(
            GlProfile::Embedded { major: 3, minor: 2 },
            stage,
            false,
            false,
            true
        )
        .unwrap(),
        Version::Embedded {
            version: 320,
            is_webgl: false
        },
    );
    assert_eq!(
        naga_version(
            GlProfile::Embedded { major: 3, minor: 0 },
            stage,
            false,
            false,
            false
        )
        .unwrap(),
        Version::Embedded {
            version: 300,
            is_webgl: false
        },
    );
    assert!(
        naga_version(
            GlProfile::Embedded { major: 3, minor: 0 },
            stage,
            false,
            false,
            true
        )
        .is_err()
    );
    assert!(naga_version(GlProfile::WebGl2, stage, false, false, true).is_err());
}

fn naga_stage(stage: api::ShaderStage) -> Result<naga::ShaderStage> {
    Ok(match stage {
        api::ShaderStage::Vertex => naga::ShaderStage::Vertex,
        api::ShaderStage::Fragment => naga::ShaderStage::Fragment,
        api::ShaderStage::Compute => naga::ShaderStage::Compute,
        _ => return Err(Error::Unsupported("GL WGSL shader stage")),
    })
}

pub(super) fn gl_stage(stage: api::ShaderStage) -> Result<crate::backend::gl::api::GlShaderStage> {
    Ok(match stage {
        api::ShaderStage::Vertex => crate::backend::gl::api::GlShaderStage::Vertex,
        api::ShaderStage::Fragment => crate::backend::gl::api::GlShaderStage::Fragment,
        api::ShaderStage::Compute => crate::backend::gl::api::GlShaderStage::Compute,
        _ => return Err(Error::Unsupported("GL WGSL shader stage")),
    })
}

/// Checks capability evidence rather than inferring compute from a version.
/// The native probe executed a dispatch; storage images are gated separately.
/// Browser WebGL2 has no compute route and remains false.
pub(super) fn compute_supported(device: &GlDevice) -> bool {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;
        return device
            .0
            .with_native("query GL compute capability", |provider| {
                let capabilities = provider.discovery().capabilities();
                Ok(capabilities.supports_compute())
            })
            .unwrap_or(false);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = device;
        false
    }
}

fn gl_dialect(version: naga::back::glsl::Version) -> crate::backend::gl::api::GlShaderDialect {
    match version {
        naga::back::glsl::Version::Desktop(version) => {
            crate::backend::gl::api::GlShaderDialect::Desktop { version }
        }
        naga::back::glsl::Version::Embedded { version, .. } => {
            crate::backend::gl::api::GlShaderDialect::Embedded { version }
        }
    }
}

/// OVR supports arbitrary view-dependent vertex and fragment outputs only with
/// multiview2. Naga emits EXT_multiview outside WebGL, so normalize all
/// profiles to the extension actually used by our framebuffer attachment.
pub(super) fn lower_multiview(
    mut text: String,
    stage: api::ShaderStage,
    count: Option<std::num::NonZeroU32>,
) -> Result<String> {
    let Some(count) = count else {
        if text.contains("gl_ViewIndex") || text.contains("gl_ViewID_OVR") {
            return Err(Error::Shader(
                "view_index requires a multiview pipeline".into(),
            ));
        }
        return Ok(text);
    };
    const BASE: &str = "fluxel_internal_view_index_base";
    if text.contains(BASE) {
        return Err(Error::Shader("reserved multiview uniform name".into()));
    }
    text = text.replace(
        "#extension GL_EXT_multiview : require",
        "#extension GL_OVR_multiview2 : require",
    );
    if !text.contains("#extension GL_OVR_multiview2 : require") {
        let offset = text
            .find('\n')
            .map(|p| p + 1)
            .ok_or_else(|| Error::Shader("GL shader missing version line".into()))?;
        text.insert_str(offset, "#extension GL_OVR_multiview2 : require\n");
    }
    let uses_index = text.contains("gl_ViewIndex") || text.contains("gl_ViewID_OVR");
    text = text.replace("gl_ViewID_OVR", &format!("(gl_ViewID_OVR + uint({BASE}))"));
    text = text.replace(
        "gl_ViewIndex",
        &format!("int(gl_ViewID_OVR + uint({BASE}))"),
    );
    let insertion: usize = text
        .split_inclusive('\n')
        .take_while(|line| line.trim_start().starts_with('#'))
        .map(str::len)
        .sum();
    let mut declarations = String::new();
    if uses_index {
        declarations.push_str(&format!("uniform int {BASE};\n"));
    }
    if stage == api::ShaderStage::Vertex {
        // GLSL's num_views is a literal, not a runtime uniform. Keep an
        // existing declaration only when it matches the pipeline count.
        let mut declarations_found = Vec::new();
        let mut remaining = text.as_str();
        while let Some(start) = remaining.find("layout") {
            remaining = &remaining[start + "layout".len()..];
            let trimmed = remaining.trim_start();
            let Some(body) = trimmed.strip_prefix('(') else {
                continue;
            };
            let Some(end) = body.find(')') else {
                break;
            };
            let qualifiers = &body[..end];
            if let Some(index) = qualifiers.find("num_views") {
                let tail = qualifiers[index + "num_views".len()..].trim_start();
                let value = tail.strip_prefix('=').map(str::trim_start).unwrap_or("");
                declarations_found.push(
                    value
                        .chars()
                        .take_while(char::is_ascii_digit)
                        .collect::<String>(),
                );
            }
            remaining = &body[end + 1..];
        }
        if declarations_found.len() > 1
            || declarations_found
                .first()
                .is_some_and(|value| value.parse::<u32>() != Ok(count.get()))
        {
            return Err(Error::Shader(
                "GLSL num_views does not match pipeline multiview count".into(),
            ));
        }
        if declarations_found.is_empty() {
            declarations.push_str(&format!("layout(num_views = {}) in;\n", count.get()));
        }
    }
    text.insert_str(insertion, &declarations);
    Ok(text)
}

/// Naga terminates generated entry points with `return;`. Insert before that
/// statement so the public WebGPU depth-to-GL clip conversion is reachable.
fn apply_naga_clip_depth_abi(source: &mut crate::backend::gl::api::GlShaderSource) -> Result<()> {
    if source.stage != crate::backend::gl::api::GlShaderStage::Vertex {
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

fn constant_value(value: api::SpecializationValue) -> Result<f64> {
    use api::SpecializationValue as V;
    Ok(match value {
        V::Bool(v) => {
            if v {
                1.0
            } else {
                0.0
            }
        }
        V::U8(v) => v as f64,
        V::I8(v) => v as f64,
        V::U16(v) => v as f64,
        V::I16(v) => v as f64,
        V::U32(v) => v as f64,
        V::I32(v) => v as f64,
        V::U64(v) => {
            let f = v as f64;
            if f as u128 != v as u128 {
                return Err(Error::Shader(
                    "override integer cannot be represented".into(),
                ));
            }
            f
        }
        V::I64(v) => {
            let f = v as f64;
            if f as i128 != v as i128 {
                return Err(Error::Shader(
                    "override integer cannot be represented".into(),
                ));
            }
            f
        }
        V::F32(v) => v as f64,
        V::F64(v) => v,
        V::F16Bits(v) => {
            let sign = if v & 0x8000 == 0 { 1.0 } else { -1.0 };
            let exponent = (v >> 10) & 31;
            let fraction = (v & 1023) as f64;
            sign * match exponent {
                0 => fraction * 2f64.powi(-24),
                31 if fraction == 0.0 => f64::INFINITY,
                31 => f64::NAN,
                _ => (1.0 + fraction / 1024.0) * 2f64.powi(exponent as i32 - 15),
            }
        }
    })
}

/// Collect the top-level members actually referenced by the selected entry's
/// reachable functions. Some GL drivers report every member of an active
/// struct uniform, including WGSL members never read by the shader.
fn active_push_ranges(module: &naga::Module, entry_index: usize) -> Vec<std::ops::Range<u32>> {
    use naga::{Expression as E, Statement as S, TypeInner as T};
    fn root(
        function: &naga::Function,
        mut expr: naga::Handle<E>,
        module: &naga::Module,
    ) -> Option<(naga::Handle<naga::GlobalVariable>, Option<u32>)> {
        let mut member = None;
        loop {
            match function.expressions[expr] {
                E::GlobalVariable(handle)
                    if module.global_variables[handle].space
                        == naga::AddressSpace::PushConstant =>
                {
                    return Some((handle, member));
                }
                E::AccessIndex { base, index } => {
                    member = Some(index);
                    expr = base;
                }
                E::Access { base, .. } => {
                    member = None;
                    expr = base;
                }
                _ => return None,
            }
        }
    }
    fn add(
        module: &naga::Module,
        root: (naga::Handle<naga::GlobalVariable>, Option<u32>),
        out: &mut Vec<std::ops::Range<u32>>,
    ) {
        let ty = &module.types[module.global_variables[root.0].ty].inner;
        if let T::Struct { members, span } = ty {
            match root.1.and_then(|index| {
                members
                    .get(index as usize)
                    .map(|member| (index as usize, member.offset))
            }) {
                Some((index, start)) => {
                    out.push(start..members.get(index + 1).map_or(*span, |member| member.offset))
                }
                None => out.push(0..*span),
            }
        } else {
            // This range is only used to find overlapping reflected leaves;
            // the device limit is checked separately when the layout is made.
            out.push(0..u32::MAX);
        }
    }
    fn block(
        module: &naga::Module,
        function: &naga::Function,
        body: &naga::Block,
        visited: &mut std::collections::HashSet<naga::Handle<naga::Function>>,
        out: &mut Vec<std::ops::Range<u32>>,
    ) {
        for statement in body {
            match statement {
                S::Call {
                    function: called,
                    arguments,
                    ..
                } => {
                    for arg in arguments {
                        if let Some(value) = root(function, *arg, module) {
                            add(module, value, out);
                        }
                    }
                    if visited.insert(*called) {
                        walk(module, &module.functions[*called], visited, out);
                    }
                }
                S::Block(inner) => block(module, function, inner, visited, out),
                S::If { accept, reject, .. } => {
                    block(module, function, accept, visited, out);
                    block(module, function, reject, visited, out);
                }
                S::Switch { cases, .. } => {
                    for case in cases {
                        block(module, function, &case.body, visited, out);
                    }
                }
                S::Loop {
                    body, continuing, ..
                } => {
                    block(module, function, body, visited, out);
                    block(module, function, continuing, visited, out);
                }
                _ => {}
            }
        }
    }
    fn walk(
        module: &naga::Module,
        function: &naga::Function,
        visited: &mut std::collections::HashSet<naga::Handle<naga::Function>>,
        out: &mut Vec<std::ops::Range<u32>>,
    ) {
        for (handle, expression) in function.expressions.iter() {
            match expression {
                E::AccessIndex { .. } | E::Access { .. } => {
                    if let Some(value) = root(function, handle, module) {
                        add(module, value, out);
                    }
                }
                E::Load { pointer } => {
                    if let Some(value) = root(function, *pointer, module) {
                        add(module, value, out);
                    }
                }
                _ => {}
            }
        }
        block(module, function, &function.body, visited, out);
    }
    let mut out = Vec::new();
    walk(
        module,
        &module.entry_points[entry_index].function,
        &mut std::collections::HashSet::new(),
        &mut out,
    );
    out
}
