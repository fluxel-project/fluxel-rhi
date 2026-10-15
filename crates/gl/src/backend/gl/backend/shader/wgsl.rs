//! WGSL parsing, profile-specific GLSL generation, and reflected binding metadata.
//!
//! Naga lowering preserves WebGPU coordinates while adding only the GL-family
//! clip-depth and texture-origin adaptations required by the public contract.

use super::*;

/// Parses shared WGSL and emits one profile-appropriate GLSL entry point.
///
/// `WriterFlags` stay empty. The public framebuffer convention already agrees
/// with the GL default framebuffer's vertical orientation; Naga's combined
/// Y/Z adjustment would therefore introduce the historical upside-down image.
/// This backend injects only the required WebGPU-depth-to-GL-depth conversion.
pub(super) fn compile_wgsl(device: &GlDevice, desc: &ShaderDesc) -> Result<GlShader> {
    let profile = shader_profile(device).ok_or(Error::Unsupported("GL platform context"))?;
    if desc
        .entries
        .iter()
        .any(|entry| entry.stage == api::ShaderStage::Compute)
        && !compute_storage_supported(device)
    {
        return Err(Error::Unsupported("GL compute/storage-image capability"));
    }
    let (source, bindings) = lower_wgsl(profile, desc)?;
    publish_shader(device, source, bindings, "compile WGSL-derived GL shader")
}

pub(super) fn lower_wgsl(
    profile: crate::backend::gl::profile::GlProfile,
    desc: &ShaderDesc,
) -> Result<(
    crate::backend::gl::api::GlShaderSource,
    Vec<GlNagaBindingName>,
)> {
    let [entry] = desc.entries.as_slice() else {
        return Err(Error::Unsupported("one WGSL GL entry point"));
    };
    let source =
        std::str::from_utf8(&desc.bytes).map_err(|_| Error::Shader("WGSL is not UTF-8".into()))?;
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| Error::Shader(error.emit_to_string(source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| Error::Shader(error.emit_to_string(source)))?;
    let version = naga_version(profile, entry.stage)?;
    let expected_stage = naga_stage(entry.stage)?;
    let entry_index = module
        .entry_points
        .iter()
        .position(|candidate| candidate.name == entry.name && candidate.stage == expected_stage)
        .ok_or(Error::Unsupported("WGSL GL entry point"))?;
    let image_units = naga_storage_image_units(&module, info.get_entry_point(entry_index))?;
    let options = naga::back::glsl::Options {
        version,
        writer_flags: naga::back::glsl::WriterFlags::empty(),
        binding_map: image_units.clone(),
        zero_initialize_workgroup_memory: true,
    };
    let pipeline = naga::back::glsl::PipelineOptions {
        shader_stage: naga_stage(entry.stage)?,
        entry_point: entry.name.clone(),
        multiview: None,
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
    let sampled_images = reflection
        .texture_mapping
        .iter()
        .filter_map(|(name, mapping)| {
            (is_y_flippable_image(&module, mapping.texture)
                && !is_storage_image(&module, mapping.texture))
            .then_some(name.as_str())
        })
        .collect::<std::collections::HashSet<_>>();
    let storage_images = reflection
        .texture_mapping
        .iter()
        .filter_map(|(name, mapping)| {
            (is_y_flippable_image(&module, mapping.texture)
                && is_storage_image(&module, mapping.texture))
            .then_some(name.as_str())
        })
        .collect::<std::collections::HashSet<_>>();
    let text = super::super::shader_texture_origin::lower(&text, &sampled_images, &storage_images);
    let bindings = collect_naga_bindings(&module, reflection, &image_units);
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
    Ok((lowered, bindings))
}

fn is_y_flippable_image(module: &naga::Module, global: naga::Handle<naga::GlobalVariable>) -> bool {
    matches!(
        module.types[module.global_variables[global].ty].inner,
        naga::TypeInner::Image {
            dim: naga::ImageDimension::D2,
            ..
        }
    )
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
) -> Vec<GlNagaBindingName> {
    let mut bindings = Vec::new();
    for (global, name) in reflection.uniforms {
        push_binding(&mut bindings, module, global, name, image_units);
    }
    for (name, mapping) in reflection.texture_mapping {
        // GLSL has a combined texture/sampler uniform. Both WebGPU bindings
        // must resolve to that one reflected name for layout validation.
        push_binding(
            &mut bindings,
            module,
            mapping.texture,
            name.clone(),
            image_units,
        );
        if let Some(sampler) = mapping.sampler {
            push_binding(&mut bindings, module, sampler, name, image_units);
        }
    }
    bindings.sort_by(|left, right| {
        (left.group, left.binding, &left.name).cmp(&(right.group, right.binding, &right.name))
    });
    bindings.dedup_by(|left, right| {
        left.group == right.group && left.binding == right.binding && left.name == right.name
    });
    bindings
}

fn push_binding(
    output: &mut Vec<GlNagaBindingName>,
    module: &naga::Module,
    global: naga::Handle<naga::GlobalVariable>,
    name: String,
    image_units: &naga::back::glsl::BindingMap,
) {
    if let Some(binding) = module.global_variables[global].binding {
        output.push(GlNagaBindingName {
            group: binding.group,
            binding: binding.binding,
            name,
            image_unit: image_units.get(&binding).copied().map(u32::from),
        });
    }
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
) -> Result<naga::back::glsl::Version> {
    let compute = stage == api::ShaderStage::Compute;
    Ok(match profile {
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
        crate::backend::gl::profile::GlProfile::Embedded { .. } if !compute => {
            naga::back::glsl::Version::Embedded {
                version: 300,
                is_webgl: false,
            }
        }
        crate::backend::gl::profile::GlProfile::WebGl2 if !compute => {
            naga::back::glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            }
        }
        _ => return Err(Error::Unsupported("GL shader stage/profile")),
    })
}

fn naga_stage(stage: api::ShaderStage) -> Result<naga::ShaderStage> {
    Ok(match stage {
        api::ShaderStage::Vertex => naga::ShaderStage::Vertex,
        api::ShaderStage::Fragment => naga::ShaderStage::Fragment,
        api::ShaderStage::Compute => naga::ShaderStage::Compute,
        _ => return Err(Error::Unsupported("GL WGSL shader stage")),
    })
}

fn gl_stage(stage: api::ShaderStage) -> Result<crate::backend::gl::api::GlShaderStage> {
    Ok(match stage {
        api::ShaderStage::Vertex => crate::backend::gl::api::GlShaderStage::Vertex,
        api::ShaderStage::Fragment => crate::backend::gl::api::GlShaderStage::Fragment,
        api::ShaderStage::Compute => crate::backend::gl::api::GlShaderStage::Compute,
        _ => return Err(Error::Unsupported("GL WGSL shader stage")),
    })
}

/// Checks capability evidence rather than inferring compute from a version.
/// The native probe has executed both a dispatch and an image load/store bind;
/// browser WebGL2 has neither route and remains false.
pub(super) fn compute_storage_supported(device: &GlDevice) -> bool {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::{GlCapability, GlFamilyApi as _};
        return device
            .0
            .with_native("query GL compute capability", |provider| {
                let capabilities = provider.discovery().capabilities();
                Ok(capabilities.supports(GlCapability::Compute)
                    && capabilities.supports(GlCapability::StorageImage))
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
        "gl_Position.z = 2.0 * gl_Position.z - gl_Position.w;\n    ",
    );
    Ok(())
}
