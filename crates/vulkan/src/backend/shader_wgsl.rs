//! WGSL source validation and per-pipeline SPIR-V specialization.
use super::*;
use std::collections::HashSet;

fn stage(stage: api::ShaderStage) -> Result<naga::ShaderStage> {
    match stage {
        api::ShaderStage::Vertex => Ok(naga::ShaderStage::Vertex),
        api::ShaderStage::Fragment => Ok(naga::ShaderStage::Fragment),
        api::ShaderStage::Compute => Ok(naga::ShaderStage::Compute),
        _ => Err(Error::Unsupported("WGSL shader stage")),
    }
}
fn parsed(
    device: &VulkanDevice,
    desc: &api::ShaderDesc,
) -> Result<(naga::Module, naga::valid::ModuleInfo)> {
    let source =
        std::str::from_utf8(&desc.bytes).map_err(|_| Error::Shader("WGSL is not UTF-8".into()))?;
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| Error::Shader(e.emit_to_string(source)))?;
    use naga::valid::Capabilities as C;
    let mut caps = C::PUSH_CONSTANT | C::SHADER_FLOAT16_IN_FLOAT32;
    if desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_EARLY_DEPTH_TEST)
    {
        caps |= C::EARLY_DEPTH_TEST;
    }
    let features = &device.0.features;
    if device.0.nonuniform_sampled_image {
        caps |= C::SAMPLER_NON_UNIFORM_INDEXING;
    }
    // Naga groups these two independent Vulkan features under one capability.
    // Validate the precise SPIR-V capabilities after code generation below.
    if device.0.nonuniform_sampled_image || device.0.nonuniform_storage_buffer {
        caps |= C::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    }
    if device.0.nonuniform_storage_image {
        caps |= C::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING;
    }
    if device.0.nonuniform_uniform_buffer {
        caps |= C::UNIFORM_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    }
    if device.0.multiview {
        caps |= C::MULTIVIEW;
    }
    if features.image_cube_array == vk::TRUE {
        caps |= C::CUBE_ARRAY_TEXTURES;
    }
    if features.shader_float64 == vk::TRUE {
        caps |= C::FLOAT64;
    }
    if features.shader_int64 == vk::TRUE {
        caps |= C::SHADER_INT64;
    }
    if features.shader_cull_distance == vk::TRUE {
        caps |= C::CULL_DISTANCE;
    }
    if features.sample_rate_shading == vk::TRUE {
        caps |= C::MULTISAMPLED_SHADING;
    }
    if features.dual_src_blend == vk::TRUE {
        caps |= C::DUAL_SOURCE_BLENDING;
    }
    if features.shader_storage_image_extended_formats == vk::TRUE {
        caps |= C::STORAGE_TEXTURE_16BIT_NORM_FORMATS;
    }
    if features.geometry_shader == vk::TRUE {
        caps |= C::PRIMITIVE_INDEX;
    }
    if desc
        .required_features
        .iter()
        .any(|f| f == api::SHADER_FEATURE_RAY_QUERY)
    {
        caps |= naga::valid::Capabilities::RAY_QUERY;
    }
    if desc
        .required_features
        .iter()
        .any(|f| f == "shaderClipDistance")
    {
        caps |= naga::valid::Capabilities::CLIP_DISTANCE;
    }
    if super::subgroup::requested(desc) {
        caps |= naga::valid::Capabilities::SUBGROUP;
        if super::subgroup::supports_stage(device, api::ShaderStage::Vertex) {
            caps |= naga::valid::Capabilities::SUBGROUP_VERTEX_STAGE;
        }
        if super::subgroup::supports_stage(device, api::ShaderStage::Compute) {
            caps |= naga::valid::Capabilities::SUBGROUP_BARRIER;
        }
    }
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
        .validate(&module)
        .map_err(|e| Error::Shader(e.emit_to_string(source)))?;
    for entry in &desc.entries {
        let kind = stage(entry.stage)?;
        if !module
            .entry_points
            .iter()
            .any(|v| v.name == entry.name && v.stage == kind)
        {
            return Err(Error::Shader(format!(
                "missing shader entry {}",
                entry.name
            )));
        }
    }
    Ok((module, info))
}
pub(crate) fn create_wgsl(device: &VulkanDevice, desc: &api::ShaderDesc) -> Result<VulkanShader> {
    for feature in &desc.required_features {
        match feature.as_str() {
            api::SHADER_FEATURE_EARLY_DEPTH_TEST => {}
            "shaderClipDistance" if device.0.shader_clip_distance => {}
            api::SHADER_FEATURE_RAY_QUERY if device.0.ray_support.query => {}
            "subgroups" => super::subgroup::validate_declared(device, desc)?,
            _ => return Err(Error::Unsupported("Vulkan WGSL required feature")),
        }
    }
    parsed(device, desc)?;
    Ok(VulkanShader(Arc::new(ShaderState {
        device: device.0.clone(),
        module: vk::ShaderModule::null(),
        entries: desc.entries.clone(),
        words: Vec::new(),
        wgsl: Some(desc.clone()),
    })))
}
/// Returns an owned native module and whether Naga consumed specialization.
pub(crate) fn compile_variant(
    device: &VulkanDevice,
    shader: &VulkanShader,
    layout: &super::super::pipeline::VulkanPipelineLayout,
    entry: &str,
    shader_stage: api::ShaderStage,
    constants: &[(String, api::SpecializationValue)],
    push_base: u32,
) -> Result<(VulkanShader, bool)> {
    let Some(desc) = &shader.0.wgsl else {
        let shader = super::push::relocate(device, shader.clone(), push_base)?;
        return Ok((super::sampler::remap(device, shader, layout)?, false));
    };
    let (module, info) = parsed(device, desc)?;
    let mut values = naga::back::PipelineConstants::default();
    let mut seen = HashSet::new();
    for (name, value) in constants {
        let item = module
            .overrides
            .iter()
            .find(|(_, item)| {
                item.id.map(|id| id.to_string()) == Some(name.clone())
                    || (item.id.is_none() && item.name.as_deref() == Some(name.as_str()))
            })
            .map(|(_, item)| item)
            .ok_or_else(|| Error::Shader(format!("unknown override {name}")))?;
        let (kind, v) = match value {
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
        if !matches!(module.types[item.ty].inner, naga::TypeInner::Scalar(s) if s.kind == kind && (kind == naga::ScalarKind::Bool || s.width == 4))
            || !seen.insert(name)
        {
            return Err(Error::Shader(format!(
                "invalid or duplicate override {name}"
            )));
        }
        values.insert(name.clone(), v);
    }
    let kind = stage(shader_stage)?;
    let (module, info) = naga::back::pipeline_constants::process_overrides(
        &module,
        &info,
        Some((kind, entry)),
        &values,
    )
    .map_err(|e| Error::Shader(e.to_string()))?;
    let mut options = naga::back::spv::Options::default();
    // Vulkan's subgroup capabilities require SPIR-V 1.3. The backend's
    // advertised target is Vulkan 1.3, so emit that version instead of
    // Naga's conservative 1.0 default when subgroup builtins are present.
    if super::subgroup::requested(desc) {
        options.lang_version = (1, 3);
    }
    // Raster recording already uses a negative-height viewport to match the
    // WebGPU coordinate convention; a shader Y flip would invert it twice.
    options
        .flags
        .remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);
    let words = naga::back::spv::write_vec(
        &module,
        &info,
        &options,
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: kind,
            entry_point: entry.to_owned(),
        }),
    )
    .map_err(|e| Error::Shader(e.to_string()))?;
    let mut cursor = 5;
    while cursor < words.len() {
        let len = (words[cursor] >> 16) as usize;
        if len == 0 || cursor + len > words.len() {
            return Err(Error::Shader("invalid generated SPIR-V".into()));
        }
        if words[cursor] & 0xffff == 17 && len >= 2 {
            let supported = match words[cursor + 1] {
                5307 => device.0.nonuniform_sampled_image,
                5308 => device.0.nonuniform_storage_buffer,
                _ => true,
            };
            if !supported {
                return Err(Error::Unsupported(
                    "Vulkan non-uniform descriptor indexing feature",
                ));
            }
        }
        cursor += len;
    }
    let mut variant = desc.clone();
    variant.format = api::ShaderFormat::SpirV;
    variant.bytes = Arc::from(
        words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    let shader = super::push::relocate(device, super::create(device, &variant)?, push_base)?;
    Ok((super::sampler::remap(device, shader, layout)?, true))
}
