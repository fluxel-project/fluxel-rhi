//! Shader creation and WGSL-to-GLSL lowering for the immediate GL backend.
//!
//! Source lowering is a cold, context-profile-specific operation. It produces
//! GLSL only; it neither records commands nor owns a program cache. Naga uses
//! its ordinary separate-texture model, while the reflection data retained on
//! [`GlShader`] tells program linking which emitted combined-sampler name
//! represents each Core group/binding pair.

use std::future::{Ready, ready};

use fluxel_rhi_core::api::{
    self, Error, Result, ShaderBackend, ShaderDesc, ShaderRoute, ShaderTarget,
};

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
use super::spi::native_error;
use super::types::{GlNagaBindingName, GlShader, GlShaderInner};
use super::{Gl, GlDevice};

impl ShaderBackend for Gl {
    type DirectFuture<'a> = Ready<Result<GlShader>>;
    type NagaFuture<'a> = Ready<Result<GlShader>>;

    fn shader_route(device: &GlDevice, desc: &ShaderDesc) -> ShaderRoute {
        let profile = shader_profile(device);
        let compute = desc
            .entries
            .iter()
            .any(|entry| entry.stage == api::ShaderStage::Compute);
        // The static SPI type is present for every GL build, but this exact
        // adopted context may not have a proved compute/image route.  Refuse
        // the shader before choosing a compiler target so GL 4.0--4.2 and
        // WebGL2 never appear to accept an executable compute pipeline.
        if compute && !compute_storage_supported(device) {
            return ShaderRoute::Unsupported;
        }
        if matches!(desc.format, api::ShaderFormat::Wgsl) {
            return profile.map_or(ShaderRoute::Unsupported, |profile| {
                ShaderRoute::Naga(ShaderTarget {
                    format: api::ShaderFormat::Wgsl,
                    environment: naga_environment(profile, compute).into(),
                    abi_key: "fluxel-gl-position-z-v1".into(),
                })
            });
        }

        let Some(profile) = profile else {
            return ShaderRoute::Unsupported;
        };
        let [entry] = desc.entries.as_slice() else {
            return ShaderRoute::Unsupported;
        };
        if crate::backend::gl::translate::lower_shader_source(profile, desc, entry).is_err() {
            return ShaderRoute::Unsupported;
        }
        let format = match desc.format {
            api::ShaderFormat::Glsl(version) => api::ShaderFormat::Glsl(version),
            api::ShaderFormat::GlslEs(version) => api::ShaderFormat::GlslEs(version),
            _ => return ShaderRoute::Unsupported,
        };
        ShaderRoute::Direct(ShaderTarget {
            format,
            environment: naga_environment(profile, false).into(),
            abi_key: "fluxel-gl-position-z-v1".into(),
        })
    }

    fn compile_direct<'a>(
        device: &'a GlDevice,
        desc: &'a ShaderDesc,
        _: &'a ShaderTarget,
    ) -> Self::DirectFuture<'a> {
        ready(compile_direct(device, desc))
    }

    fn compile_naga<'a>(
        device: &'a GlDevice,
        desc: &'a ShaderDesc,
        _: &'a ShaderTarget,
    ) -> Self::NagaFuture<'a> {
        ready(compile_wgsl(device, desc))
    }
}

/// Resolves the profile once without placing a platform context object in Core.
fn shader_profile(device: &GlDevice) -> Option<crate::backend::gl::profile::GlProfile> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;

        return device
            .0
            .with_native(
                "select GL shader profile",
                |provider| Ok(provider.profile()),
            )
            .ok();
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        return Some(device.0.owner.profile());
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = device;
        None
    }
}

fn naga_environment(
    profile: crate::backend::gl::profile::GlProfile,
    compute: bool,
) -> &'static str {
    match (profile, compute) {
        (crate::backend::gl::profile::GlProfile::Desktop { .. }, true) => "glsl-430",
        (crate::backend::gl::profile::GlProfile::Desktop { .. }, false) => "glsl-400",
        (crate::backend::gl::profile::GlProfile::Embedded { .. }, true) => "gles-310",
        (crate::backend::gl::profile::GlProfile::Embedded { .. }, false) => "gles-300",
        (crate::backend::gl::profile::GlProfile::WebGl2, false) => "webgl2-essl-300",
        (crate::backend::gl::profile::GlProfile::WebGl2, true) => "webgl2-essl-300",
    }
}

/// Parses shared WGSL and emits one profile-appropriate GLSL entry point.
///
/// `WriterFlags` stay empty. The public framebuffer convention already agrees
/// with the GL default framebuffer's vertical orientation; Naga's combined
/// Y/Z adjustment would therefore introduce the historical upside-down image.
/// This backend injects only the required WebGPU-depth-to-GL-depth conversion.
fn compile_wgsl(device: &GlDevice, desc: &ShaderDesc) -> Result<GlShader> {
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

fn lower_wgsl(
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
    let text = super::shader_texture_origin::lower(&text, &sampled_images, &storage_images);
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
fn compute_storage_supported(device: &GlDevice) -> bool {
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

fn compile_direct(device: &GlDevice, desc: &ShaderDesc) -> Result<GlShader> {
    let [entry] = desc.entries.as_slice() else {
        return Err(Error::Unsupported("one direct GL shader entry point"));
    };
    let profile = shader_profile(device).ok_or(Error::Unsupported("GL platform context"))?;
    let (_, mut source) = crate::backend::gl::translate::lower_shader_source(profile, desc, entry)?;
    apply_embedded_precision_defaults(&mut source);
    apply_public_clip_depth_abi(&mut source)?;
    source.source_hash = crate::backend::gl::api::GlShaderKey(source_key(
        source.text.as_bytes(),
        &source.entry_point,
    ));
    publish_shader(device, source, Vec::new(), "compile Core GL shader")
}

fn publish_shader(
    device: &GlDevice,
    source: crate::backend::gl::api::GlShaderSource,
    naga_bindings: Vec<GlNagaBindingName>,
    operation: &'static str,
) -> Result<GlShader> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlShaderApi as _;

        let id = device
            .0
            .with_native(operation, |provider| provider.create_shader(&source))
            .map_err(native_error)?;
        return Ok(GlShader::from_inner(GlShaderInner {
            state: device.0.clone(),
            id,
            key: source.source_hash,
            source,
            naga_bindings,
        }));
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        // Browser program creation compiles both stage sources together so it
        // can return link diagnostics. Retaining the lowered source here is
        // therefore the native shader object's truthful equivalent.
        let _ = operation;
        return Ok(GlShader::from_inner(GlShaderInner {
            state: device.0.clone(),
            key: source.source_hash,
            source,
            naga_bindings,
        }));
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = (device, source, naga_bindings, operation);
        Err(Error::Unsupported("GL platform context"))
    }
}

/// Inserts the one GL-family clip-depth adaptation owned by this backend.
fn apply_public_clip_depth_abi(source: &mut crate::backend::gl::api::GlShaderSource) -> Result<()> {
    if source.stage != crate::backend::gl::api::GlShaderStage::Vertex {
        return Ok(());
    }
    if !source.text.contains("void main") || !source.text.contains("gl_Position") {
        return Err(Error::Unsupported("GL vertex shader main/position ABI"));
    }
    let end = source
        .text
        .rfind('}')
        .ok_or(Error::Unsupported("GL vertex shader main/position ABI"))?;
    source.text.insert_str(
        end,
        "\n    gl_Position.z = 2.0 * gl_Position.z - gl_Position.w;\n",
    );
    Ok(())
}

/// Gives GLSL ES defined defaults for unqualified scalar types.
///
/// Desktop GLSL supplies a default float precision; GLSL ES fragment shaders
/// do not. Naga leaves source-level precision choices open, so native GLES and
/// browser WebGL apply the same backend policy before compiling the source.
/// Explicit author precision statements remain authoritative.
fn apply_embedded_precision_defaults(source: &mut crate::backend::gl::api::GlShaderSource) {
    if !matches!(
        source.dialect,
        crate::backend::gl::api::GlShaderDialect::Embedded { .. }
    ) {
        return;
    }
    let mut declarations = Vec::new();
    let mut body = String::with_capacity(source.text.len());
    for line in source.text.split_inclusive('\n') {
        if line.trim_start().starts_with("precision ") {
            declarations.push(line.to_owned());
        } else {
            body.push_str(line);
        }
    }
    let has_default = |ty: &str| {
        declarations.iter().any(|line| {
            line.trim_end()
                .strip_suffix(';')
                .is_some_and(|declaration| declaration.split_whitespace().last() == Some(ty))
        })
    };
    let has_float_default = has_default("float");
    let has_int_default = has_default("int");
    if !has_float_default {
        declarations.push("precision highp float;\n".into());
    }
    if !has_int_default {
        declarations.push("precision highp int;\n".into());
    }

    // Naga and the texture-origin lowering can place helper functions before
    // Naga's default precision declarations. GLSL ES requires defaults before
    // every use, so move all precision declarations after directives/comments.
    let mut insertion = 0;
    for line in body.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
            insertion += line.len();
        } else {
            break;
        }
    }
    let declarations = declarations.concat();
    body.insert_str(insertion, &declarations);
    source.text = body;
}

fn source_key(bytes: &[u8], entry: &str) -> [u8; 32] {
    let mut state = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes.iter().chain(entry.as_bytes()) {
        state = (state ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
    }
    let mut key = [0; 32];
    for index in 0..4 {
        key[index * 8..(index + 1) * 8]
            .copy_from_slice(&state.wrapping_add(index as u64).to_le_bytes());
    }
    key
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};

    use super::{apply_embedded_precision_defaults, lower_wgsl, source_key};
    use crate::backend::gl::api::{GlShaderDialect, GlShaderKey, GlShaderSource, GlShaderStage};

    const TEXTURED_WGSL: &str = r#"
        struct Scene { value: f32, };
        @group(0) @binding(0) var<uniform> scene: Scene;
        @group(0) @binding(1) var color_texture: texture_2d<f32>;
        @group(0) @binding(2) var color_sampler: sampler;
        @vertex fn vs_main(@location(0) position: vec3f) -> @builtin(position) vec4f {
            return vec4f(position, 1.0);
        }
        @fragment fn fs_main() -> @location(0) vec4f {
            return textureSample(color_texture, color_sampler, vec2f(scene.value));
        }
    "#;

    #[test]
    fn embedded_precision_defaults_follow_version_and_extensions() {
        let mut source = GlShaderSource {
            stage: GlShaderStage::Fragment,
            dialect: GlShaderDialect::Embedded { version: 300 },
            entry_point: "main".into(),
            source_hash: GlShaderKey([0; 32]),
            text: "#version 300 es\n#extension GL_EXT_texture_norm16 : enable\nout vec4 color;\n"
                .into(),
            debug_name: None,
        };
        apply_embedded_precision_defaults(&mut source);
        assert!(source.text.contains(
            "#extension GL_EXT_texture_norm16 : enable\nprecision highp float;\nprecision highp int;\nout vec4 color;"
        ));

        let mut desktop = GlShaderSource {
            stage: GlShaderStage::Vertex,
            dialect: GlShaderDialect::Desktop { version: 400 },
            entry_point: "main".into(),
            source_hash: GlShaderKey([0; 32]),
            text: "#version 400 core\nvoid main() {}\n".into(),
            debug_name: None,
        };
        let original = desktop.text.clone();
        apply_embedded_precision_defaults(&mut desktop);
        assert_eq!(desktop.text, original);
    }
    const COMPUTE_WGSL: &str = r#"
        @group(0) @binding(0) var source_image: texture_2d<f32>;
        @group(0) @binding(1) var result_image: texture_storage_2d<rgba8unorm, write>;
        @compute @workgroup_size(1)
        fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
            let color = textureLoad(source_image, vec2<i32>(id.xy), 0);
            textureStore(result_image, vec2<i32>(id.xy), color);
        }
    "#;
    const WEBGL2_RASTER_FALLBACK_WGSL: &str =
        include_str!("../../../../../rhi/examples/assets/06_compute/filters_raster.wgsl");

    fn desc(entry: &str, stage: ShaderStage) -> ShaderDesc {
        ShaderDesc {
            label: Some("shader-lowering-test".into()),
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(TEXTURED_WGSL.as_bytes()),
            entries: vec![ShaderEntryPoint {
                name: entry.into(),
                stage,
                workgroup_size: None,
            }],
            required_features: Vec::new(),
        }
    }

    fn compute_desc() -> ShaderDesc {
        ShaderDesc {
            label: Some("compute-origin-test".into()),
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(COMPUTE_WGSL.as_bytes()),
            entries: vec![ShaderEntryPoint {
                name: "cs".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: Vec::new(),
        }
    }

    fn raster_fallback_desc(entry: &str) -> ShaderDesc {
        ShaderDesc {
            label: Some("06-webgl2-raster-fallback-test".into()),
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(WEBGL2_RASTER_FALLBACK_WGSL.as_bytes()),
            entries: vec![ShaderEntryPoint {
                name: entry.into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: Vec::new(),
        }
    }

    #[test]
    fn naga_lowers_wgsl_for_each_supported_glsl_profile() {
        use crate::backend::gl::profile::GlProfile;

        for profile in [
            GlProfile::Desktop { major: 4, minor: 0 },
            GlProfile::Embedded { major: 3, minor: 0 },
            GlProfile::WebGl2,
        ] {
            let (vertex, _) = lower_wgsl(profile, &desc("vs_main", ShaderStage::Vertex)).unwrap();
            assert!(
                vertex
                    .text
                    .contains("gl_Position.z = 2.0 * gl_Position.z - gl_Position.w")
            );
            assert_eq!(
                vertex.source_hash.0,
                source_key(vertex.text.as_bytes(), "vs_main")
            );

            let (_, bindings) =
                lower_wgsl(profile, &desc("fs_main", ShaderStage::Fragment)).unwrap();
            assert!(
                bindings
                    .iter()
                    .any(|binding| binding.group == 0 && binding.binding == 1)
            );
            assert!(
                bindings
                    .iter()
                    .any(|binding| binding.group == 0 && binding.binding == 2)
            );
            let (fragment, _) =
                lower_wgsl(profile, &desc("fs_main", ShaderStage::Fragment)).unwrap();
            assert!(fragment.text.contains("fluxel_internal_flip_uv("));
        }
    }

    #[test]
    fn naga_compute_lowering_flips_texture_load_and_storage_store_rows() {
        use crate::backend::gl::profile::GlProfile;

        let (compute, bindings) =
            lower_wgsl(GlProfile::Desktop { major: 4, minor: 6 }, &compute_desc()).unwrap();

        assert!(compute.text.contains("fluxel_internal_flip_texel("));
        assert!(compute.text.contains("textureSize("));
        assert!(compute.text.contains("imageSize("), "{}", compute.text);
        assert!(
            compute.text.contains("layout(binding = 0,rgba8)"),
            "{}",
            compute.text
        );
        assert!(bindings.iter().any(|binding| binding.binding == 0));
        assert!(
            bindings
                .iter()
                .any(|binding| binding.binding == 1 && binding.image_unit == Some(0))
        );
    }

    #[test]
    fn webgl2_raster_filter_fallback_lowers_texture_loads_to_es_300() {
        use crate::backend::gl::profile::GlProfile;

        for entry in ["emboss_fs", "edgedetect_fs", "sharpen_fs"] {
            let (fragment, _) =
                lower_wgsl(GlProfile::WebGl2, &raster_fallback_desc(entry)).unwrap();
            assert!(fragment.text.contains("#version 300 es"));
            assert!(fragment.text.contains("textureSize("), "{}", fragment.text);
            assert!(fragment.text.contains("texelFetch("), "{}", fragment.text);
        }
    }
}
