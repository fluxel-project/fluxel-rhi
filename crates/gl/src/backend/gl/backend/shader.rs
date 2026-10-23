//! Static shader SPI routing, direct-source compilation, and common shader publication.
//!
//! Naga-specific WGSL parsing and GLSL generation live in `shader/wgsl.rs`;
//! this module chooses that route from the active profile and Core contract.
//! Compilation is a cold path and creates no program cache or command stream.

use std::future::{Ready, ready};

use fluxel_rhi_core::api::{
    self, Error, Result, ShaderBackend, ShaderDesc, ShaderRoute, ShaderTarget,
};

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
use super::spi::native_error;
use super::types::{GlNagaBindingName, GlShader, GlShaderInner};
use super::{Gl, GlDevice};

mod binding_array;
mod binding_array_dynamic;
mod binding_array_dynamic_buffer;
mod binding_array_dynamic_load;
mod binding_array_dynamic_sample;
mod binding_array_dynamic_sampled_load;
mod binding_array_dynamic_sampled_query;
mod binding_array_dynamic_storage_query;
mod binding_array_helper;
mod pairs;
mod remap;
mod specialize;
mod texture_num_levels;
mod texture_num_samples;
mod wgsl;

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
        // adopted context may not have a proved compute route.  Refuse
        // the shader before choosing a compiler target so GL 4.0--4.2 and
        // WebGL2 never appear to accept an executable compute pipeline.
        if compute
            && !matches!(desc.format, api::ShaderFormat::Wgsl)
            && !wgsl::compute_supported(device)
        {
            return ShaderRoute::Unsupported;
        }
        if matches!(desc.format, api::ShaderFormat::Wgsl) {
            return profile.map_or(ShaderRoute::Unsupported, |profile| {
                ShaderRoute::Naga(ShaderTarget {
                    format: api::ShaderFormat::Wgsl,
                    environment: naga_environment(profile, compute).into(),
                    abi_key: "fluxel-gl-position-yz-row-order-v2".into(),
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
            abi_key: "fluxel-gl-position-yz-row-order-v2".into(),
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
        ready(wgsl::compile_wgsl(device, desc))
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
    publish_shader(
        device,
        source,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        "compile Core GL shader",
    )
}

fn publish_shader(
    device: &GlDevice,
    source: crate::backend::gl::api::GlShaderSource,
    naga_bindings: Vec<GlNagaBindingName>,
    push_uniforms: Vec<super::push::PushUniform>,
    mip_uniforms: Vec<super::types::GlMipUniform>,
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
            id: Some(id),
            source: Some(source),
            wgsl: None,
            naga_bindings,
            push_uniforms,
            mip_uniforms,
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
            source: Some(source),
            wgsl: None,
            naga_bindings,
            push_uniforms,
            mip_uniforms,
        }));
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = (device, source, naga_bindings, push_uniforms, operation);
        Err(Error::Unsupported("GL platform context"))
    }
}

pub(super) fn materialize_stage(
    device: &GlDevice,
    shader: &GlShader,
    entry: &str,
    stage: api::ShaderStage,
    constants: &[(String, api::SpecializationValue)],
    multiview: Option<std::num::NonZeroU32>,
) -> Result<GlShader> {
    if !std::rc::Rc::ptr_eq(&device.0, &shader.state) {
        return Err(Error::InvalidState);
    }
    if let Some(desc) = &shader.wgsl {
        let selected = desc
            .entries
            .iter()
            .find(|item| item.name == entry && item.stage == stage)
            .ok_or_else(|| {
                Error::Shader(format!("WGSL entry point {entry} not declared for stage"))
            })?;
        if stage == api::ShaderStage::Compute && !wgsl::compute_supported(device) {
            return Err(Error::Unsupported("GL compute capability"));
        }
        let mut selected_desc = desc.clone();
        selected_desc.entries = vec![selected.clone()];
        let profile = shader_profile(device).ok_or(Error::Unsupported("GL platform context"))?;
        let parameters = if stage == api::ShaderStage::Vertex {
            draw_parameters(device)?
        } else {
            DrawParameters::None
        };
        let (image_limit, storage_buffer_limit) = shader_resource_limits(device, stage)?;
        let sampled_limit = shader_sampled_limit(device, stage)?;
        let uniform_limit = shader_uniform_limit(device, stage)?;
        let (source, bindings, push_uniforms, mip_uniforms) = wgsl::lower_wgsl_with_limits(
            profile,
            &selected_desc,
            constants,
            parameters,
            multiview,
            image_limit,
            storage_buffer_limit,
            sampled_limit,
            uniform_limit,
        )?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        if matches!(
            profile,
            crate::backend::gl::profile::GlProfile::Embedded { major: 3, minor: 1 }
        ) && bindings.iter().any(|binding| {
            matches!(
                binding.resource_type,
                Some(super::types::GlNagaResourceType::SampledTexture {
                    dimension: api::TextureViewDimension::D2Array,
                    multisampled: true,
                    ..
                })
            )
        }) {
            use crate::backend::gl::api::GlFamilyApi as _;
            let supported = device
                .0
                .with_native("query multisample array shader capability", |provider| {
                    Ok(provider
                        .discovery()
                        .extensions()
                        .raw_reported_names()
                        .any(|name| name == "GL_OES_texture_storage_multisample_2d_array"))
                })
                .map_err(native_error)?;
            if !supported {
                return Err(Error::Unsupported(
                    "GLES multisample array shader extension",
                ));
            }
        }
        return publish_shader(
            device,
            source,
            bindings,
            push_uniforms,
            mip_uniforms,
            "compile WGSL-derived GL shader",
        );
    }
    if !constants.is_empty() {
        return Err(Error::Unsupported("GL direct GLSL pipeline constants"));
    }
    let source = shader.source.as_ref().ok_or(Error::InvalidState)?;
    if source.entry_point != entry || source.stage != wgsl::gl_stage(stage)? {
        return Err(Error::InvalidState);
    }
    if multiview.is_some() {
        let mut source = source.clone();
        source.text = wgsl::lower_multiview(source.text, stage, multiview)?;
        source.source_hash = crate::backend::gl::api::GlShaderKey(source_key(
            source.text.as_bytes(),
            &source.entry_point,
        ));
        return publish_shader(
            device,
            source,
            shader.naga_bindings.clone(),
            shader.push_uniforms.clone(),
            shader.mip_uniforms.clone(),
            "compile multiview GLSL stage",
        );
    }
    Ok(shader.clone())
}

fn shader_resource_limits(device: &GlDevice, stage: api::ShaderStage) -> Result<(u32, u32)> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;
        return device
            .0
            .with_native("query GL shader image limits", |provider| {
                let limits = provider.discovery().limits();
                let per_stage = match stage {
                    api::ShaderStage::Vertex => limits.max_vertex_image_uniforms,
                    api::ShaderStage::Fragment => limits.max_fragment_image_uniforms,
                    api::ShaderStage::Compute => limits.max_compute_image_uniforms,
                    _ => 0,
                };
                let storage_stage = match stage {
                    api::ShaderStage::Vertex => limits.max_vertex_storage_blocks,
                    api::ShaderStage::Fragment => limits.max_fragment_storage_blocks,
                    api::ShaderStage::Compute => limits.max_compute_storage_blocks,
                    _ => 0,
                };
                Ok((
                    limits
                        .max_image_units
                        .min(limits.max_combined_image_units)
                        .min(per_stage),
                    limits
                        .max_storage_buffer_bindings
                        .min(limits.max_combined_storage_blocks)
                        .min(storage_stage),
                ))
            })
            .map_err(native_error);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = (device, stage);
        Ok((0, 0))
    }
}

fn shader_uniform_limit(device: &GlDevice, stage: api::ShaderStage) -> Result<u32> {
    let choose = |limits: &crate::backend::gl::api::GlLimits| {
        let per_stage = match stage {
            api::ShaderStage::Vertex => limits.max_vertex_uniform_blocks,
            api::ShaderStage::Fragment => limits.max_fragment_uniform_blocks,
            api::ShaderStage::Compute => limits.max_compute_uniform_blocks,
            _ => 0,
        };
        per_stage
            .min(limits.max_combined_uniform_blocks)
            .min(limits.max_uniform_buffer_bindings)
    };
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;
        return device
            .0
            .with_native("query GL shader uniform limits", |provider| {
                Ok(choose(&provider.discovery().limits()))
            })
            .map_err(native_error);
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        return Ok(choose(&device.0.owner.discovery_snapshot().limits()));
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = (device, choose);
        Ok(0)
    }
}

fn shader_sampled_limit(device: &GlDevice, stage: api::ShaderStage) -> Result<u32> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;
        return device
            .0
            .with_native("query GL shader texture limits", |provider| {
                let snapshot = provider.discovery();
                let limits = snapshot.limits();
                let per_stage = match stage {
                    api::ShaderStage::Vertex => limits.max_vertex_texture_image_units,
                    api::ShaderStage::Fragment => limits.max_fragment_texture_image_units,
                    api::ShaderStage::Compute => limits.max_compute_texture_image_units,
                    _ => 0,
                };
                Ok(per_stage.min(limits.max_combined_texture_image_units))
            })
            .map_err(native_error);
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        let discovery = device.0.owner.discovery_snapshot();
        let limits = discovery.limits();
        let per_stage = match stage {
            api::ShaderStage::Vertex => limits.max_vertex_texture_image_units,
            api::ShaderStage::Fragment => limits.max_fragment_texture_image_units,
            api::ShaderStage::Compute => limits.max_compute_texture_image_units,
            _ => 0,
        };
        return Ok(per_stage.min(limits.max_combined_texture_image_units));
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = (device, stage);
        Ok(0)
    }
}

/// Vertex builtin support is a context capability, not a profile assumption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DrawParameters {
    None,
    Core,
    Arb,
}

pub(super) fn draw_parameters(device: &GlDevice) -> Result<DrawParameters> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;
        return device
            .0
            .with_native("query GL shader draw parameters", |provider| {
                let profile = provider.profile();
                if let crate::backend::gl::api::GlFamilyProfile::Desktop { major, minor } = profile
                {
                    if major > 4 || major == 4 && minor >= 6 {
                        return Ok(DrawParameters::Core);
                    }
                    if provider
                        .discovery()
                        .extensions()
                        .raw_reported_names()
                        .any(|name| name == "GL_ARB_shader_draw_parameters")
                    {
                        return Ok(DrawParameters::Arb);
                    }
                }
                Ok(DrawParameters::None)
            })
            .map_err(native_error);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = device;
        Ok(DrawParameters::None)
    }
}

/// Inserts the one GL-family clip-depth adaptation owned by this backend.
fn apply_public_clip_depth_abi(source: &mut crate::backend::gl::api::GlShaderSource) -> Result<()> {
    if source.stage != crate::backend::gl::api::GlShaderStage::Vertex {
        return Ok(());
    }
    // Rename the user's entry with the GLSL preprocessor, then wrap it. A
    // wrapper also handles early returns inside main, unlike an insertion at
    // the final brace (which may belong to a different function).
    if !source.text.contains("gl_Position") {
        return Err(Error::Unsupported("GL vertex shader main/position ABI"));
    }
    let mut insertion = 0;
    let mut version_end = None;
    for line in source.text.split_inclusive('\n') {
        insertion += line.len();
        if line.trim_start().starts_with("#version") {
            version_end = Some(insertion);
            break;
        }
    }
    let insertion = if let Some(end) = version_end {
        if !source.text[..end].ends_with('\n') {
            source.text.insert(end, '\n');
            end + 1
        } else {
            end
        }
    } else {
        // ShaderFormat carries the version even when the source omits it.
        let version = match source.dialect {
            crate::backend::gl::api::GlShaderDialect::Desktop { version } => {
                format!("#version {version} core\n")
            }
            crate::backend::gl::api::GlShaderDialect::Embedded { version } => {
                format!("#version {version} es\n")
            }
        };
        source.text.insert_str(0, &version);
        version.len()
    };
    let mut entry_name = "fluxel_internal_vertex_main".to_owned();
    while source.text.contains(&entry_name) {
        entry_name.push('_');
    }
    source
        .text
        .insert_str(insertion, &format!("#define main {entry_name}\n"));
    source.text.push_str(&format!("\n#undef main\nvoid main() {{ {entry_name}(); gl_Position.y = -gl_Position.y; gl_Position.z = 2.0 * gl_Position.z - gl_Position.w; }}\n"));
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

pub(super) fn source_key(bytes: &[u8], entry: &str) -> [u8; 32] {
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

    use super::wgsl::lower_wgsl;
    use super::{apply_embedded_precision_defaults, source_key};
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
    fn indirect_instance_shader_modes_keep_direct_offset_uniform() {
        let mut shader = desc("vs", ShaderStage::Vertex);
        shader.bytes = Arc::from(br#"@vertex fn vs(@builtin(instance_index) i:u32)->@builtin(position) vec4f { return vec4f(f32(i)); }"#.as_slice());
        let profile = crate::backend::gl::profile::GlProfile::Desktop { major: 4, minor: 6 };
        for (mode, builtin, version) in [
            (super::DrawParameters::Core, "gl_BaseInstance)", 460),
            (super::DrawParameters::Arb, "gl_BaseInstanceARB)", 400),
        ] {
            let (result, _, _) =
                super::wgsl::lower_wgsl_with_parameters(profile, &shader, &[], mode).unwrap();
            assert!(result.text.contains(builtin), "{}", result.text);
            assert!(result.text.contains("+ naga_vs_first_instance"));
            assert_eq!(result.dialect, GlShaderDialect::Desktop { version });
            if mode == super::DrawParameters::Arb {
                assert!(
                    result
                        .text
                        .contains("#extension GL_ARB_shader_draw_parameters : require")
                );
            }
        }
    }

    #[test]
    fn wgsl_overrides_select_entry_and_validate_aliases() {
        use crate::backend::gl::profile::GlProfile;
        use fluxel_rhi_core::api::SpecializationValue as V;
        let source = r#"
            @id(7) override amount: f32;
            override unused: f32;
            @fragment fn a() -> @location(0) vec4f { return vec4f(amount); }
            @fragment fn b() -> @location(0) vec4f { return vec4f(0.5); }
        "#;
        let make = |entry: &str| ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_bytes()),
            entries: vec![ShaderEntryPoint {
                name: entry.into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: vec![],
        };
        let profile = GlProfile::WebGl2;
        let lower = |entry, values: &[(String, V)]| {
            super::wgsl::lower_wgsl_with_constants(profile, &make(entry), values)
        };
        assert!(lower("a", &[]).is_err());
        let (first, _, _) = lower("a", &[("amount".into(), V::F32(0.25))]).unwrap();
        let (second, _, _) = lower("a", &[("7".into(), V::F32(0.75))]).unwrap();
        assert_ne!(first.text, second.text);
        assert!(lower("b", &[]).is_ok());
        assert!(lower("a", &[("missing".into(), V::F32(0.25))]).is_err());
        assert!(lower("a", &[("7".into(), V::U32(1))]).is_err());
        assert!(
            lower(
                "a",
                &[("7".into(), V::F32(0.25)), ("amount".into(), V::F32(0.75))]
            )
            .is_err()
        );
    }

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
    fn direct_multiview_keeps_matching_layout_and_rejects_mismatch() {
        let count = std::num::NonZeroU32::new(2);
        let source = "#version 400 core\nlayout (\n num_views = 2\n) in;\nvoid main(){ uint v = gl_ViewID_OVR; }\n";
        let lowered =
            super::wgsl::lower_multiview(source.into(), ShaderStage::Vertex, count).unwrap();
        assert_eq!(lowered.matches("num_views").count(), 1);
        assert!(lowered.contains("gl_ViewID_OVR + uint(fluxel_internal_view_index_base)"));
        assert!(
            super::wgsl::lower_multiview(
                source.into(),
                ShaderStage::Vertex,
                std::num::NonZeroU32::new(3)
            )
            .is_err()
        );
    }

    #[test]
    fn multiview_wgsl_uses_ovr_and_pipeline_view_count() {
        use crate::backend::gl::profile::GlProfile;
        let bytes: Arc<[u8]> = Arc::from(br#"@vertex fn vs(@builtin(view_index) view:i32)->@builtin(position) vec4f { return vec4f(f32(view),0.,0.,1.); }"#.as_slice());
        let desc = ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes,
            entries: vec![ShaderEntryPoint {
                name: "vs".into(),
                stage: ShaderStage::Vertex,
                workgroup_size: None,
            }],
            required_features: vec![],
        };
        for profile in [
            GlProfile::Desktop { major: 4, minor: 6 },
            GlProfile::Embedded { major: 3, minor: 1 },
            GlProfile::WebGl2,
        ] {
            let (source, _, _) = super::wgsl::lower_wgsl_with_multiview(
                profile,
                &desc,
                &[],
                super::DrawParameters::None,
                std::num::NonZeroU32::new(2),
            )
            .unwrap();
            assert!(
                source
                    .text
                    .contains("#extension GL_OVR_multiview2 : require"),
                "{}",
                source.text
            );
            assert!(!source.text.contains("GL_EXT_multiview"), "{}", source.text);
            assert!(
                source.text.contains("layout(num_views = 2) in;"),
                "{}",
                source.text
            );
            assert!(
                source
                    .text
                    .contains("gl_ViewID_OVR + uint(fluxel_internal_view_index_base)"),
                "{}",
                source.text
            );
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
            let (vertex, _, _) =
                lower_wgsl(profile, &desc("vs_main", ShaderStage::Vertex)).unwrap();
            assert!(
                vertex
                    .text
                    .contains("gl_Position.z = 2.0 * gl_Position.z - gl_Position.w")
            );
            assert!(vertex.text.contains("gl_Position.y = -gl_Position.y"));
            assert_eq!(
                vertex.source_hash.0,
                source_key(vertex.text.as_bytes(), "vs_main")
            );

            let (_, bindings, _) =
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
            let (fragment, _, _) =
                lower_wgsl(profile, &desc("fs_main", ShaderStage::Fragment)).unwrap();
            assert!(!fragment.text.contains("fluxel_internal_flip_uv("));
        }
    }

    #[test]
    fn naga_compute_lowering_preserves_core_texture_coordinates() {
        use crate::backend::gl::profile::GlProfile;

        let (compute, bindings, _) =
            lower_wgsl(GlProfile::Desktop { major: 4, minor: 6 }, &compute_desc()).unwrap();

        assert!(!compute.text.contains("fluxel_internal_flip_texel("));
        assert!(compute.text.contains("texelFetch("));
        assert!(compute.text.contains("imageStore("), "{}", compute.text);
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
            let (fragment, _, _) =
                lower_wgsl(GlProfile::WebGl2, &raster_fallback_desc(entry)).unwrap();
            assert!(fragment.text.contains("#version 300 es"));
            assert!(fragment.text.contains("textureSize("), "{}", fragment.text);
            assert!(fragment.text.contains("texelFetch("), "{}", fragment.text);
        }
    }
}
