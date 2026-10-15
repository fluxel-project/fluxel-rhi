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
        // adopted context may not have a proved compute/image route.  Refuse
        // the shader before choosing a compiler target so GL 4.0--4.2 and
        // WebGL2 never appear to accept an executable compute pipeline.
        if compute && !wgsl::compute_storage_supported(device) {
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
