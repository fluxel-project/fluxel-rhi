//! Shader publication and WGSL lowering regression tests.

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
        spirv: None,
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
        spirv: None,
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
    include_str!("../../../../../../rhi/examples/assets/06_compute/filters_raster.wgsl");

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
    let lowered = super::wgsl::lower_multiview(source.into(), ShaderStage::Vertex, count).unwrap();
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
        let (vertex, _, _) = lower_wgsl(profile, &desc("vs_main", ShaderStage::Vertex)).unwrap();
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
        let (fragment, _, _) = lower_wgsl(GlProfile::WebGl2, &raster_fallback_desc(entry)).unwrap();
        assert!(fragment.text.contains("#version 300 es"));
        assert!(fragment.text.contains("textureSize("), "{}", fragment.text);
        assert!(fragment.text.contains("texelFetch("), "{}", fragment.text);
    }
}
