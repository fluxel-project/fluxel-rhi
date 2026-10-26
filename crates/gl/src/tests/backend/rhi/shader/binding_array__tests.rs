    use super::*;

    #[test]
    fn fixed_texture_element_remains_valid_naga_ir() {
        let source = r#"
@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@fragment fn main() -> @location(0) vec4f {
    return textureLoad(images[1], vec2i(0), 0);
}
"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        split_static(&mut module, &mut aliases).unwrap();
        assert_eq!(aliases.len(), 1);
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let options = naga::back::glsl::Options {
            version: naga::back::glsl::Version::Desktop(450),
            writer_flags: naga::back::glsl::WriterFlags::empty(),
            binding_map: Default::default(),
            zero_initialize_workgroup_memory: true,
        };
        let pipeline = naga::back::glsl::PipelineOptions {
            shader_stage: naga::ShaderStage::Fragment,
            entry_point: "main".into(),
            multiview: None,
        };
        let mut text = String::new();
        let mut writer = naga::back::glsl::Writer::new(
            &mut text,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap();
        writer.write().unwrap();
        assert!(text.contains("texelFetch"));
    }

    #[test]
    fn reflection_preserves_original_binding_and_selected_element() {
        use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};
        let source = b"const SELECTED: u32 = 1u; @group(1) @binding(3) var images: binding_array<texture_2d<f32>, 2>; @fragment fn main() -> @location(0) vec4f { return textureLoad(images[SELECTED], vec2i(0), 0); }";
        let desc = ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: std::sync::Arc::from(source.as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: Vec::new(),
        };
        let (_, bindings, _) = super::super::wgsl::lower_wgsl(
            crate::backend::profile::GlProfile::Desktop { major: 4, minor: 6 },
            &desc,
        )
        .unwrap();
        assert!(bindings.iter().any(|binding| binding.group == 1
            && binding.binding == 3
            && binding.element == Some(1)));
    }

    #[test]
    fn separate_array_elements_keep_their_sampler_pairs() {
        use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};
        let source = b"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>; @group(1) @binding(0) var samplers: binding_array<sampler, 2>; @fragment fn main() -> @location(0) vec4f { return textureSample(images[0], samplers[1], vec2f(0.5)) + textureSample(images[1], samplers[0], vec2f(0.5)); }";
        let desc = ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: std::sync::Arc::from(source.as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: Vec::new(),
        };
        let (_, bindings, _) = super::super::wgsl::lower_wgsl(
            crate::backend::profile::GlProfile::Desktop { major: 4, minor: 6 },
            &desc,
        )
        .unwrap();
        for (texture, sampler) in [(0, 1), (1, 0)] {
            let image = bindings
                .iter()
                .find(|v| v.group == 0 && v.element == Some(texture))
                .unwrap();
            assert!(
                bindings
                    .iter()
                    .any(|v| v.group == 1 && v.element == Some(sampler) && v.name == image.name)
            );
        }
    }

    #[test]
    fn constant_array_element_can_be_passed_to_helper() {
        use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};
        let source = b"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>; fn fetch(image: texture_2d<f32>) -> vec4f { return textureLoad(image, vec2i(0), 0); } @fragment fn main() -> @location(0) vec4f { return fetch(images[1]); }";
        let desc = ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: std::sync::Arc::from(source.as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: Vec::new(),
        };
        let (_, bindings, _) = super::super::wgsl::lower_wgsl(
            crate::backend::profile::GlProfile::Desktop { major: 4, minor: 6 },
            &desc,
        )
        .unwrap();
        assert!(
            bindings
                .iter()
                .any(|v| v.binding == 0 && v.element == Some(1))
        );
    }

    #[test]
    fn storage_buffer_array_element_keeps_store_side_effect() {
        use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};
        let source = b"struct Data { value: u32 }; @group(0) @binding(0) var<storage, read_write> buffers: binding_array<Data, 2>; @compute @workgroup_size(1) fn main() { buffers[1].value = 42u; }";
        let desc = ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: std::sync::Arc::from(source.as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: Vec::new(),
        };
        let (glsl, bindings, _) = super::super::wgsl::lower_wgsl(
            crate::backend::profile::GlProfile::Desktop { major: 4, minor: 6 },
            &desc,
        )
        .unwrap();
        assert!(
            bindings
                .iter()
                .any(|binding| binding.binding == 0 && binding.element == Some(1))
        );
        assert!(glsl.text.contains("42u"));
    }

    #[test]
    fn storage_image_array_element_keeps_store_and_unit() {
        use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};
        let source = b"@group(0) @binding(2) var images: binding_array<texture_storage_2d<rgba8unorm, write>, 2>; @compute @workgroup_size(1) fn main() { textureStore(images[1], vec2i(0), vec4f(1.)); }";
        let desc = ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: std::sync::Arc::from(source.as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: Vec::new(),
        };
        let (glsl, bindings, _) = super::super::wgsl::lower_wgsl(
            crate::backend::profile::GlProfile::Desktop { major: 4, minor: 6 },
            &desc,
        )
        .unwrap();
        assert!(bindings.iter().any(|binding| binding.binding == 2
            && binding.element == Some(1)
            && binding.image_unit.is_some()));
        assert!(glsl.text.contains("imageStore"));
    }
