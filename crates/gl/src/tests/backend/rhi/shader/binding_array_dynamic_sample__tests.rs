    use super::*;
    #[test]
    fn shared_selector_uses_only_diagonal_pairs() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 4>;
@group(1) @binding(0) var samplers: binding_array<sampler, 3>;
@fragment fn main(@location(0) uv: vec2f, @location(1) @interpolate(flat) i: u32) -> @location(0) vec4f {
    return textureSample(images[i], samplers[i], uv);
}"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 3).unwrap();
        super::super::pairs::split(&mut module).unwrap();
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
        let mut glsl = String::new();
        let reflection = naga::back::glsl::Writer::new(
            &mut glsl,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap()
        .write()
        .unwrap();
        assert_eq!(reflection.texture_mapping.len(), 3, "{glsl}");
    }
    #[test]
    fn cube_array_shadow_samples_before_divergent_selection() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_depth_cube_array, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler_comparison, 2>;
@fragment fn main(@location(0) dir: vec3f, @location(1) @interpolate(flat) i: u32) -> @location(0) f32 {
    return textureSampleCompare(images[i], samplers[i], dir, 0, 0.5);
}"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 4).unwrap();
        super::super::pairs::split(&mut module).unwrap();
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
        let mut glsl = String::new();
        naga::back::glsl::Writer::new(
            &mut glsl,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap()
        .write()
        .unwrap();
        let switch = glsl.find("switch").unwrap();
        assert!(
            glsl.find("texture(").unwrap() < switch && !glsl.contains("textureGrad"),
            "{glsl}"
        );
    }
    #[test]
    fn implicit_derivatives_are_computed_before_selector_switch() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@fragment fn main(@location(0) uv: vec2f, @location(1) @interpolate(flat) i: u32) -> @location(0) vec4f {
    return textureSample(images[i], samplers[i], uv);
}"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 4).unwrap();
        super::super::pairs::split(&mut module).unwrap();
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
        let mut glsl = String::new();
        naga::back::glsl::Writer::new(
            &mut glsl,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap()
        .write()
        .unwrap();
        let switch = glsl.find("switch").unwrap();
        assert!(
            glsl.find("dFdx").unwrap() < switch
                && glsl.find("dFdy").unwrap() < switch
                && glsl.contains("textureGrad"),
            "{glsl}"
        );
    }

    #[test]
    fn dynamic_bias_is_lowered_into_scaled_gradients() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@fragment fn main(@location(0) uv: vec2f, @location(1) @interpolate(flat) i: u32) -> @location(0) vec4f {
    return textureSampleBias(images[i], samplers[i], uv, 1.0);
}"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 4).unwrap();
        super::super::pairs::split(&mut module).unwrap();
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
        let mut glsl = String::new();
        naga::back::glsl::Writer::new(
            &mut glsl,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap()
        .write()
        .unwrap();
        let switch = glsl.find("switch").unwrap();
        assert!(
            glsl.find("dFdx").unwrap() < switch
                && glsl.find("dFdy").unwrap() < switch
                && glsl.contains("exp2")
                && glsl.contains("textureGrad"),
            "{glsl}"
        );
    }

    #[test]
    fn dynamic_gather_is_specialized_per_resource_pair() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@fragment fn main(@location(0) uv: vec2f, @location(1) @interpolate(flat) i: u32) -> @location(0) vec4f {
    return textureGather(0, images[i], samplers[i], uv);
}"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 4).unwrap();
        super::super::pairs::split(&mut module).unwrap();
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
        let mut glsl = String::new();
        naga::back::glsl::Writer::new(
            &mut glsl,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap()
        .write()
        .unwrap();
        assert!(
            glsl.contains("textureGather") && glsl.contains("switch"),
            "{glsl}"
        );
    }

    fn verify(source: &str, expected_aliases: usize, limit: u32) -> String {
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
        lower(&mut module, &mut aliases, limit).unwrap();
        assert_eq!(aliases.len(), expected_aliases);
        super::super::pairs::split(&mut module).unwrap();
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
            shader_stage: naga::ShaderStage::Compute,
            entry_point: "main".into(),
            multiview: None,
        };
        let mut output = String::new();
        naga::back::glsl::Writer::new(
            &mut output,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap()
        .write()
        .unwrap();
        assert!(
            output.contains("switch") && output.contains("textureLod"),
            "{output}"
        );
        output
    }
    #[test]
    fn dynamic_image_sampler_different_groups_and_offset() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d_array<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 3>;
@group(0) @binding(1) var<uniform> ti: u32;
@group(1) @binding(1) var<uniform> si: i32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { output[0] = textureSampleLevel(images[ti], samplers[si], vec2f(0.25, 0.75), 1, 1.5, vec2i(1, -1)); }"#;
        let glsl = verify(source, 5, 6);
        assert!(glsl.contains("textureLodOffset"), "{glsl}");
    }
    #[test]
    fn static_texture_dynamic_sampler() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_3d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@group(1) @binding(1) var<uniform> si: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { output[0] = textureSampleLevel(images[1], samplers[si], vec3f(0.1, 0.2, 0.3), 2.); }"#,
            3,
            2,
        );
    }
    #[test]
    fn dynamic_texture_static_sampler() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@group(0) @binding(1) var<uniform> ti: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { output[0] = textureSampleLevel(images[ti], samplers[1], vec2f(0.2), 0.); }"#,
            3,
            2,
        );
    }
    #[test]
    fn shared_selected_resources() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@group(0) @binding(1) var<uniform> ti: u32;
@group(1) @binding(1) var<uniform> si: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { let image = images[ti]; let sampler0 = samplers[si]; output[0] = textureSampleLevel(image, sampler0, vec2f(0.2), 0.) + textureSampleLevel(image, sampler0, vec2f(0.4), 1.); }"#,
            4,
            4,
        );
    }
    #[test]
    fn too_many_combined_pairs_are_rejected() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 3>;
@group(0) @binding(1) var<uniform> ti: u32;
@group(1) @binding(1) var<uniform> si: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { output[0] = textureSampleLevel(images[ti], samplers[si], vec2f(0.2), 0.); }"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        assert!(matches!(
            lower(&mut module, &mut aliases, 5),
            Err(Error::Unsupported(_))
        ));
        assert!(aliases.is_empty());
    }
