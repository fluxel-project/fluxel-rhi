    use super::*;

    fn verify(source: &str, needle: &str, with_consumers: bool) {
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
        lower(&mut module, &mut aliases, 2).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        if with_consumers {
            super::super::binding_array_dynamic_sampled_load::lower(&mut module, &mut aliases, 2)
                .unwrap();
            super::super::binding_array_dynamic_sample::lower(&mut module, &mut aliases, 2)
                .unwrap();
        }
        assert_eq!(aliases.len(), 2);
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
        assert!(glsl.contains("switch") && glsl.contains(needle), "{glsl}");
        if with_consumers {
            assert!(
                glsl.contains("texelFetch") && glsl.contains("textureLod"),
                "{glsl}"
            );
        }
    }

    #[test]
    fn dimensions_and_levels() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(0) @binding(1) var<uniform> selected: i32;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1) fn main() { let t = images[selected]; output[0] = textureDimensions(t, 1).x + textureNumLevels(t); }"#,
            "textureSize",
            false,
        );
    }

    #[test]
    fn array_layers() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_2d_array<f32>, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1) fn main() { output[0] = textureNumLayers(images[selected]); }"#,
            "textureSize",
            false,
        );
    }

    #[test]
    fn samples() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_multisampled_2d<f32>, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1) fn main() { output[0] = textureNumSamples(images[selected]); }"#,
            "textureSamples",
            false,
        );
    }

    #[test]
    fn depth_dimensions() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_depth_2d, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec2u>;
@compute @workgroup_size(1) fn main() { output[0] = textureDimensions(images[selected], 0); }"#,
            "textureSize",
            false,
        );
    }

    #[test]
    fn shared_query_load_sample() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samp: sampler;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { let t = images[selected]; let n = textureDimensions(t).x; output[0] = textureLoad(t, vec2i(0), 0) + textureSampleLevel(t, samp, vec2f(0.2), f32(n) * 0.); }"#,
            "textureSize",
            true,
        );
    }

    #[test]
    fn static_and_dynamic_alias_and_zero_default() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_3d<u32>, 2>;
@group(0) @binding(1) var<uniform> selected: i32;
@group(0) @binding(2) var<storage, read_write> output: array<vec3u>;
@compute @workgroup_size(1) fn main() { output[0] = textureDimensions(images[selected], 1) + textureDimensions(images[1], 0); }"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
        lower(&mut module, &mut aliases, 2).unwrap();
        assert_eq!(aliases.len(), 2);
        let f = &module.entry_points[0].function;
        assert!(
            f.expressions
                .iter()
                .any(|(_, e)| matches!(e, Expression::ZeroValue(_)))
        );
        let switches: Vec<_> = f
            .body
            .iter()
            .filter_map(|s| {
                if let Statement::Switch { cases, .. } = s {
                    Some(cases)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(switches.len(), 1);
        assert!(
            switches[0]
                .iter()
                .any(|c| matches!(c.value, naga::SwitchValue::Default) && c.body.is_empty())
        );
        assert!(
            switches[0]
                .iter()
                .any(|c| matches!(c.value, naga::SwitchValue::I32(0)))
        );
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
        assert!(glsl.contains("textureSize"), "{glsl}");
    }

    #[test]
    fn selected_entry_ignores_other_entry_array() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<i32>, 2>;
@group(0) @binding(1) var unused: binding_array<texture_2d<i32>, 100>;
@group(0) @binding(2) var<uniform> selected: u32;
@group(0) @binding(3) var<storage, read_write> output: array<vec2u>;
@compute @workgroup_size(1) fn keep() { output[0] = textureDimensions(images[selected]); }
@compute @workgroup_size(1) fn other() { output[0] = textureDimensions(unused[selected]); }"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        module.entry_points.retain(|e| e.name == "keep");
        naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 2).unwrap();
        assert_eq!(aliases.len(), 2);
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
