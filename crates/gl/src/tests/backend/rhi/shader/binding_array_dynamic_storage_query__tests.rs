    use super::*;
    fn validate(module: &naga::Module) -> naga::valid::ModuleInfo {
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(module)
        .unwrap()
    }
    fn verify(source: &str, consumers: bool) {
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        validate(&module);
        let mut aliases = HashMap::new();
        super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
        lower(&mut module, &mut aliases, 2).unwrap();
        validate(&module);
        if consumers {
            super::super::binding_array_dynamic::lower(&mut module, &mut aliases, 2).unwrap();
            validate(&module);
            super::super::binding_array_dynamic_load::lower(&mut module, &mut aliases, 2).unwrap();
            validate(&module);
        }
        assert_eq!(aliases.len(), 2);
        let info = validate(&module);
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
        assert!(
            glsl.contains("switch") && glsl.contains("imageSize"),
            "{glsl}"
        );
        if consumers {
            assert!(
                glsl.contains("imageLoad") && glsl.contains("imageStore"),
                "{glsl}"
            );
        }
        assert!(
            module.entry_points[0]
                .function
                .expressions
                .iter()
                .any(|(_, e)| matches!(e, Expression::ZeroValue(_)))
        );
    }
    #[test]
    fn dimensions_signed_static_alias() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_storage_2d<rgba8unorm, read_write>, 2>;
@group(0) @binding(1) var<uniform> selected: i32;
@group(0) @binding(2) var<storage, read_write> output: array<vec2u>;
@compute @workgroup_size(1) fn main() { output[0] = textureDimensions(images[selected]) + textureDimensions(images[1]); }"#,
            false,
        );
    }
    #[test]
    fn array_layers() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_storage_2d_array<rgba8unorm, read_write>, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1) fn main() { output[0] = textureNumLayers(images[selected]); }"#,
            false,
        );
    }
    #[test]
    fn shared_query_store_load() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_storage_2d<rgba8unorm, read_write>, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { let image = images[selected]; let size = textureDimensions(image); textureStore(image, vec2i(0), vec4f(0.5)); output[0] = textureLoad(image, vec2i(0)) + vec4f(f32(size.x) * 0.); }"#,
            true,
        );
    }
