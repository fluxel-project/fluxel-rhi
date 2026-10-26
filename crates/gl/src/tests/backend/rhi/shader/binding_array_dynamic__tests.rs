    use super::*;
    fn verify(source: &str, needle: &str) {
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 256).unwrap();
        assert_eq!(aliases.len(), 2);
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
        assert!(glsl.contains("switch"), "{glsl}");
        assert!(glsl.contains(needle), "{glsl}");
    }
    #[test]
    fn dynamic_store_in_helper_and_loop() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;
@group(0) @binding(1) var<uniform> select: u32;
fn put(i:u32) { if (i < 4u) { for(var n=0u;n<1u;n++) { textureStore(images[i],vec2i(0),vec4f(1.)); } } }
@compute @workgroup_size(1) fn main() { put(select); }"#,
            "imageStore",
        );
    }
    #[test]
    fn signed_dynamic_store() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;
@group(0) @binding(1) var<uniform> select: i32;
@compute @workgroup_size(1) fn main() { textureStore(images[select],vec2i(0),vec4f(1.)); }"#,
            "imageStore",
        );
    }
    #[test]
    fn dynamic_atomic() {
        verify(
            r#"
@group(0) @binding(0) var images: binding_array<texture_storage_2d<r32uint, atomic>, 2>;
@group(0) @binding(1) var<uniform> select: u32;
@compute @workgroup_size(1) fn main() { textureAtomicAdd(images[select],vec2i(0),1u); }"#,
            "imageAtomicAdd",
        );
    }
    #[test]
    fn oversized_array_rejected_before_mutation() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_storage_2d<rgba8unorm, write>, 65>;
@group(0) @binding(1) var<uniform> select:u32;
@compute @workgroup_size(1) fn main() { textureStore(images[select],vec2i(0),vec4f(1.)); }"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let before = module.global_variables.len();
        assert!(matches!(
            lower(&mut module, &mut HashMap::new(), 64),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(module.global_variables.len(), before);
    }
