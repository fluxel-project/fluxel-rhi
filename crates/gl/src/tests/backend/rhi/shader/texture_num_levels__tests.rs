
use super::*;
fn run(source: &str, dynamic: bool) {
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut aliases = HashMap::new();
    super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
    super::super::binding_array_dynamic_sampled_query::lower(&mut module, &mut aliases, 2).unwrap();
    let mappings = lower(&mut module, "comp").unwrap();
    assert_eq!(mappings.len(), if dynamic { 2 } else { 1 });
    super::super::binding_array_dynamic_sampled_load::lower(&mut module, &mut aliases, 2).unwrap();
    super::super::binding_array_dynamic_sample::lower(&mut module, &mut aliases, 2).unwrap();
    super::super::pairs::split(&mut module).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let options = naga::back::glsl::Options {
        version: naga::back::glsl::Version::Embedded {
            version: 310,
            is_webgl: false,
        },
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
    assert!(!glsl.contains("textureQueryLevels"), "{glsl}");
    for (name, _) in mappings {
        assert!(glsl.contains(&format!("uint {name} = 0u;")), "{glsl}");
    }
    if dynamic {
        assert!(
            glsl.contains("switch") && glsl.contains("texelFetch") && glsl.contains("textureLod"),
            "{glsl}"
        );
    }
}
#[test]
fn dynamic_shared_query_load_sample() {
    run(
        r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samp: sampler;
@group(0) @binding(1) var<uniform> selected: i32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { let t = images[selected]; let n = textureNumLevels(t); output[0] = textureLoad(t, vec2i(0), 0) + textureSampleLevel(t, samp, vec2f(0.2), f32(n) * 0.); }"#,
        true,
    );
}
#[test]
fn direct_texture() {
    run(
        r#"@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1) fn main() { output[0] = textureNumLevels(image); }"#,
        false,
    );
}
