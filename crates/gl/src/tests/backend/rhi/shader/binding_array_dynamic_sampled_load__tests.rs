use super::*;

fn verify(source: &str) {
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut aliases = HashMap::new();
    lower(&mut module, &mut aliases, 2).unwrap();
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
        output.contains("switch") && output.contains("texelFetch"),
        "{output}"
    );
}

#[test]
fn d2_float_in_helper_loop() {
    verify(
        r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
fn fetch_image(i: u32) -> vec4f { var value = vec4f(0.); for (var j=0; j<2; j++) { value += textureLoad(images[i], vec2i(j), 0); } return value; }
@compute @workgroup_size(1) fn main() { output[0] = fetch_image(selected); }"#,
    );
}
#[test]
fn d2_array_sint() {
    verify(
        r#"@group(0) @binding(0) var images: binding_array<texture_2d_array<i32>, 2>;
@group(0) @binding(1) var<uniform> selected: i32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4i>;
@compute @workgroup_size(1) fn main() { output[0] = textureLoad(images[selected], vec2i(0), 1, 0); }"#,
    );
}
#[test]
fn d3_uint() {
    verify(
        r#"@group(0) @binding(0) var images: binding_array<texture_3d<u32>, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4u>;
@compute @workgroup_size(1) fn main() { output[0] = textureLoad(images[selected], vec3i(0), 0); }"#,
    );
}
#[test]
fn multisample() {
    verify(
        r#"@group(0) @binding(0) var images: binding_array<texture_multisampled_2d<f32>, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { output[0] = textureLoad(images[selected], vec2i(0), 1); }"#,
    );
}
#[test]
fn depth_2d() {
    verify(
        r#"@group(0) @binding(0) var images: binding_array<texture_depth_2d, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(1) fn main() { output[0] = textureLoad(images[selected], vec2i(0), 0); }"#,
    );
}
#[test]
fn limit_is_checked_before_alias_allocation() {
    let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 3>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { output[0] = textureLoad(images[selected], vec2i(0), 0); }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let globals = module.global_variables.len();
    let mut aliases = HashMap::new();
    assert!(matches!(
        lower(&mut module, &mut aliases, 2),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(module.global_variables.len(), globals);
    assert!(aliases.is_empty());
}
#[test]
fn static_and_dynamic_share_aliases() {
    let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { output[0] = textureLoad(images[1], vec2i(0), 0) + textureLoad(images[selected], vec2i(0), 0); }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut aliases = HashMap::new();
    super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
    lower(&mut module, &mut aliases, 2).unwrap();
    assert_eq!(aliases.len(), 2);
    super::super::pairs::split(&mut module).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn shared_texture_load_and_sample() {
    let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samp: sampler;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { let t = images[selected]; output[0] = textureLoad(t, vec2i(0), 0) + textureSampleLevel(t, samp, vec2f(0.2, 0.8), 0.); }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut aliases = HashMap::new();
    lower(&mut module, &mut aliases, 2).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    super::super::binding_array_dynamic_sample::lower(&mut module, &mut aliases, 2).unwrap();
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
        output.contains("texelFetch") && output.contains("textureLod"),
        "{output}"
    );
}

#[test]
fn shared_texture_multiple_loads() {
    verify(
        r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { let t = images[selected]; output[0] = textureLoad(t, vec2i(0), 0) + textureLoad(t, vec2i(1), 0); }"#,
    );
}
