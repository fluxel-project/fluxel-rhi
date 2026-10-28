use super::*;
#[test]
fn nested_two_resources() {
    let source = r#"@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@group(0) @binding(1) var<uniform> ti: u32;
@group(1) @binding(1) var<uniform> si: i32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
fn inner(t: texture_2d<f32>, s: sampler) -> vec4f { return textureSampleLevel(t, s, vec2f(0.2), 0.); }
fn outer(t: texture_2d<f32>, s: sampler) -> vec4f { return inner(t, s); }
@compute @workgroup_size(1) fn main() { output[0] = outer(textures[ti], samplers[si]); }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut aliases = HashMap::new();
    super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
    lower(&mut module).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    super::super::binding_array_dynamic_sample::lower(&mut module, &mut aliases, 8).unwrap();
    super::super::pairs::split(&mut module).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let opts = naga::back::glsl::Options {
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
        &opts,
        &pipeline,
        Default::default(),
    )
    .unwrap()
    .write()
    .unwrap();
    assert!(
        glsl.contains("switch") && glsl.contains("textureLod"),
        "{glsl}"
    );
}
#[test]
fn caller_without_resource_parameters_and_shared_access() {
    let source = r#"@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 2>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> index: u32;
@group(0) @binding(3) var<storage, read_write> output: array<vec4f>;
fn fetch_value(t: texture_2d<f32>) -> vec4f { return textureLoad(t, vec2i(0), 0); }
fn wrapper() -> vec4f { let t = textures[index]; return fetch_value(t) + textureLoad(t, vec2i(0), 0); }
@compute @workgroup_size(1) fn main() { output[0] = wrapper(); }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut aliases = HashMap::new();
    super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
    lower(&mut module).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    super::super::binding_array_dynamic_sampled_load::lower(&mut module, &mut aliases, 2).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
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
    assert!(
        glsl.contains("switch") && glsl.contains("texelFetch"),
        "{glsl}"
    );
}

#[test]
fn fragment_implicit_derivatives() {
    let source = r#"@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 2>;
@group(0) @binding(1) var samplers: binding_array<sampler, 2>;
@group(0) @binding(2) var<uniform> index: u32;
fn sample_it(t: texture_2d<f32>, s: sampler, uv: vec2f) -> vec4f { return textureSample(t, s, uv); }
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return sample_it(textures[index], samplers[index], uv); }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut aliases = HashMap::new();
    lower(&mut module).unwrap();
    super::super::binding_array_dynamic_sample::lower(&mut module, &mut aliases, 8).unwrap();
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
        glsl.contains("dFdx") && glsl.contains("textureGrad"),
        "{glsl}"
    );
}
#[test]
fn static_texture_dynamic_sampler() {
    let source = r#"@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@group(1) @binding(1) var<uniform> index: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
fn level(t: texture_2d<f32>, s: sampler) -> vec4f { return textureSampleLevel(t, s, vec2f(0.2), 0.); }
@compute @workgroup_size(1) fn main() { output[0] = level(textures[1], samplers[index]); }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut aliases = HashMap::new();
    super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
    lower(&mut module).unwrap();
    super::super::binding_array_dynamic_sample::lower(&mut module, &mut aliases, 8).unwrap();
    super::super::specialize::specialize(&mut module).unwrap();
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
    assert!(
        glsl.contains("switch") && glsl.contains("textureLod"),
        "{glsl}"
    );
}
