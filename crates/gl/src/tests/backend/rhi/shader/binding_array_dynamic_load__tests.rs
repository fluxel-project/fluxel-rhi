use super::*;
fn verify(source: &str) {
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
    assert!(output.contains("switch"), "{output}");
    assert!(output.contains("imageLoad"), "{output}");
}
#[test]
fn helper_loop_and_order() {
    verify(
        r#"@group(0) @binding(0) var images:binding_array<texture_storage_2d<rgba8unorm, read>,2>;
@group(0) @binding(1) var<uniform> selected:u32;
@group(0) @binding(2) var<storage,read_write> out:array<vec4f>;
fn sample(i:u32)->vec4f { var sum=vec4f(0.); for(var k=0u;k<2u;k++) { sum += textureLoad(images[i+k],vec2i(i32(k))); } return sum; }
@compute @workgroup_size(1) fn main() { out[0]=sample(selected); }"#,
    );
}
#[test]
fn signed_selector() {
    verify(
        r#"@group(0) @binding(0) var images:binding_array<texture_storage_2d<rgba8unorm, read>,2>;
@group(0) @binding(1) var<uniform> selected:i32;
@group(0) @binding(2) var<storage,read_write> out:array<vec4f>;
@compute @workgroup_size(1) fn main() { out[0]=textureLoad(images[selected],vec2i(0)); }"#,
    );
}
#[test]
fn store_load_and_static_share_one_alias_per_element() {
    let source = r#"@group(0) @binding(0) var images:binding_array<texture_storage_2d<rgba8unorm, read_write>,2>;
@group(0) @binding(1) var<uniform> selected:u32;
@group(0) @binding(2) var<storage,read_write> output:array<vec4f>;
@compute @workgroup_size(1) fn main() {
    textureStore(images[selected],vec2i(0),vec4f(1.));
    output[0]=textureLoad(images[selected],vec2i(0));
    textureStore(images[1],vec2i(0),vec4f(0.));
}"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut aliases = HashMap::new();
    super::super::binding_array_dynamic::lower(&mut module, &mut aliases, 256).unwrap();
    super::lower(&mut module, &mut aliases, 256).unwrap();
    super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
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
        output.contains("imageLoad") && output.contains("imageStore"),
        "{output}"
    );
}
