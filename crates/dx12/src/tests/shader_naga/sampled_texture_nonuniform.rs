//! HLSL code generation must preserve divergent sampled-texture indexing.

use super::*;
use std::sync::Arc;

#[test]
fn sampled_texture_array_index_uses_nonuniform_resource_index() {
    let desc = api::ShaderDesc {
        label: Some("sampled texture nonuniform HLSL fixture".into()),
        format: api::ShaderFormat::Wgsl,
        bytes: Arc::from(
            b"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>; @group(0) @binding(1) var<storage, read_write> output: array<u32, 2>; @compute @workgroup_size(2) fn main(@builtin(local_invocation_index) i: u32) { let value = textureLoad(images[i], vec2<i32>(0), 0).r; output[i] = u32(round(value * 255.0)); }"
                .as_slice(),
        ),
        entries: vec![api::ShaderEntryPoint {
            name: "main".into(),
            stage: api::ShaderStage::Compute,
            workgroup_size: Some([2, 1, 1]),
        }],
        required_features: vec![],
    };
    let entry = &desc.entries[0];
    let compiled = compile_variant(&desc, entry, &[]).unwrap();
    assert!(compiled.hlsl.contains("NonUniformResourceIndex"));
}
