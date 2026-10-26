use super::*;
use crate::backend::bindings::{MetalBindGroupLayout, MetalPipelineLayout};
#[test]
fn push_constants_follow_resource_buffers() {
    let layout = MetalPipelineLayout {
        groups: vec![MetalBindGroupLayout {
            entries: vec![api::BindGroupLayoutEntry {
                binding: 40,
                visibility: vec![api::ShaderStage::Compute],
                ty: api::BindingType::UniformBuffer {
                    min_size: 16,
                    dynamic_offset: false,
                },
                count: api::BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        }],
        push_constants: vec![api::PushConstantRange {
            stages: vec![api::ShaderStage::Compute],
            range: 0..16,
        }],
    };
    let source = b"struct Data { value: vec4<u32> }; @group(0) @binding(40) var<uniform> data: Data; var<push_constant> push: Data; @compute @workgroup_size(1) fn main() { let x = data.value + push.value; }";
    let variant = translate(
        source,
        "main",
        api::ShaderStage::Compute,
        &[],
        &layout,
        0,
        128,
    )
    .unwrap();
    assert_eq!(variant.map.slot(0, 40).unwrap().index, 0);
    assert_eq!(variant.map.push_slot, Some(1));
}
#[test]
fn specialization_maps_storage_and_its_runtime_length() {
    let layout = MetalPipelineLayout {
        groups: vec![MetalBindGroupLayout {
            entries: vec![api::BindGroupLayoutEntry {
                binding: 9,
                visibility: vec![api::ShaderStage::Compute],
                ty: api::BindingType::StorageBuffer {
                    min_size: 4,
                    dynamic_offset: true,
                    access: api::StorageAccess::ReadWrite,
                },
                count: api::BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        }],
        push_constants: Vec::new(),
    };
    let source = b"override scale: u32 = 1; @group(0) @binding(9) var<storage, read_write> data: array<u32>; @compute @workgroup_size(1) fn main() { if (arrayLength(&data) > 0u) { data[0] = scale; } }";
    let v = translate(
        source,
        "main",
        api::ShaderStage::Compute,
        &[("scale".into(), api::SpecializationValue::U32(7))],
        &layout,
        0,
        128,
    )
    .unwrap();
    assert_eq!(v.size_bindings, vec![(0, 9, 1)]);
    assert_eq!(v.map.slot(0, 9).unwrap().index, 0);
    assert_eq!(v.map.sizes_slot, Some(1));
    assert!(v.source.contains("_mslBufferSizes"));
}
