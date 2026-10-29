//! WGSL-to-HLSL lowering, binding mapping, and specialization regression cases.

use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};
use std::sync::Arc;

#[test]
fn compute_wgsl_compiles_with_mapped_raw_buffers() {
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&b"@group(2) @binding(3) var<storage, read_write> data: array<u32>; @compute @workgroup_size(1) fn main() { data[0] = 42u; }"[..]),
        entries: vec![ShaderEntryPoint { name: "main".into(), stage: ShaderStage::Compute, workgroup_size: Some([1, 1, 1]) }],
        required_features: vec![],
    };
    let result = super::compile(&desc).unwrap();
    assert!(
        result.entries[0].hlsl.contains("register(u3, space2)"),
        "{}",
        result.entries[0].hlsl
    );
    assert!(result.entries[0].hlsl.contains("RWByteAddressBuffer"));
    assert_eq!(result.entries[0].raw_storage_bindings, vec![(2, 3)]);
    assert!(!result.entries[0].bytes.is_empty());
    assert_eq!(result.entries[0].translated_name, "main");
}

#[test]
fn unsupported_abi_is_rejected() {
    let make = |source: &'static [u8]| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let huge_space = make(b"@group(256) @binding(0) var<storage,read> data: array<u32>; @compute @workgroup_size(1) fn main() { let x = data[0]; }");
    assert!(matches!(
        super::compile(&huge_space),
        Err(fluxel_rhi_core::api::Error::Unsupported(_))
    ));
    let num_workgroups = make(b"@compute @workgroup_size(1) fn main(@builtin(num_workgroups) n: vec3<u32>) { let x = n.x; }");
    assert!(super::compile(&num_workgroups).unwrap().entries[0].num_workgroups_constants);
    let mut wrong_workgroup = make(b"@compute @workgroup_size(2) fn main() {}");
    assert!(matches!(
        super::compile(&wrong_workgroup),
        Err(fluxel_rhi_core::api::Error::Shader(_))
    ));
    wrong_workgroup
        .entries
        .push(wrong_workgroup.entries[0].clone());
    assert!(matches!(
        super::compile(&wrong_workgroup),
        Err(fluxel_rhi_core::api::Error::Shader(_))
    ));
}

#[test]
fn sampler_array_uses_descriptor_table() {
    let source = b"@group(0) @binding(1) var s: binding_array<sampler, 2>; @group(0) @binding(4) var t: texture_2d<f32>; @group(0) @binding(5) var<uniform> index: u32; @group(0) @binding(6) var<storage,read_write> output: array<vec4<f32>>; @compute @workgroup_size(1) fn main() { output[0] = textureSampleLevel(t, s[index], vec2<f32>(0.5), 0.0); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let compiled = super::compile(&desc).unwrap();
    assert!(
        compiled.entries[0]
            .hlsl
            .contains("[2] : register(s1, space0)"),
        "{}",
        compiled.entries[0].hlsl
    );
    assert!(!compiled.entries[0].hlsl.contains("nagaSamplerHeap"));
    let entry = &desc.entries[0];
    let map = [
        (0, 1, 1, 3, 1),
        (0, 4, 4, 1, 1),
        (0, 5, 5, 0, 1),
        (0, 6, 6, 2, 1),
    ];
    assert!(matches!(
        super::compile_variant_with_map(&desc, entry, &[], &map),
        Err(fluxel_rhi_core::api::Error::Shader(_))
    ));
    let map = [
        (0, 1, 1, 1, 2),
        (0, 4, 4, 1, 1),
        (0, 5, 5, 0, 1),
        (0, 6, 6, 2, 1),
    ];
    assert!(matches!(
        super::compile_variant_with_map(&desc, entry, &[], &map),
        Err(fluxel_rhi_core::api::Error::Shader(_))
    ));
}

#[test]
fn runtime_binding_arrays_specialize_to_the_layout_capacity() {
    use fluxel_rhi_core::api::{
        BindingType, StorageAccess, TextureSampleType, TextureViewDimension,
    };
    let source = b"@group(0) @binding(0) var images: binding_array<texture_2d<f32>>; @group(0) @binding(1) var samplers: binding_array<sampler>; @group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>; @compute @workgroup_size(1) fn main() { out[0] = textureSampleLevel(images[1u], samplers[1u], vec2f(0.5), 0.0); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let map = [(0, 0, 0, 1, 3), (0, 1, 1, 3, 3), (0, 2, 2, 2, 1)];
    let types = [
        (
            0,
            0,
            BindingType::SampledTexture {
                sample: TextureSampleType::Float,
                dimension: TextureViewDimension::D2,
                multisampled: false,
            },
        ),
        (
            0,
            1,
            BindingType::Sampler {
                filtering: true,
                comparison: false,
            },
        ),
        (
            0,
            2,
            BindingType::StorageBuffer {
                dynamic_offset: false,
                min_size: 16,
                access: StorageAccess::ReadWrite,
            },
        ),
    ];
    let compiled = super::compile_variant_with_typed_map(
        &desc,
        &desc.entries[0],
        &[],
        &map,
        &types,
        &[(0, 0, 3), (0, 1, 3)],
        &[],
        None,
    )
    .unwrap();
    assert!(compiled.hlsl.contains("images[3]"), "{}", compiled.hlsl);
    assert!(
        compiled.hlsl.contains("[3] : register(s1"),
        "{}",
        compiled.hlsl
    );
    assert!(compiled.hlsl.contains("register(t0"), "{}", compiled.hlsl);
    assert!(matches!(
        super::compile_variant_with_typed_map(
            &desc,
            &desc.entries[0],
            &[],
            &map,
            &types,
            &[],
            &[],
            None,
        ),
        Err(fluxel_rhi_core::api::Error::Shader(_))
    ));
}

#[test]
fn required_wave_size_is_baked_into_wgsl_derived_dxil() {
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&b"@compute @workgroup_size(32) fn main() {}"[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([32, 1, 1]),
        }],
        required_features: vec![],
    };
    let compiled = super::compile_variant_with_typed_map(
        &desc,
        &desc.entries[0],
        &[],
        &[],
        &[],
        &[],
        &[],
        Some(32),
    )
    .unwrap();
    assert!(
        compiled.hlsl.contains("[WaveSize(32)]"),
        "{}",
        compiled.hlsl
    );
    assert!(compiled.bytes.starts_with(b"DXBC"));
}

#[test]
fn comparison_sampler_uses_regular_root_table() {
    let source = b"@group(0) @binding(1) var s: sampler_comparison; @group(0) @binding(2) var t: texture_depth_2d; @group(0) @binding(3) var<storage, read_write> out: array<f32>; @compute @workgroup_size(1) fn main() { out[0] = textureSampleCompareLevel(t, s, vec2<f32>(0.5), 0.5); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let compiled = super::compile(&desc).unwrap();
    assert!(
        compiled.entries[0]
            .hlsl
            .contains("SamplerComparisonState s : register(s1, space0)")
    );
}

#[test]
fn vertex_fragment_compute_compile_to_independent_dxil() {
    let source = br#"
        @vertex fn vs_main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
            return vec4<f32>(pos, 0.0, 1.0);
        }
        @fragment fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(0.25, 0.5, 0.75, 1.0);
        }
        @compute @workgroup_size(1) fn cs_main() {}
    "#;
    let entries = [
        ("vs_main", ShaderStage::Vertex),
        ("fs_main", ShaderStage::Fragment),
        ("cs_main", ShaderStage::Compute),
    ]
    .into_iter()
    .map(|(name, stage)| ShaderEntryPoint {
        name: name.into(),
        stage,
        workgroup_size: None,
    })
    .collect();
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries,
        required_features: vec![],
    };
    let compiled = super::compile(&desc).unwrap();
    assert_eq!(compiled.entries.len(), 3);
    for (entry, expected) in compiled.entries.iter().zip(&desc.entries) {
        assert_eq!(entry.name, expected.name);
        assert_eq!(entry.stage, expected.stage);
        assert!(entry.bytes.starts_with(b"DXBC"));
    }
}

#[test]
fn float_texture_load_uses_sampled_texture_register() {
    let source = b"@group(1) @binding(4) var image: texture_2d<f32>; @group(0) @binding(0) var<storage,read_write> data: array<u32>; @compute @workgroup_size(1) fn main() { data[0] = u32(textureLoad(image, vec2<i32>(0), 0).x); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let result = super::compile(&desc).unwrap();
    assert!(
        result.entries[0].hlsl.contains("register(t4, space1)"),
        "{}",
        result.entries[0].hlsl
    );
}

#[test]
fn storage_texture_write_uses_uav_register() {
    let source = b"@group(2) @binding(5) var image: texture_storage_2d<rgba8unorm,write>; @compute @workgroup_size(1) fn main() { textureStore(image, vec2<i32>(0), vec4<f32>(1.0)); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let result = super::compile(&desc).unwrap();
    assert!(
        result.entries[0].hlsl.contains("register(u5, space2)"),
        "{}",
        result.entries[0].hlsl
    );
}

#[test]
fn specialization_variants_validate_values_and_produce_distinct_dxil() {
    use fluxel_rhi_core::api::{Error, SpecializationValue};
    let source = b"@id(7) override value: u32 = 1u; @group(0) @binding(0) var<storage,read_write> out: array<u32>; @compute @workgroup_size(1) fn main() { out[0] = value; }";
    let entry = ShaderEntryPoint {
        name: "main".into(),
        stage: ShaderStage::Compute,
        workgroup_size: Some([1, 1, 1]),
    };
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![entry.clone()],
        required_features: vec![],
    };
    let a = super::compile_variant(&desc, &entry, &[("7".into(), SpecializationValue::U32(2))])
        .unwrap();
    let b = super::compile_variant(&desc, &entry, &[("7".into(), SpecializationValue::U32(3))])
        .unwrap();
    assert_ne!(a.bytes, b.bytes);
    assert!(matches!(
        super::compile_variant(&desc, &entry, &[("7".into(), SpecializationValue::I32(2))]),
        Err(Error::Shader(_))
    ));
    assert!(matches!(
        super::compile_variant(
            &desc,
            &entry,
            &[("missing".into(), SpecializationValue::U32(2))]
        ),
        Err(Error::Shader(_))
    ));
    assert!(matches!(
        super::compile_variant(
            &desc,
            &entry,
            &[
                ("7".into(), SpecializationValue::U32(2)),
                ("7".into(), SpecializationValue::U32(3))
            ]
        ),
        Err(Error::Shader(_))
    ));
}

#[test]
fn scalar_samplers_keep_group_and_binding() {
    let source = b"@group(0) @binding(0) var tex: texture_2d<f32>; @group(0) @binding(2) var a: sampler; @group(1) @binding(2) var b: sampler; @fragment fn main() -> @location(0) vec4<f32> { return textureSample(tex, a, vec2<f32>(0.5)) + textureSample(tex, b, vec2<f32>(0.5)); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Fragment,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let result = super::compile(&desc).unwrap();
    let hlsl = &result.entries[0].hlsl;
    assert!(hlsl.contains("register(s2, space0)"), "{hlsl}");
    assert!(hlsl.contains("register(s2, space1)"), "{hlsl}");
    assert!(!hlsl.contains("SamplerIndexArray"));
}
