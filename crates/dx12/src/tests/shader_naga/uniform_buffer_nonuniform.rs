//! Reconstructed CBV-array coverage for native declarations and divergent indexing.

use super::*;
use std::sync::Arc;

#[test]
fn uniform_binding_array_emits_native_cbv_array_and_nonuniform_selector() {
    let source = include_str!("../../../tests/assets/uniform_buffer_nonuniform.wgsl");
    for source in [
        source.to_owned(),
        source.replace(
            "output[i] = values[i].value.x;",
            "let item = values[i]; output[i] = item.value.x;",
        ),
    ] {
        let desc = api::ShaderDesc {
            label: Some("reconstructed uniform buffer array".into()),
            format: api::ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_bytes()),
            entries: vec![api::ShaderEntryPoint {
                name: "main".into(),
                stage: api::ShaderStage::Compute,
                workgroup_size: Some([2, 1, 1]),
            }],
            required_features: vec![],
        };
        let binding = api::BindingType::UniformBuffer {
            dynamic_offset: false,
            min_size: 32,
        };
        let entry = api::BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![api::ShaderStage::Compute],
            ty: binding.clone(),
            count: api::BindingCount::Fixed(2),
            partially_bound: false,
            nonuniform_indexing: true,
        };
        let compile = |entry| {
            compile_variant_with_typed_map(
                &desc,
                &desc.entries[0],
                &[],
                &[(0, 0, 0, 0, 2), (0, 2, 2, 2, 1)],
                &[
                    (0, 0, binding.clone()),
                    (
                        0,
                        2,
                        api::BindingType::StorageBuffer {
                            dynamic_offset: false,
                            min_size: 8,
                            access: api::StorageAccess::ReadWrite,
                        },
                    ),
                ],
                &[],
                &[(0, entry)],
                None,
            )
        };
        let compiled = compile(entry.clone()).unwrap();
        eprintln!("{}", compiled.hlsl);
        assert!(
            compiled
                .hlsl
                .contains("ConstantBuffer<Value> values[2] : register(b0)")
        );
        assert!(compiled.hlsl.contains("values[NonUniformResourceIndex(i)]"));
        assert!(compiled.hlsl.contains("register(u2)"));
        assert!(!compiled.hlsl.contains("cbuffer values[2]"));
        assert!(!compiled.bytes.is_empty());
        for invalid in [
            api::BindGroupLayoutEntry {
                nonuniform_indexing: false,
                ..entry.clone()
            },
            api::BindGroupLayoutEntry {
                partially_bound: true,
                ..entry.clone()
            },
            api::BindGroupLayoutEntry {
                count: api::BindingCount::RuntimeSized { max_count: 2 },
                ..entry.clone()
            },
            api::BindGroupLayoutEntry {
                ty: api::BindingType::UniformBuffer {
                    dynamic_offset: true,
                    min_size: 32,
                },
                ..entry.clone()
            },
        ] {
            assert!(matches!(compile(invalid), Err(api::Error::Unsupported(_))));
        }
    }
}
