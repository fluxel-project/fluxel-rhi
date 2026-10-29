//! Shader-layout typing and declared-capability admission regressions.

use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};
use std::sync::Arc;

#[test]
fn texture_binding_array_compiles_with_layout_class_and_count() {
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&b"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>; @group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>; @compute @workgroup_size(1) fn main() { out[0] = textureLoad(images[1], vec2<i32>(0), 0); }"[..]),
        entries: vec![ShaderEntryPoint { name: "main".into(), stage: ShaderStage::Compute, workgroup_size: Some([1,1,1]) }],
        required_features: vec![],
    };
    let entry = &desc.entries[0];
    let result =
        super::compile_variant_with_map(&desc, entry, &[], &[(0, 0, 0, 1, 2), (0, 2, 2, 2, 1)])
            .unwrap();
    assert!(!result.bytes.is_empty());
    assert!(
        super::compile_variant_with_map(&desc, entry, &[], &[(0, 0, 0, 1, 1), (0, 2, 2, 2, 1)])
            .is_err()
    );
    assert!(
        super::compile_variant_with_map(&desc, entry, &[], &[(0, 0, 0, 2, 2), (0, 2, 2, 2, 1)])
            .is_err()
    );
}

#[test]
fn typed_texture_and_sampler_mismatches_are_rejected() {
    use fluxel_rhi_core::api::{
        BindingType as B, StorageAccess as A, TextureFormat as F, TextureSampleType as S,
        TextureViewDimension as D,
    };
    fn check(source: &'static [u8], class: u8, good: B, bad: Vec<B>) {
        let desc = ShaderDesc {
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
        let map = [(0, 0, 0, class, 1), (0, 1, 1, 2, 1), (0, 2, 2, 1, 1)];
        let out = B::StorageBuffer {
            dynamic_offset: false,
            min_size: 4,
            access: A::ReadWrite,
        };
        let compile = |ty: B| {
            super::compile_variant_with_typed_map(
                &desc,
                &desc.entries[0],
                &[],
                &map,
                &[
                    (0, 0, ty),
                    (0, 1, out.clone()),
                    (
                        0,
                        2,
                        B::SampledTexture {
                            sample: S::Depth,
                            dimension: D::D2,
                            multisampled: false,
                        },
                    ),
                ],
                &[],
                &[],
                None,
            )
        };
        assert!(compile(good).is_ok());
        for ty in bad {
            assert!(matches!(
                compile(ty),
                Err(fluxel_rhi_core::api::Error::Shader(_))
            ));
        }
    }
    let sampled = |sample, dimension, multisampled| B::SampledTexture {
        sample,
        dimension,
        multisampled,
    };
    check(b"@group(0) @binding(0) var img: texture_3d<u32>; @group(0) @binding(1) var<storage,read_write> o: array<u32>; @compute @workgroup_size(1) fn main() { o[0] = textureLoad(img, vec3<i32>(0), 0).x; }", 1, sampled(S::Uint,D::D3,false), vec![sampled(S::Uint,D::D2,false),sampled(S::Float,D::D3,false),sampled(S::Uint,D::D3,true)]);
    let storage = |format, access| B::StorageTexture {
        format,
        dimension: D::D2,
        access,
    };
    check(b"@group(0) @binding(0) var img: texture_storage_2d<r32uint, read_write>; @group(0) @binding(1) var<storage,read_write> o: array<u32>; @compute @workgroup_size(1) fn main() { o[0] = textureLoad(img, vec2<i32>(0)).x; textureStore(img, vec2<i32>(0), vec4<u32>(1)); }", 2, storage(F::R32Uint,A::ReadWrite), vec![storage(F::R32Sint,A::ReadWrite),storage(F::R32Uint,A::Write)]);
    check(b"@group(0) @binding(0) var sam: sampler_comparison; @group(0) @binding(1) var<storage,read_write> o: array<u32>; @group(0) @binding(2) var tex: texture_depth_2d; @compute @workgroup_size(1) fn main() { o[0] = u32(textureSampleCompareLevel(tex,sam,vec2<f32>(0),0.5)); }", 3, B::Sampler { filtering: false, comparison: true }, vec![B::Sampler { filtering: false, comparison: false }]);
}

#[test]
fn ray_query_requires_declared_capability() {
    let source = b"@group(0) @binding(0) var scene: acceleration_structure; @compute @workgroup_size(1) fn main() { var q: ray_query; rayQueryInitialize(&q,scene,RayDesc(1u,255u,0.0,10.0,vec3<f32>(0.0),vec3<f32>(0.0,0.0,1.0))); while(rayQueryProceed(&q)) {} }";
    let mut desc = ShaderDesc {
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
    assert!(super::validate_source(&desc).is_err());
    desc.required_features
        .push(fluxel_rhi_core::api::SHADER_FEATURE_RAY_QUERY.into());
    super::validate_source(&desc).unwrap();
    super::compile(&desc).unwrap();
}
