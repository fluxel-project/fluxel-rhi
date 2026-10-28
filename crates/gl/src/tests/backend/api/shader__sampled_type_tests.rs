use super::*;
#[test]
fn linked_sampler_type_must_match_binding_shape_and_comparison() {
    let texture = GlSampledBindingType::Texture {
        dimension: GlSampledDimension::D2Array,
        kind: GlSampledKind::Depth,
        multisampled: false,
    };
    assert!(texture.matches_uniform(0x8DC4));
    assert!(!texture.matches_uniform(0x8B62));
    assert!(!texture.matches_uniform(0x8DCF));
    assert!(GlSampledBindingType::Sampler { comparison: true }.matches_uniform(0x8DC4));
    assert!(!GlSampledBindingType::Sampler { comparison: false }.matches_uniform(0x8DC4));
}

#[test]
fn linked_multisample_array_sampler_type_must_match_dimension_and_sample_kind() {
    for (uniform, kind) in [
        (0x910B, GlSampledKind::Float),
        (0x910C, GlSampledKind::Sint),
        (0x910D, GlSampledKind::Uint),
    ] {
        assert!(
            GlSampledBindingType::Texture {
                dimension: GlSampledDimension::D2Array,
                kind,
                multisampled: true,
            }
            .matches_uniform(uniform)
        );
    }
    assert!(
        !GlSampledBindingType::Texture {
            dimension: GlSampledDimension::D2,
            kind: GlSampledKind::Float,
            multisampled: true,
        }
        .matches_uniform(0x910B)
    );
    assert!(
        !GlSampledBindingType::Texture {
            dimension: GlSampledDimension::D2Array,
            kind: GlSampledKind::Float,
            multisampled: false,
        }
        .matches_uniform(0x910B)
    );
    assert!(!GlSampledBindingType::Sampler { comparison: false }.matches_uniform(0x910B));
}
