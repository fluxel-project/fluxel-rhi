use super::*;

#[test]
fn group_index_keeps_same_numbered_uniforms_distinct() {
    let entries = [GlBindingLayoutEntry {
        binding: 0,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
        slot: GlBindingSlot::UniformBuffer {
            dynamic_offset: false,
            min_size: 16,
        },
    }];
    let group_zero = lower_program_bindings(0, &entries).expect("group zero layout");
    let group_one = lower_program_bindings(1, &entries).expect("group one layout");

    assert_ne!(group_zero[0].location, group_one[0].location);
    assert_eq!(group_zero[0].names, ["binding_0"]);
    assert_eq!(group_one[0].names, ["group_1_binding_0"]);
}

#[test]
fn texture_load_layout_needs_no_sampler_binding() {
    let entries = [GlBindingLayoutEntry {
        binding: 0,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
        slot: GlBindingSlot::SampledTexture {
            dimension: api::TextureViewDimension::D2,
            sample: api::TextureSampleType::Float,
            multisampled: false,
        },
    }];
    let lowered = lower_program_bindings(0, &entries).expect("texture-only layout");
    assert_eq!(lowered.len(), 1);
    assert_eq!(lowered[0].pair, None);
    assert!(matches!(lowered[0].kind, GlShaderResourceKind::Texture));
}

#[test]
fn texture_load_layout_does_not_require_a_sampler_resource() {
    let entries = [GlBindingLayoutEntry {
        binding: 0,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
        slot: GlBindingSlot::SampledTexture {
            dimension: api::TextureViewDimension::D2,
            sample: api::TextureSampleType::Float,
            multisampled: false,
        },
    }];
    let lowered = lower_program_bindings(0, &entries).expect("textureLoad layout");

    assert_eq!(lowered.len(), 1);
    assert_eq!(lowered[0].kind, GlShaderResourceKind::Texture);
    assert_eq!(lowered[0].pair, None);
    assert_eq!(lowered[0].names, ["binding_0"]);
}
