//! Unit tests for the corresponding implementation module.

use super::valid;
use fluxel_rhi_core::api::{AddressMode, FilterMode, SamplerDesc};

fn desc() -> SamplerDesc {
    SamplerDesc {
        label: None,
        address: [AddressMode::ClampToEdge; 3],
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mip_filter: FilterMode::Linear,
        lod: 0.0..16.0,
        anisotropy: 16,
        compare: None,
        border: None,
    }
}

#[test]
fn rejects_sampler_states_outside_the_native_subset() {
    assert!(valid(&desc()));
    let mut invalid = desc();
    invalid.address[0] = AddressMode::ClampToBorder;
    assert!(!valid(&invalid));
    invalid.border = Some(fluxel_rhi_core::api::BorderColor::OpaqueBlack);
    assert!(valid(&invalid));
    let mut invalid = desc();
    invalid.anisotropy = 17;
    assert!(!valid(&invalid));
    let mut invalid = desc();
    invalid.lod = 2.0..1.0;
    assert!(!valid(&invalid));
}
