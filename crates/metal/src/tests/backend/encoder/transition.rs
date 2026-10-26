//! Unit tests for the corresponding implementation module.

use super::{texture_aspects, validate_buffer_range, validate_texture_range};
use fluxel_rhi_core::api::{
    TextureAspects, TextureDesc, TextureDimension, TextureFormat, TextureUsage,
};

fn color() -> TextureDesc {
    TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [16, 16, 1],
        array_layers: 2,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 3,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![TextureFormat::Rgba8Unorm],
    }
}

#[test]
fn buffer_range_is_checked_against_the_actual_allocation() {
    assert!(validate_buffer_range(64, &(0..64)).is_ok());
    assert!(validate_buffer_range(64, &(100..200)).is_err());
    assert!(validate_buffer_range(64, &(0..65)).is_err());
    assert!(validate_buffer_range(64, &(32..32)).is_err());
    assert!(validate_buffer_range(64, &(32..16)).is_err());
}

#[test]
fn texture_range_requires_nonempty_bounds_and_matching_aspects() {
    let desc = color();
    assert!(validate_texture_range(&desc, TextureAspects::COLOR, &(0..3), &(0..2)).is_ok());
    assert!(validate_texture_range(&desc, TextureAspects::DEPTH, &(0..1), &(0..1)).is_err());
    assert!(validate_texture_range(&desc, TextureAspects::COLOR, &(3..3), &(0..1)).is_err());
    assert!(validate_texture_range(&desc, TextureAspects::COLOR, &(0..1), &(2..3)).is_err());
}

#[test]
fn combined_depth_stencil_accepts_both_native_aspects() {
    assert_eq!(
        texture_aspects(TextureFormat::Depth24PlusStencil8),
        TextureAspects::DEPTH | TextureAspects::STENCIL
    );
}
