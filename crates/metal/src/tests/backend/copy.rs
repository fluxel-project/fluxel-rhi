//! Unit tests for the corresponding implementation module.

use super::{validate_copy_layout, validate_region_fields};
use fluxel_rhi_core::api::{
    TexelCopyBufferLayout, TextureAspects, TextureDesc, TextureDimension, TextureFormat,
    TextureUsage,
};

fn desc() -> TextureDesc {
    TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [64, 32, 1],
        array_layers: 2,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 7,
        samples: 1,
        usage: TextureUsage(TextureUsage::COPY_SRC.0 | TextureUsage::COPY_DST.0),
        view_formats: vec![TextureFormat::Rgba8Unorm],
    }
}

#[test]
fn texture_copy_region_accepts_valid_mip_and_array_slice() {
    assert!(
        validate_region_fields(&desc(), 2, [3, 2, 1], [8, 4, 1], TextureAspects::COLOR,).is_ok()
    );
}

#[test]
fn texture_copy_region_rejects_out_of_bounds_and_wrong_aspect() {
    let desc = desc();
    assert!(
        validate_region_fields(&desc, 0, [63, 0, 0], [2, 1, 1], TextureAspects::COLOR,).is_err()
    );
    assert!(
        validate_region_fields(&desc, 0, [0, 0, 2], [1, 1, 1], TextureAspects::COLOR,).is_err()
    );
    assert!(
        validate_region_fields(&desc, 0, [0, 0, 0], [1, 1, 1], TextureAspects::DEPTH,).is_err()
    );
}

#[test]
fn buffer_texture_layout_checks_row_pitch_offset_and_final_row_bounds() {
    let valid = api_layout(0, 4, 2);
    assert!(validate_copy_layout(valid, 4, 2, 9).is_ok());
    assert!(validate_copy_layout(valid, 5, 2, 9).is_err());
    assert!(validate_copy_layout(api_layout(8, 1, 4), 4, 2, 11).is_err());
    assert!(validate_copy_layout(api_layout(u64::MAX, 4, 2), 4, 2, u64::MAX).is_err());
}

#[test]
fn resolve_capability_distinguishes_integer_and_family_limits() {
    use super::color_resolve_supported as supports;
    assert!(!supports(TextureFormat::Rgba8Uint, true, true, true));
    assert!(supports(TextureFormat::Rgba8Unorm, false, false, false));
    assert!(!supports(TextureFormat::R32Float, true, false, false));
    assert!(supports(TextureFormat::R32Float, true, true, false));
    assert!(!supports(TextureFormat::Rg16Unorm, false, false, false));
    assert!(supports(TextureFormat::Rg16Unorm, true, false, false));
    assert!(supports(TextureFormat::Rgb9e5Ufloat, true, false, true));
    assert!(!supports(TextureFormat::Rgb9e5Ufloat, false, false, true));
}

#[test]
fn validates_compressed_blocks_and_multilayer_footprints() {
    let mut texture = desc();
    texture.format = TextureFormat::Bc1RgbaUnorm;
    assert!(
        validate_region_fields(&texture, 0, [4, 4, 0], [8, 8, 2], TextureAspects::COLOR).is_ok()
    );
    assert!(
        validate_region_fields(&texture, 0, [1, 0, 0], [4, 4, 1], TextureAspects::COLOR).is_err()
    );
    let layout = api_layout(0, 16, 3);
    assert!(super::validate_copy_images(layout, 16, 2, 2, 80).is_ok());
    assert!(super::validate_copy_images(layout, 16, 2, 2, 79).is_err());
    texture.dimension = TextureDimension::D3;
    texture.extent[2] = 8;
    texture.array_layers = 1;
    assert!(
        validate_region_fields(&texture, 1, [0, 0, 1], [4, 4, 3], TextureAspects::COLOR).is_ok()
    );
    assert!(
        validate_region_fields(&texture, 1, [0, 0, 2], [4, 4, 3], TextureAspects::COLOR).is_err()
    );
}

fn api_layout(offset: u64, bytes_per_row: u32, rows_per_image: u32) -> TexelCopyBufferLayout {
    TexelCopyBufferLayout {
        offset,
        bytes_per_row,
        rows_per_image,
    }
}
