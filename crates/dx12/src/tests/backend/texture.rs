//! Texture descriptor and format validation tests.
use super::*;

#[test]
fn bc_subsets_reject_non_dxgi_and_invalid_layouts() {
    let mut desc = api::TextureDesc {
        label: None,
        dimension: api::TextureDimension::D2,
        extent: [8, 8, 1],
        array_layers: 1,
        format: api::TextureFormat::Bc1RgbaUnorm,
        mip_levels: 1,
        samples: 1,
        usage: api::TextureUsage::TEXTURE_BINDING | api::TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    assert!(valid(&desc).is_ok());
    desc.extent[0] = 7;
    assert!(valid(&desc).is_ok()); // Device feature query decides nonaligned top levels.
    desc.extent[0] = 8;
    desc.mip_levels = 2;
    assert!(valid(&desc).is_ok());
    desc.mip_levels = 1;
    desc.format = api::TextureFormat::Etc2Rgb8Unorm;
    assert!(valid(&desc).is_err());
    desc.format = api::TextureFormat::Astc {
        block: [4, 4],
        srgb: false,
        hdr: false,
    };
    assert!(valid(&desc).is_err());
}

#[test]
fn dsv_array_range_preserves_the_selected_layers() {
    let view = api::TextureViewDesc {
        format: api::TextureFormat::Depth32Float,
        dimension: api::TextureViewDimension::D2Array,
        aspects: api::TextureAspects::DEPTH,
        mips: 0..1,
        layers: 3..5,
    };
    assert_eq!(dsv_layers(&view), Ok(DsvLayers { start: 3, count: 2 }));
}

#[test]
fn dsv_array_range_rejects_an_empty_range() {
    let view = api::TextureViewDesc {
        format: api::TextureFormat::Depth32Float,
        dimension: api::TextureViewDimension::D2Array,
        aspects: api::TextureAspects::DEPTH,
        mips: 0..1,
        layers: 2..2,
    };
    assert!(matches!(dsv_layers(&view), Err(Error::InvalidRange)));
}

#[test]
fn array_depth_view_lowers_its_selected_layers() {
    let view = api::TextureViewDesc {
        format: api::TextureFormat::Depth32Float,
        dimension: api::TextureViewDimension::D2Array,
        aspects: api::TextureAspects::DEPTH,
        mips: 0..1,
        layers: 3..5,
    };
    let desc = depth_dsv_desc(&view, 1, true).unwrap();
    assert_eq!(desc.ViewDimension, D3D12_DSV_DIMENSION_TEXTURE2DARRAY);
    // SAFETY: `depth_dsv_desc` initialized the union's Texture2DArray arm
    // for the asserted view dimension immediately above.
    let array = unsafe { desc.Anonymous.Texture2DArray };
    assert_eq!(array.FirstArraySlice, 3);
    assert_eq!(array.ArraySize, 2);

    let ms_desc = depth_dsv_desc(&view, 4, true).unwrap();
    assert_eq!(ms_desc.ViewDimension, D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY);
    // SAFETY: the asserted dimension selects the initialized union arm.
    let ms_array = unsafe { ms_desc.Anonymous.Texture2DMSArray };
    assert_eq!(ms_array.FirstArraySlice, 3);
    assert_eq!(ms_array.ArraySize, 2);
}

#[test]
fn depth_view_selects_nonzero_mip() {
    let view = api::TextureViewDesc {
        format: api::TextureFormat::Depth32Float,
        dimension: api::TextureViewDimension::D2Array,
        aspects: api::TextureAspects::DEPTH,
        mips: 2..3,
        layers: 1..2,
    };
    let desc = depth_dsv_desc(&view, 1, true).unwrap();
    // SAFETY: the D2Array view selects the initialized Texture2DArray arm.
    assert_eq!(unsafe { desc.Anonymous.Texture2DArray }.MipSlice, 2);
    assert!(depth_dsv_desc(&view, 4, true).is_err());
}
