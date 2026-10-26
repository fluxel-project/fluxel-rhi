//! Unit tests for the corresponding implementation module.

use super::{MAX_2D_EXTENT, MAX_ARRAY_LAYERS, max_mip_levels, valid_desc};
use fluxel_rhi_core::api::{TextureDesc, TextureDimension, TextureFormat, TextureUsage};

fn desc() -> TextureDesc {
    TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [512, 256, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 10,
        samples: 1,
        usage: TextureUsage::TEXTURE_BINDING,
        view_formats: vec![TextureFormat::Rgba8Unorm],
    }
}

#[test]
fn texture_descriptor_bounds_match_the_advertised_subset() {
    let base = desc();
    assert!(valid_desc(&base));
    assert_eq!(max_mip_levels(base.extent), 10);
    let mut invalid = base.clone();
    invalid.extent[0] = MAX_2D_EXTENT + 1;
    assert!(!valid_desc(&invalid));
    let mut invalid = base.clone();
    invalid.array_layers = MAX_ARRAY_LAYERS + 1;
    assert!(!valid_desc(&invalid));
    let mut invalid = base;
    invalid.usage = TextureUsage::STORAGE_BINDING;
    assert!(valid_desc(&invalid));
    invalid.format = TextureFormat::Bgra8Unorm;
    invalid.view_formats.clear();
    assert!(valid_desc(&invalid));
    let mut multisampled = desc();
    multisampled.samples = 4;
    multisampled.mip_levels = 1;
    multisampled.usage = TextureUsage::RENDER_ATTACHMENT;
    assert!(valid_desc(&multisampled));
    multisampled.mip_levels = 2;
    assert!(!valid_desc(&multisampled));
    multisampled.mip_levels = 1;
    multisampled.usage = TextureUsage(multisampled.usage.0 | TextureUsage::TEXTURE_BINDING.0);
    assert!(valid_desc(&multisampled));
    let mut depth_stencil = desc();
    depth_stencil.format = TextureFormat::Depth32FloatStencil8;
    depth_stencil.mip_levels = 1;
    depth_stencil.usage = TextureUsage::RENDER_ATTACHMENT;
    depth_stencil.view_formats = vec![TextureFormat::Depth32FloatStencil8];
    assert!(valid_desc(&depth_stencil));
    depth_stencil.usage =
        TextureUsage(TextureUsage::RENDER_ATTACHMENT.0 | TextureUsage::TEXTURE_BINDING.0);
    assert!(valid_desc(&depth_stencil));
    let mut invalid = desc();
    invalid.usage = TextureUsage(1 << 63);
    assert!(!valid_desc(&invalid));
}

#[test]
fn texture_dimensions_and_compatible_views_are_checked() {
    let mut d1 = desc();
    d1.dimension = TextureDimension::D1;
    d1.extent = [512, 1, 1];
    assert!(valid_desc(&d1));
    d1.extent[1] = 2;
    assert!(!valid_desc(&d1));
    let mut d3 = desc();
    d3.dimension = TextureDimension::D3;
    d3.extent = [64, 32, 16];
    d3.mip_levels = 7;
    assert!(valid_desc(&d3));
    d3.array_layers = 2;
    assert!(!valid_desc(&d3));
    let mut view = desc();
    view.view_formats.push(TextureFormat::Rgba8UnormSrgb);
    assert!(valid_desc(&view));
    view.view_formats.push(TextureFormat::Bgra8Unorm);
    assert!(!valid_desc(&view));
}
