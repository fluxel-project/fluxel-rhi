//! Reports queried sample counts and exercises each supported texture allocation.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
#[path = "support/mod.rs"]
mod support;
use support::block;
#[test]
fn queried_samples_create_on_device() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let mut desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [8, 8, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    };
    let counts = match device.texture_support(&desc) {
        Support::Supported(support) => support.sample_counts,
        Support::Unsupported(reason) => panic!("format unavailable: {reason}"),
    };
    println!("RGBA8 queried sample counts: {counts:?}");
    for count in counts {
        desc.samples = count;
        device.create_texture(&desc).unwrap();
    }
}
