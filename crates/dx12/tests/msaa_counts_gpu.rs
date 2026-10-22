//! Reports queried sample counts and exercises each supported texture allocation.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::future::Future;
fn block<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}
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
