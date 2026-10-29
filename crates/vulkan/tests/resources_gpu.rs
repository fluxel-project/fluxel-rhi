//! Hardware checks for Vulkan resource creation and reported format support.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};
struct WakeThread(std::thread::Thread);
impl Wake for WakeThread {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
fn block_on<F: Future>(f: F) -> F::Output {
    let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park(),
        }
    }
}
#[test]
fn native_resource_variants() {
    let device = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    for (dimension, extent, layers, view) in [
        (
            TextureDimension::D1,
            [16, 1, 1],
            1,
            TextureViewDimension::D1,
        ),
        (
            TextureDimension::D2,
            [16, 16, 1],
            6,
            TextureViewDimension::Cube,
        ),
        (
            TextureDimension::D3,
            [16, 8, 4],
            1,
            TextureViewDimension::D3,
        ),
    ] {
        let desc = TextureDesc {
            label: None,
            dimension,
            extent,
            array_layers: layers,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING,
            view_formats: vec![TextureFormat::Rgba8UnormSrgb],
        };
        assert!(matches!(
            device.texture_support(&desc),
            Support::Supported(_)
        ));
        let texture = device.create_texture(&desc).unwrap();
        device
            .create_view(
                &texture,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8UnormSrgb,
                    dimension: view,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..layers,
                },
            )
            .unwrap();
    }
    let large_uniform = BufferDesc {
        label: None,
        size: 1024 * 1024,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        memory: MemoryClass::Device,
    };
    assert!(matches!(
        device.buffer_support(&large_uniform),
        Support::Supported(_)
    ));
    device.create_buffer(&large_uniform).unwrap();
    let srgb_storage = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [16, 16, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8UnormSrgb,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::TEXTURE_BINDING | TextureUsage::STORAGE_BINDING,
        view_formats: vec![TextureFormat::Rgba8Unorm],
    };
    if matches!(device.texture_support(&srgb_storage), Support::Supported(_)) {
        let texture = device.create_texture(&srgb_storage).unwrap();
        device
            .create_view(
                &texture,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
            )
            .unwrap();
        eprintln!("mutable sRGB image with linear storage view supported");
    }
    let sampler = SamplerDesc {
        label: None,
        address: [AddressMode::ClampToZero; 3],
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mip_filter: FilterMode::Nearest,
        lod: 0.0..1.0,
        anisotropy: 1,
        compare: Some(CompareFunction::LessEqual),
        border: None,
    };
    device.create_sampler(&sampler).unwrap();
    for format in [
        TextureFormat::R8Unorm,
        TextureFormat::Rgba16Float,
        TextureFormat::R32Uint,
        TextureFormat::Depth16Unorm,
        TextureFormat::Depth24Plus,
        TextureFormat::Depth24PlusStencil8,
        TextureFormat::Bc1RgbaUnorm,
        TextureFormat::Etc2Rgb8Unorm,
        TextureFormat::Astc {
            block: [4, 4],
            srgb: false,
            hdr: false,
        },
    ] {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        };
        if matches!(device.texture_support(&desc), Support::Supported(_)) {
            device.create_texture(&desc).unwrap();
        }
    }
}

#[test]
fn memory_observation_tracks_default_pool_buffer_lifetime() {
    let device = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    let observation = device
        .memory_observation()
        .expect("Vulkan exposes memory observation");
    let before = observation.snapshot().unwrap();
    let before_pool = before.allocator.expect("Vulkan default resource pool");
    assert_eq!(
        before_pool.source,
        AllocatorMemorySource::VulkanDefaultResourcePool
    );
    assert_eq!(
        before_pool.reserved_bytes,
        before_pool.live_range_bytes + before_pool.free_bytes
    );
    assert!(!before.native_reports.is_empty());
    for report in &before.native_reports {
        assert_eq!(report.source, MemoryReportSource::VulkanProcessHeapEstimate);
        let MemoryReportScope::VulkanHeap {
            heap_index,
            heap_size_bytes,
            ..
        } = report.scope
        else {
            panic!("Vulkan report must name a physical heap");
        };
        assert!(heap_size_bytes > 0, "heap {heap_index} has a nonzero size");
        // `None` represents an unavailable extension result. If available,
        // usage and budget stay driver estimates and are deliberately not
        // ordered against each other here.
        if report.usage_bytes.is_none() {
            assert!(report.budget_bytes.is_none());
        }
    }

    let buffer = device
        .create_buffer(&BufferDesc {
            label: Some("Vulkan memory observation buffer".into()),
            size: 4096,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let during_pool = observation.snapshot().unwrap().allocator.unwrap();
    assert!(during_pool.live_range_bytes > before_pool.live_range_bytes);
    assert!(during_pool.live_allocation_count > before_pool.live_allocation_count);
    assert_eq!(
        during_pool.reserved_bytes,
        during_pool.live_range_bytes + during_pool.free_bytes
    );

    drop(buffer);
    let after_pool = observation.snapshot().unwrap().allocator.unwrap();
    assert_eq!(after_pool.live_range_bytes, before_pool.live_range_bytes);
    assert_eq!(
        after_pool.live_allocation_count,
        before_pool.live_allocation_count
    );
    assert!(after_pool.free_bytes >= before_pool.free_bytes);
    assert!(after_pool.peak_live_range_bytes >= during_pool.live_range_bytes);
}
