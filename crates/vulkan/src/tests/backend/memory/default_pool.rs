//! Unit tests for the corresponding implementation module.

//! Range, budget, resource-churn, and real-adapter mapping checks.

use super::{coalesce, take_range};

#[test]
fn aligned_ranges_split_coalesce_and_reuse() {
    let mut split = vec![(0, 1024)];
    assert_eq!(take_range(&mut split, 128, 256), Some(0));
    assert_eq!(take_range(&mut split, 128, 256), Some(256));
    assert_eq!(take_range(&mut split, 128, 256), Some(512));
    assert_eq!(take_range(&mut split, 128, 256), Some(768));
    assert!(take_range(&mut split, 129, 1).is_none());

    let mut free = vec![(0, 256), (256, 256), (512, 256)];
    coalesce(&mut free);
    assert!(free.contains(&(0, 768)));
    assert_eq!(take_range(&mut free, 768, 256), Some(0));
}

#[test]
fn overflow_does_not_consume_a_range() {
    let mut free = vec![(u64::MAX - 7, 7)];
    assert_eq!(take_range(&mut free, 16, 16), None);
    assert_eq!(free, vec![(u64::MAX - 7, 7)]);
}

#[test]
fn budget_gate_accepts_exact_fit_and_rejects_overflow_or_excess() {
    assert!(super::within_budget(100, 200, 100));
    assert!(!super::within_budget(101, 200, 100));
    assert!(!super::within_budget(u64::MAX - 3, u64::MAX, 8));
    assert!(super::within_budget(0, 0, 16));
}

#[cfg(all(windows, feature = "hardware-tests"))]
#[test]
#[ignore = "allocation-churn diagnostic; run explicitly on a real Vulkan adapter"]
fn compare_pooled_and_committed_buffer_texture_churn() {
    use crate::backend::{Vulkan, VulkanDevice, VulkanInstance, memory::DefaultResourcePool};
    use fluxel_rhi_core::api::{
        BufferDesc, BufferUsage, DeviceId, DeviceRequest, InstanceId, MemoryClass, TextureDesc,
        TextureDimension, TextureFormat, TextureUsage,
    };
    use fluxel_rhi_core::backend::{InstanceBackend, ResourceBackend};
    use std::{
        future::Future,
        time::{Duration, Instant},
    };

    fn block<F: Future>(future: F) -> F::Output {
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        let mut future = std::pin::pin!(future);
        loop {
            if let std::task::Poll::Ready(result) = future.as_mut().poll(&mut context) {
                return result;
            }
            std::thread::yield_now();
        }
    }
    fn median(mut values: Vec<Duration>) -> Duration {
        values.sort_unstable();
        values[values.len() / 2]
    }
    fn churn(device: &VulkanDevice) {
        let buffer = BufferDesc {
            label: None,
            size: 4096,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        };
        let texture = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [32, 32, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
            view_formats: vec![],
        };
        let buffers = (0..1024)
            .map(|_| Vulkan::create_buffer(device, &buffer).unwrap())
            .collect::<Vec<_>>();
        let textures = (0..256)
            .map(|_| Vulkan::create_texture(device, &texture).unwrap())
            .collect::<Vec<_>>();
        drop((buffers, textures));
    }

    let instance = VulkanInstance::new().unwrap();
    let instance_id = InstanceId::allocate().unwrap();
    let adapter = Vulkan::enumerate(&instance, instance_id)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let request = DeviceRequest {
        adapter: adapter.id,
        required_features: Vec::new(),
    };
    let (device, _) = block(Vulkan::open(
        &instance,
        DeviceId::allocate().unwrap(),
        &request,
    ))
    .unwrap();
    // SAFETY: the physical handle and retained instance belong to this live device.
    let properties = unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_properties(device.0.physical)
    };
    let adapter_name =
        unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }.to_string_lossy();
    println!(
        "Vulkan adapter={adapter_name}; vendor={:#06x}; device={:#06x}; driver={:#010x}; API={:#010x}; page=16MiB",
        properties.vendor_id,
        properties.device_id,
        properties.driver_version,
        properties.api_version,
    );
    let pool: &DefaultResourcePool = &device.0.default_resource_pool;
    println!("VK_EXT_memory_budget enabled={}", pool.budget_enabled());
    if let Some(budgets) = pool.budget_snapshot() {
        for (heap, (usage, budget)) in budgets.into_iter().enumerate() {
            println!(
                "Vulkan heap {heap} process usage={}MiB budget={}MiB",
                usage / 1024 / 1024,
                budget / 1024 / 1024,
            );
        }
    }
    for enabled in [true, false] {
        pool.set_enabled(enabled);
        pool.clear();
        let mut samples = Vec::with_capacity(5);
        for _ in 0..5 {
            let start = Instant::now();
            churn(&device);
            samples.push(start.elapsed());
        }
        println!(
            "{} 1024 buffers + 256 textures create/drop median={:?}; retained pool pages={}",
            if enabled { "pooled" } else { "committed" },
            median(samples),
            pool.page_count(),
        );
    }
    pool.set_enabled(true);
}
