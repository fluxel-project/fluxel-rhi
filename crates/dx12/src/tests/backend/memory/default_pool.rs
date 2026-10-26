//! Range, lease-lifetime, and real-adapter churn checks for the default pool.

#[cfg(all(windows, feature = "hardware-tests"))]
use super::MAX_EMPTY_PAGES_PER_CLASS;
use super::{free_range, take_range};

#[test]
fn aligned_ranges_split_coalesce_and_reuse() {
    let mut free = vec![(0, 1024)];
    assert_eq!(take_range(&mut free, 100, 256), Some(0));
    assert_eq!(take_range(&mut free, 100, 256), Some(256));
    assert_eq!(take_range(&mut free, 100, 256), Some(512));
    assert_eq!(take_range(&mut free, 100, 256), Some(768));
    assert_eq!(take_range(&mut free, 200, 1), None);

    let mut free = Vec::new();
    free_range(&mut free, 256, 100);
    free_range(&mut free, 0, 256);
    free_range(&mut free, 356, 156);
    assert_eq!(free, vec![(0, 512)]);
    assert_eq!(take_range(&mut free, 512, 256), Some(0));
}

#[test]
fn alignment_overflow_fails_without_mutating_free_list() {
    let mut free = vec![(u64::MAX - 3, 3)];
    assert_eq!(take_range(&mut free, 8, 8), None);
    assert_eq!(free, vec![(u64::MAX - 3, 3)]);
}

#[test]
fn budget_gate_accepts_exact_fit_and_rejects_overflow_or_excess() {
    assert!(super::within_budget(100, 200, 100));
    assert!(!super::within_budget(101, 200, 100));
    assert!(!super::within_budget(u64::MAX - 3, u64::MAX, 8));
    assert!(!super::within_budget(0, 0, 1));
}

#[cfg(all(windows, feature = "hardware-tests"))]
#[test]
#[ignore = "CPU stage diagnostic; run explicitly on a real DX12 adapter"]
fn profile_default_resource_creation_stages() {
    use crate::backend::{Dx12, Dx12Instance, memory};
    use fluxel_rhi_core::api::{self};
    use fluxel_rhi_core::backend::ResourceBackend;
    use std::{
        future::Future,
        hint::black_box,
        time::{Duration, Instant},
    };

    fn block<F: Future>(future: F) -> F::Output {
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                std::task::Poll::Ready(value) => return value,
                std::task::Poll::Pending => std::thread::sleep(Duration::from_millis(1)),
            }
        }
    }

    fn median_ms(mut operation: impl FnMut() -> Duration) -> f64 {
        for _ in 0..2 {
            black_box(operation());
        }
        let mut samples = (0..7)
            .map(|_| operation().as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
        samples.sort_by(f64::total_cmp);
        samples[3]
    }

    let public_device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    // SAFETY: the public device owns this native device throughout the fixture.
    let device = unsafe { public_device.native_ref() };
    let desc = api::BufferDesc {
        label: None,
        size: 4096,
        usage: api::BufferUsage::UNIFORM | api::BufferUsage::COPY_DST,
        memory: api::MemoryClass::Device,
    };
    let (native_desc, _, initial_state) = memory::buffer_desc(&desc).unwrap();
    let class_bits = memory::buffer_class_bits(device, desc.memory);
    let class = (0..64).find(|bit| class_bits & (1u64 << bit) != 0).unwrap();
    let requirements = memory::requirements(device, &native_desc, class_bits).unwrap();

    let support_ms = median_ms(|| {
        let started = Instant::now();
        for _ in 0..1024 {
            assert!(matches!(
                <Dx12 as ResourceBackend>::buffer_support(device, &desc),
                api::Support::Supported(_)
            ));
        }
        started.elapsed()
    });
    let requirements_ms = median_ms(|| {
        let started = Instant::now();
        for _ in 0..1024 {
            black_box(memory::requirements(device, &native_desc, class_bits).unwrap());
        }
        started.elapsed()
    });
    let pool_allocate_ms = median_ms(|| {
        let started = Instant::now();
        let leases = (0..1024)
            .map(|_| {
                device
                    .0
                    .default_resources
                    .allocate(class, requirements.size, requirements.alignment)
                    .unwrap()
                    .expect("pool allocation")
            })
            .collect::<Vec<_>>();
        black_box(&leases);
        let elapsed = started.elapsed();
        drop(leases);
        elapsed
    });
    let pool_release_ms = median_ms(|| {
        let leases = (0..1024)
            .map(|_| {
                device
                    .0
                    .default_resources
                    .allocate(class, requirements.size, requirements.alignment)
                    .unwrap()
                    .expect("pool allocation")
            })
            .collect::<Vec<_>>();
        let started = Instant::now();
        drop(leases);
        started.elapsed()
    });
    let placed_create_ms = median_ms(|| {
        let allocations = (0..1024)
            .map(|_| {
                device
                    .0
                    .default_resources
                    .allocate(class, requirements.size, requirements.alignment)
                    .unwrap()
                    .expect("pool allocation")
            })
            .collect::<Vec<_>>();
        let started = Instant::now();
        let mut resources: Vec<windows::Win32::Graphics::Direct3D12::ID3D12Resource> =
            Vec::with_capacity(allocations.len());
        for allocation in &allocations {
            let mut resource = None;
            // SAFETY: every pool lease satisfies the native resource's queried size,
            // alignment and heap class, and each offset is exclusively leased.
            unsafe {
                device.0.native.CreatePlacedResource(
                    &allocation.heap,
                    allocation.offset,
                    &native_desc,
                    initial_state,
                    None,
                    &mut resource,
                )
            }
            .unwrap();
            resources.push(resource.unwrap());
        }
        let elapsed = started.elapsed();
        drop(resources);
        drop(allocations);
        elapsed
    });
    let full_create_drop_ms = median_ms(|| {
        let started = Instant::now();
        let buffers = (0..1024)
            .map(|_| <Dx12 as ResourceBackend>::create_buffer(device, &desc).unwrap())
            .collect::<Vec<_>>();
        drop(buffers);
        started.elapsed()
    });

    println!(
        "DX12 stage medians per 1024 buffers: support={support_ms:.4}ms allocation_info={requirements_ms:.4}ms pool_allocate={pool_allocate_ms:.4}ms pool_release={pool_release_ms:.4}ms CreatePlacedResource_only={placed_create_ms:.4}ms full_create_drop={full_create_drop_ms:.4}ms"
    );
}

#[cfg(all(windows, feature = "hardware-tests"))]
#[test]
fn native_resource_clone_keeps_lease_until_last_drop_then_reuses_range() {
    use crate::backend::{Dx12, Dx12Instance};
    use fluxel_rhi_core::api::{self};
    use fluxel_rhi_core::backend::ResourceBackend;
    use std::future::Future;

    fn block<F: Future>(future: F) -> F::Output {
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                std::task::Poll::Ready(value) => return value,
                std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
            }
        }
    }

    let public_device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    // SAFETY: the public device owns this native device for the test scope.
    let device = unsafe { public_device.native_ref() };
    let desc = api::BufferDesc {
        label: None,
        size: 4096,
        usage: api::BufferUsage::COPY_SRC | api::BufferUsage::COPY_DST,
        memory: api::MemoryClass::Device,
    };

    let first = <Dx12 as ResourceBackend>::create_buffer(device, &desc).unwrap();
    let lease = first.default_allocation.as_ref().expect("pooled buffer");
    let first_page = lease.0.page_id;
    let first_offset = lease.0.offset;
    let clones: Vec<_> = (0..16).map(|_| first.clone()).collect();
    drop(first);
    {
        let pool = device.0.default_resources.inner.lock().unwrap();
        let page = pool
            .pages
            .iter()
            .find(|page| page.id == first_page)
            .unwrap();
        assert_eq!(page.live, 1, "clone keeps its range leased");
    }
    let drops: Vec<_> = clones
        .into_iter()
        .map(|buffer| std::thread::spawn(move || drop(buffer)))
        .collect();
    for drop in drops {
        drop.join().unwrap();
    }
    {
        let pool = device.0.default_resources.inner.lock().unwrap();
        let page = pool
            .pages
            .iter()
            .find(|page| page.id == first_page)
            .unwrap();
        assert_eq!(page.live, 0, "final owner returns its range");
    }

    let reused = <Dx12 as ResourceBackend>::create_buffer(device, &desc).unwrap();
    let lease = reused.default_allocation.as_ref().expect("pooled buffer");
    assert_eq!(
        (lease.0.page_id, lease.0.offset),
        (first_page, first_offset)
    );
}

#[cfg(all(windows, feature = "hardware-tests"))]
#[test]
#[ignore = "allocation-churn diagnostic; run explicitly on a real DX12 adapter"]
fn compare_pooled_and_committed_buffer_churn() {
    use crate::backend::{Dx12, Dx12Instance, memory};
    use fluxel_rhi_core::api::{self};
    use fluxel_rhi_core::backend::ResourceBackend;
    use std::{future::Future, sync::Arc, time::Instant};
    use windows::Win32::Graphics::{
        Direct3D12::*,
        Dxgi::{
            DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
            DXGI_QUERY_VIDEO_MEMORY_INFO,
        },
    };

    fn block<F: Future>(future: F) -> F::Output {
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                std::task::Poll::Ready(value) => return value,
                std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
            }
        }
    }

    let public_device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    // SAFETY: the public device owns this native device for the test scope.
    let device = unsafe { public_device.native_ref() };
    let desc = api::BufferDesc {
        label: None,
        size: 4096,
        usage: api::BufferUsage::COPY_SRC | api::BufferUsage::COPY_DST,
        memory: api::MemoryClass::Device,
    };
    let texture_desc = api::TextureDesc {
        label: None,
        dimension: api::TextureDimension::D2,
        extent: [16, 16, 1],
        array_layers: 1,
        format: api::TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: api::TextureUsage::COPY_SRC | api::TextureUsage::COPY_DST,
        view_formats: Vec::new(),
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    };

    // Compare page scales with the same full backend creation path. The
    // committed baseline repeats support, descriptor and requirements work.
    let page_sizes = [4, 8, 16, 32, 64].map(|mib| mib * 1024 * 1024);
    let mut page_results = Vec::new();
    let mut texture_page_results = Vec::new();
    for page_bytes in page_sizes {
        {
            let mut pool = device.0.default_resources.inner.lock().unwrap();
            assert!(pool.pages.iter().all(|page| page.live == 0));
            pool.pages.clear();
            pool.page_bytes = page_bytes;
        }
        drop(<Dx12 as ResourceBackend>::create_buffer(device, &desc).unwrap());
        let mut samples = Vec::new();
        let mut peak_pages = 0usize;
        for _ in 0..5 {
            let started = Instant::now();
            let pooled: Vec<_> = (0..1024)
                .map(|_| <Dx12 as ResourceBackend>::create_buffer(device, &desc).unwrap())
                .collect();
            peak_pages =
                peak_pages.max(device.0.default_resources.inner.lock().unwrap().pages.len());
            drop(pooled);
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(f64::total_cmp);
        let pool = device.0.default_resources.inner.lock().unwrap();
        let retained = pool.pages.iter().filter(|page| page.live == 0).count();
        assert!(
            pool.pages
                .iter()
                .filter(|page| page.live == 0)
                .map(|page| page.class)
                .all(|class| pool
                    .pages
                    .iter()
                    .filter(|page| page.live == 0 && page.class == class)
                    .count()
                    <= MAX_EMPTY_PAGES_PER_CLASS)
        );
        page_results.push((page_bytes, samples[2], peak_pages, retained));
    }
    for page_bytes in page_sizes {
        {
            let mut pool = device.0.default_resources.inner.lock().unwrap();
            assert!(pool.pages.iter().all(|page| page.live == 0));
            pool.pages.clear();
            pool.page_bytes = page_bytes;
        }
        drop(<Dx12 as ResourceBackend>::create_texture(device, &texture_desc).unwrap());
        let mut samples = Vec::new();
        let mut peak_pages = 0usize;
        for _ in 0..5 {
            let started = Instant::now();
            let pooled: Vec<_> = (0..256)
                .map(|_| <Dx12 as ResourceBackend>::create_texture(device, &texture_desc).unwrap())
                .collect();
            peak_pages =
                peak_pages.max(device.0.default_resources.inner.lock().unwrap().pages.len());
            drop(pooled);
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(f64::total_cmp);
        let pool = device.0.default_resources.inner.lock().unwrap();
        let retained = pool.pages.iter().filter(|page| page.live == 0).count();
        assert!(
            pool.pages
                .iter()
                .filter(|page| page.live == 0)
                .map(|page| page.class)
                .all(|class| pool
                    .pages
                    .iter()
                    .filter(|page| page.live == 0 && page.class == class)
                    .count()
                    <= MAX_EMPTY_PAGES_PER_CLASS)
        );
        texture_page_results.push((page_bytes, samples[2], peak_pages, retained));
    }
    {
        let mut pool = device.0.default_resources.inner.lock().unwrap();
        pool.pages.clear();
        pool.page_bytes = super::PAGE_BYTES;
    }

    let mut committed_ms = Vec::new();
    let mut committed_texture_ms = Vec::new();
    for _ in 0..5 {
        let started = Instant::now();
        let mut committed: Vec<crate::backend::Dx12Buffer> = Vec::with_capacity(1024);
        for _ in 0..1024 {
            assert!(matches!(
                <Dx12 as ResourceBackend>::buffer_support(device, &desc),
                api::Support::Supported(_)
            ));
            let (native_desc, allocation_size, initial_state) = memory::buffer_desc(&desc).unwrap();
            let _ = memory::requirements(
                device,
                &native_desc,
                memory::buffer_class_bits(device, desc.memory),
            )
            .unwrap();
            let mut resource = None;
            // SAFETY: descriptors are initialized and the device returns
            // each committed resource as an owned COM interface.
            unsafe {
                device.0.native.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &native_desc,
                    initial_state,
                    None,
                    &mut resource,
                )
            }
            .unwrap();
            committed.push(crate::backend::Dx12Buffer {
                resource: resource.unwrap(),
                size: desc.size,
                allocation_size,
                mapped: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                usage: desc.usage,
                memory: desc.memory,
                indirect_signatures: device.0.indirect_signatures.clone(),
                backing_heap: None,
                default_allocation: None,
            });
        }
        drop(committed);
        committed_ms.push(started.elapsed().as_secs_f64() * 1000.0);

        let started = Instant::now();
        let mut committed: Vec<crate::backend::Dx12Texture> = Vec::with_capacity(256);
        for _ in 0..256 {
            let texture_native =
                crate::backend::texture::allocation_desc(device, &texture_desc).unwrap();
            let _ = memory::requirements(
                device,
                &texture_native.resource,
                memory::texture_class_bits(device, &texture_native.resource),
            )
            .unwrap();
            let mut resource = None;
            // SAFETY: the native descriptor and clear value are the same
            // values used by the backend's committed texture path.
            unsafe {
                device.0.native.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &texture_native.resource,
                    texture_native.initial_state,
                    texture_native.clear.as_ref().map(|value| value as *const _),
                    &mut resource,
                )
            }
            .unwrap();
            committed.push(crate::backend::Dx12Texture(Arc::new(
                crate::backend::texture::TextureState {
                    resource: resource.unwrap(),
                    default_allocation: None,
                    backing_heap: None,
                    desc: texture_desc.clone(),
                    sparse: None,
                },
            )));
        }
        drop(committed);
        committed_texture_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    committed_ms.sort_by(f64::total_cmp);
    committed_texture_ms.sort_by(f64::total_cmp);
    let pool = device.0.default_resources.inner.lock().unwrap();
    assert!(pool.pages.is_empty());
    drop(pool);
    for (page_bytes, pooled_ms, peak_pages, retained) in page_results {
        println!(
            "1,024 default buffers: page={}MiB pooled create+drop median={pooled_ms:.3}ms with at most {peak_pages} pages ({retained} retained empty); committed create+drop median={:.3}ms with 1,024 allocations",
            page_bytes / 1024 / 1024,
            committed_ms[2],
        );
    }
    for (page_bytes, pooled_ms, peak_pages, retained) in texture_page_results {
        println!(
            "256 default textures: page={}MiB pooled create+drop median={pooled_ms:.3}ms with at most {peak_pages} pages ({retained} retained empty); committed create+drop median={:.3}ms",
            page_bytes / 1024 / 1024,
            committed_texture_ms[2],
        );
    }
    let pool = device.0.default_resources.inner.lock().unwrap();
    if let Some(adapter) = &pool.budget_adapter {
        for (name, segment) in [
            ("local", DXGI_MEMORY_SEGMENT_GROUP_LOCAL),
            ("non-local", DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL),
        ] {
            let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            // SAFETY: retained adapter and initialized output remain valid
            // for the synchronous diagnostic query.
            match unsafe { adapter.QueryVideoMemoryInfo(0, segment, &mut info) } {
                Ok(()) if info.Budget > 0 => println!(
                    "DXGI {name} segment process usage={}MiB budget={}MiB; this is not pool-only resident memory",
                    info.CurrentUsage / 1024 / 1024,
                    info.Budget / 1024 / 1024,
                ),
                _ => println!("DXGI {name} segment budget query unavailable"),
            }
        }
    } else {
        println!("DXGI adapter budget interface unavailable");
    }
}
