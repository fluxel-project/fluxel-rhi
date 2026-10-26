//! Real-device create/drop stress for ordinary DX12 default resource allocation.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, time::Instant};

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

#[test]
fn pooled_buffers_and_textures_survive_churn_and_final_owner_drop() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    for _ in 0..4 {
        let started = Instant::now();
        let buffers: Vec<_> = (0..1024)
            .map(|_| {
                device
                    .create_buffer(&BufferDesc {
                        label: None,
                        size: 4 * 1024,
                        usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                        memory: MemoryClass::Device,
                    })
                    .unwrap()
            })
            .collect();
        let buffer_create = started.elapsed();
        let started = Instant::now();
        let textures: Vec<_> = (0..256)
            .map(|_| {
                device
                    .create_texture(&TextureDesc {
                        label: None,
                        dimension: TextureDimension::D2,
                        extent: [16, 16, 1],
                        array_layers: 1,
                        format: TextureFormat::Rgba8Unorm,
                        mip_levels: 1,
                        samples: 1,
                        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
                        view_formats: Vec::new(),
                    })
                    .unwrap()
            })
            .collect();
        let texture_create = started.elapsed();

        let retained_buffer = buffers[17].clone();
        let retained_texture = textures[23].clone();
        let started = Instant::now();
        drop(buffers);
        drop(textures);
        drop(retained_buffer);
        drop(retained_texture);
        println!(
            "pool churn: 1024 buffers create={buffer_create:?}; 256 textures create={texture_create:?}; final-owner drop={:?}",
            started.elapsed()
        );
    }

    // Resources larger than the pool page stay on the committed fallback.
    let started = Instant::now();
    let large = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 32 * 1024 * 1024,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    drop(large);
    println!(
        "32 MiB committed fallback create/drop={:?}",
        started.elapsed()
    );
}
