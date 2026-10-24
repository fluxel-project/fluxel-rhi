//! Native immutable-object reuse across repeated compute blits.
use super::*;
use fluxel_rhi_core::api::*;
use std::{
    future::Future,
    task::{Context, Poll, Waker},
};

#[test]
fn repeated_variants_reuse_the_same_native_root_and_pipeline() {
    let instance = crate::Dx12Instance::new().unwrap().instance().unwrap();
    let adapter = instance.enumerate().unwrap().remove(0).id;
    let mut open = std::pin::pin!(instance.open(DeviceRequest {
        adapter,
        required_features: vec![],
    }));
    let Poll::Ready(Ok(device)) = open.as_mut().poll(&mut Context::from_waker(Waker::noop()))
    else {
        panic!("synchronous D3D12 device opening failed");
    };
    // SAFETY: this immutable reference remains borrowed from the live device.
    let native = unsafe { device.native_ref() };
    let cache = &native.0.blit_compute_cache;
    for variant in [
        BlitComputeShader::Typed {
            source_view: 0,
            target_view: 1,
            kind: 0,
        },
        BlitComputeShader::Typed {
            source_view: 3,
            target_view: 2,
            kind: 1,
        },
        BlitComputeShader::Buffer {
            source_view: 0,
            kind: 0,
        },
    ] {
        let first = cache.get_or_create(&native.0.native, variant).unwrap();
        for _ in 0..8 {
            let repeated = cache.get_or_create(&native.0.native, variant).unwrap();
            assert!(Arc::ptr_eq(&first, &repeated));
        }
    }
    assert_eq!(cache.0.lock().unwrap().len(), 3);
}
