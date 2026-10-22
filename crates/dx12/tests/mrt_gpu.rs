//! Real GPU readback for multiple raster color targets.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};

fn block_on<F: Future>(f: F) -> F::Output {
    let w = std::task::Waker::noop();
    let mut c = std::task::Context::from_waker(w);
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut c) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

#[path = "mrt_gpu/color.rs"]
mod color;
#[path = "mrt_gpu/depth.rs"]
mod depth;
#[path = "mrt_gpu/resolve.rs"]
mod resolve;
#[path = "mrt_gpu/secondary.rs"]
mod secondary;
#[path = "mrt_gpu/validation.rs"]
mod validation;
