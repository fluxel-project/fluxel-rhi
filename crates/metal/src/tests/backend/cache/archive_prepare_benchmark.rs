//! Unit tests for the corresponding implementation module.

use super::*;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineDescriptor, MTLDevice as _, MTLLibrary as _};
use std::{future::Future, task::Poll, time::Instant};

fn block_on<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn percentile(sorted: &[u128], percentile: usize) -> u128 {
    let index = (sorted.len() - 1) * percentile / 100;
    sorted[index]
}

/// Measures only archive Prepare and its uncontended lease setup.
/// Shader compilation and native PSO compilation are outside each sample.
#[test]
#[ignore = "requires Apple Metal hardware; reports archive Prepare P50/P95/P99"]
fn metal_archive_prepare_latency_percentiles() {
    let device = block_on(crate::MetalInstance::new().unwrap().open_default()).unwrap();
    // SAFETY: the typed Metal device remains alive on this test's owner
    // thread for library/function construction and every archive call.
    let native = unsafe { device.native_ref() };
    let source = NSString::from_str(
        "#include <metal_stdlib>\nusing namespace metal;\nkernel void archive_prepare_probe(device uint* out [[buffer(0)]], uint id [[thread_position_in_grid]]) { out[id] = id; }",
    );
    let library = native
        ._state
        ._device
        .newLibraryWithSource_options_error(&source, None)
        .expect("compile diagnostic Metal library outside timed samples");
    let function_name = NSString::from_str("archive_prepare_probe");
    let function = library
        .newFunctionWithName(&function_name)
        .expect("resolve diagnostic kernel outside timed samples");
    let descriptor = MTLComputePipelineDescriptor::new();
    descriptor.setComputeFunction(Some(&function));

    let mut samples = Vec::with_capacity(51);
    for _ in 0..51 {
        let cache = match Metal::create_cache(native, None) {
            Ok(cache) => cache,
            Err(Error::Unsupported(reason)) => {
                eprintln!("SKIP METAL_ARCHIVE_PREPARE unsupported={reason}");
                return;
            }
            Err(error) => panic!("create Metal archive for timing sample: {error:?}"),
        };
        descriptor.setBinaryArchives(None);
        let start = Instant::now();
        let prepared = block_on(cache.prepare_compute(&descriptor))
            .expect("add diagnostic kernel to Metal archive");
        samples.push(start.elapsed().as_nanos());
        std::hint::black_box(prepared);
    }
    samples.sort_unstable();
    eprintln!(
        "METAL_ARCHIVE_PREPARE kind=compute samples={} p50_ns={} p95_ns={} p99_ns={}",
        samples.len(),
        percentile(&samples, 50),
        percentile(&samples, 95),
        percentile(&samples, 99),
    );
}
