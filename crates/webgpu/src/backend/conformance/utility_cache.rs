//! Real-output gate for fixed utility pipelines prepared at device opening.
use super::*;

fn monotonic_millis() -> f64 {
    let global: wasm_bindgen::JsValue = js_sys::global().into();
    let performance =
        super::super::js::property(&global, "performance").expect("browser performance clock");
    super::super::js::call0(&performance, "now")
        .expect("performance.now call")
        .as_f64()
        .expect("performance.now number")
}

/// Review3 cold-path diagnostic for the browser-visible `open_default` cost.
///
/// Each sample creates a fresh typed device and measures adapter selection,
/// device creation, capability discovery, and all four fixed utility pipeline
/// prewarms together.  Browser adapter/device caching is intentionally not
/// hidden, so the first sample is the coldest observable page-local open and
/// later samples reveal steady browser behavior.  This does not isolate one
/// individual native pipeline creation from browser bootstrap work.
#[wasm_bindgen_test(async)]
async fn review3_device_open_prewarm_timing() {
    for sample in 0..11 {
        let start = monotonic_millis();
        let device = WebGpuInstance::new()
            .expect("navigator.gpu")
            .open_default()
            .await
            .expect("typed WebGPU device");
        let elapsed_ms = monotonic_millis() - start;
        // SAFETY: `open_default` returns only after `prewarm_fixed`; the cache
        // is private immutable native state and is not mutated by this test.
        let prepared = unsafe { device.native_ref() }
            .0
            .utility_pipelines
            .borrow()
            .len();
        assert_eq!(
            prepared, 4,
            "device open must complete every fixed utility pipeline before recording"
        );
        console_log!(
            "REVIEW3_WEBGPU_DEVICE_OPEN sample={sample} elapsed_ms={elapsed_ms:.3} fixed_utility_pipelines={prepared}"
        );
    }
}

#[wasm_bindgen_test(async)]
async fn gpu_repeated_byte_patches_reuse_fixed_pipeline_and_preserve_bytes() {
    let d = open().await;
    // SAFETY: this test observes immutable backend-private cache objects after
    // the async device-open preparation has completed.
    let prepared = unsafe { d.native_ref() }.0.utility_pipelines.borrow().len();
    assert!(
        prepared >= 4,
        "all fixed utility compute pipelines are prepared before recording"
    );
    let target = buffer(
        &d,
        16,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    for count in [16, 128] {
        let mut encoder = d.encoder(d.lanes()[0].id).unwrap();
        encoder.write_buffer(&target, 0, &[0xa5; 16]).unwrap();
        for value in 0..count {
            encoder.write_buffer(&target, 1, &[value as u8]).unwrap();
        }
        let ticket = encoder.readback_buffer(&target, 0..16).unwrap();
        let point = submit(&d, encoder).await;
        // SAFETY: this point covers all patches and the readback copy.
        let actual = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        let mut expected = [0xa5; 16];
        expected[1] = (count - 1) as u8;
        assert_eq!(actual, expected);
    }
    assert_eq!(
        unsafe { d.native_ref() }.0.utility_pipelines.borrow().len(),
        prepared,
        "recording fixed lowering kernels must not synchronously create pipelines"
    );
}
