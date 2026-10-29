//! Real-output gate for fixed utility pipelines prepared at device opening.
use super::*;

/// Device opening completes all fixed utility pipeline preparation.
#[wasm_bindgen_test(async)]
async fn device_open_prepares_all_fixed_utility_pipelines() {
    let device = WebGpuInstance::new()
        .expect("navigator.gpu")
        .open_default()
        .await
        .expect("typed WebGPU device");
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
