//! Real-output gate for fixed utility pipelines prepared at device opening.
use super::*;

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
