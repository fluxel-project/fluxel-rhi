//! Real-output gate and diagnostic for repeated byte-patch pipeline reuse.
use super::*;

fn now() -> f64 {
    let performance = super::super::js::property(&js_sys::global().into(), "performance").unwrap();
    super::super::js::call0(&performance, "now")
        .unwrap()
        .as_f64()
        .unwrap()
}

#[wasm_bindgen_test(async)]
async fn gpu_repeated_byte_patches_reuse_fixed_pipeline_and_preserve_bytes() {
    let d = open().await;
    let target = buffer(
        &d,
        16,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    // This is a diagnostic cache-on/off comparison of the same fixed kernel,
    // not an old-revision benchmark or a renderer frame-rate claim.
    for count in [16, 128] {
        for round in 0..3 {
            for invalidate in [round % 2 == 0, round % 2 != 0] {
                let mut encoder = d.encoder(d.lanes()[0].id).unwrap();
                encoder.write_buffer(&target, 0, &[0xa5; 16]).unwrap();
                let start = now();
                for value in 0..count {
                    if invalidate {
                        // SAFETY: immutable native pipeline references already
                        // recorded in this encoder remain in its keep_alive.
                        unsafe { d.native_ref() }
                            .0
                            .utility_pipelines
                            .borrow_mut()
                            .remove("word-patch");
                    }
                    encoder.write_buffer(&target, 1, &[value as u8]).unwrap();
                }
                let elapsed = now() - start;
                let ticket = encoder.readback_buffer(&target, 0..16).unwrap();
                let point = submit(&d, encoder).await;
                // SAFETY: this point covers all patches and the readback copy.
                let actual = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
                let mut expected = [0xa5; 16];
                expected[1] = (count - 1) as u8;
                assert_eq!(actual, expected);
                assert!(
                    unsafe { d.native_ref() }
                        .0
                        .utility_pipelines
                        .borrow()
                        .contains_key("word-patch")
                );
                console_log!(
                    "PATCH_DIAGNOSTIC count={count} round={round} invalidate={invalidate} recording_ms={elapsed:.3}"
                );
            }
        }
    }
}
