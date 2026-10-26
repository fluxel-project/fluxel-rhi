//! Real-browser checks for encoder-local upload page ordering and reuse.

use super::*;
use crate::backend::command;

#[cfg(feature = "review3-upload-benchmark")]
fn monotonic_millis() -> f64 {
    let global: wasm_bindgen::JsValue = js_sys::global().into();
    let performance =
        super::super::js::property(&global, "performance").expect("browser performance clock");
    super::super::js::call0(&performance, "now")
        .expect("performance.now call")
        .as_f64()
        .expect("performance.now number")
}

#[cfg(feature = "review3-upload-benchmark")]
async fn record_and_complete_upload_workload(
    d: &Device<WebGpu>,
    target: &Buffer<WebGpu>,
    encoders: usize,
    writes_per_encoder: usize,
) -> (f64, f64) {
    let start = monotonic_millis();
    let mut recordings = Vec::with_capacity(encoders);
    for encoder_index in 0..encoders {
        let mut encoder = d.encoder(d.lanes()[0].id).unwrap();
        for write in 0..writes_per_encoder {
            let offset = ((write % 1024) * 4) as u64;
            encoder
                .write_buffer(&target, offset, &(encoder_index as u32).to_le_bytes())
                .unwrap();
        }
        recordings.push(encoder);
    }
    let record_ms = monotonic_millis() - start;
    for encoder in recordings {
        // `submit` awaits the corresponding queue completion. A subsequent
        // workload may therefore take only pages with real retirement evidence.
        submit(d, encoder).await;
    }
    (record_ms, monotonic_millis() - start)
}

#[wasm_bindgen_test(async)]
async fn gpu_upload_arena_many_small_writes_and_nonzero_clear_round_trip() {
    command::reset_upload_page_creates();
    let d = open().await;
    let b = buffer(
        &d,
        8192,
        BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        MemoryClass::Device,
    );
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    for i in 0..1000_u64 {
        e.write_buffer(&b, i * 4, &(i as u32).to_le_bytes())
            .unwrap();
    }
    // Odd number of words exercises repeated native copies of one nonzero pattern.
    e.clear_buffer(&b, 4096..8188, 0x7e_5a_3c_19).unwrap();
    assert_eq!(
        command::upload_page_creates(),
        1,
        "writes and clear share one page"
    );
    let ticket = e.readback_buffer(&b, 0..8192).unwrap();
    let point = submit(&d, e).await;
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for i in 0..1000_usize {
        assert_eq!(&bytes[i * 4..i * 4 + 4], &(i as u32).to_le_bytes());
    }
    for word in bytes[4096..8188].chunks_exact(4) {
        assert_eq!(word, 0x7e_5a_3c_19_u32.to_le_bytes());
    }
}

#[wasm_bindgen_test(async)]
async fn gpu_upload_arena_encoders_submit_in_reverse_without_overwrite() {
    let d = open().await;
    let a = buffer(
        &d,
        16,
        BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        MemoryClass::Device,
    );
    let b = buffer(
        &d,
        16,
        BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        MemoryClass::Device,
    );
    let mut first = d.encoder(d.lanes()[0].id).unwrap();
    first.write_buffer(&a, 0, &[1; 16]).unwrap();
    let mut second = d.encoder(d.lanes()[0].id).unwrap();
    second.write_buffer(&b, 0, &[2; 16]).unwrap();
    let p2 = submit(&d, second).await;
    let p1 = submit(&d, first).await;
    let mut read = d.encoder(d.lanes()[0].id).unwrap();
    let ta = read.readback_buffer(&a, 0..16).unwrap();
    let tb = read.readback_buffer(&b, 0..16).unwrap();
    let point = submit(&d, read).await;
    assert_eq!(
        unsafe { d.read_after(ta, &point).await }.unwrap().bytes,
        [1; 16]
    );
    assert_eq!(
        unsafe { d.read_after(tb, &point).await }.unwrap().bytes,
        [2; 16]
    );
    let _ = (p1, p2);
}

#[wasm_bindgen_test(async)]
async fn gpu_upload_arena_reuses_pages_only_after_queue_completion() {
    let d = open().await;
    command::reset_upload_page_creates();
    let b = buffer(
        &d,
        16,
        BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        MemoryClass::Device,
    );
    let mut first = d.encoder(d.lanes()[0].id).unwrap();
    first.write_buffer(&b, 0, &[3; 16]).unwrap();
    submit(&d, first).await;
    assert_eq!(command::upload_page_creates(), 1);

    // `submit` waits for the first queue boundary, which is the only path
    // that admits its staging page into the device-local reusable arena.
    let mut second = d.encoder(d.lanes()[0].id).unwrap();
    second.write_buffer(&b, 0, &[4; 16]).unwrap();
    assert_eq!(
        command::upload_page_creates(),
        1,
        "a completed encoder should reuse its staging page"
    );
    submit(&d, second).await;
}

/// Local-only Review3 allocation and browser-time diagnostic.
///
/// Each case first records all encoders before submitting any, making its cold
/// allocation count visible. It then repeats only after every prior submission
/// has completed, so the second count reports the bounded pool's safe reuse.
/// This is intentionally opt-in because the 100×1000 workload is not a normal
/// conformance assertion.
#[cfg(feature = "review3-upload-benchmark")]
#[wasm_bindgen_test(async)]
async fn review3_upload_arena_timing_matrix() {
    for encoders in [1_usize, 10, 100] {
        for writes_per_encoder in [1_usize, 100, 1000] {
            let d = open().await;
            let target = buffer(&d, 4096, BufferUsage::COPY_DST, MemoryClass::Device);
            command::reset_upload_page_creates();
            let (cold_record_ms, cold_complete_ms) =
                record_and_complete_upload_workload(&d, &target, encoders, writes_per_encoder)
                    .await;
            let cold_page_creates = command::upload_page_creates();
            let (reused_record_ms, reused_complete_ms) =
                record_and_complete_upload_workload(&d, &target, encoders, writes_per_encoder)
                    .await;
            let total_page_creates = command::upload_page_creates();
            let reused_page_creates = total_page_creates - cold_page_creates;
            assert!(
                reused_page_creates <= cold_page_creates,
                "completion-protected reuse must not allocate more than the cold workload"
            );
            console_log!(
                "REVIEW3_WEBGPU_UPLOAD encoders={encoders} writes_per_encoder={writes_per_encoder} cold_page_creates={cold_page_creates} reused_page_creates={reused_page_creates} cold_record_ms={cold_record_ms:.3} cold_complete_ms={cold_complete_ms:.3} reused_record_ms={reused_record_ms:.3} reused_complete_ms={reused_complete_ms:.3}"
            );
        }
    }
}
