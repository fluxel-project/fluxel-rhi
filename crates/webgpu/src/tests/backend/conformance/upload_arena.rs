//! Real-browser checks for encoder-local upload page ordering and reuse.

use super::*;
use crate::backend::command;

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
