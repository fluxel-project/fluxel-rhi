//! Focused real-GPU buffers output assertions.
use super::*;

#[wasm_bindgen_test(async)]
async fn gpu_unaligned_writes_preserve_neighbor_bytes_and_readback_crops() {
    let d = open().await;
    let b = buffer(
        &d,
        32,
        BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        MemoryClass::Device,
    );
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    e.write_buffer(&b, 0, &[0x55; 32]).unwrap();
    e.write_buffer(&b, 3, &[1, 2, 3, 4, 5]).unwrap();
    let ticket = e.readback_buffer(&b, 1..12).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: the awaited point covers the copy producing this ticket.
    let read = unsafe { d.read_after(ticket, &point).await }.unwrap();
    assert_eq!(
        read.bytes,
        [0x55, 0x55, 1, 2, 3, 4, 5, 0x55, 0x55, 0x55, 0x55]
    );
}

#[wasm_bindgen_test(async)]
async fn gpu_misaligned_buffer_copy_preserves_untouched_words() {
    let d = open().await;
    let source = buffer(
        &d,
        2048,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let target = buffer(
        &d,
        2048,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let input: Vec<u8> = (0..2048).map(|i| (i % 251) as u8).collect();
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    e.write_buffer(&source, 0, &input).unwrap();
    e.write_buffer(&target, 0, &[0xa5; 2048]).unwrap();
    e.copy_buffer(&source, 1, &target, 2, 1031).unwrap();
    let ticket = e.readback_buffer(&target, 0..2048).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: copy and readback precede the awaited point.
    let actual = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let mut expected = vec![0xa5; 2048];
    expected[2..1033].copy_from_slice(&input[1..1032]);
    assert_eq!(actual, expected);
}

#[wasm_bindgen_test(async)]
async fn gpu_mapping_write_read_and_exclusion_cover_exact_ranges() {
    let d = open().await;
    let upload = buffer(
        &d,
        32,
        BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC,
        MemoryClass::Upload,
    );
    // SAFETY: the new upload has never been submitted.
    let mut map = unsafe { d.map_buffer(&upload, 3..11, MapMode::Write).await }.unwrap();
    map.bytes_mut()
        .unwrap()
        .copy_from_slice(&[11, 12, 13, 14, 15, 16, 17, 18]);
    map.flush(0..8).unwrap();
    // SAFETY: this deliberately invalid alias must be rejected before another native map.
    assert!(unsafe { d.map_buffer(&upload, 0..8, MapMode::Write).await }.is_err());
    drop(map);
    let dst = buffer(
        &d,
        32,
        BufferUsage::COPY_DST | BufferUsage::MAP_READ,
        MemoryClass::Readback,
    );
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    e.copy_buffer(&upload, 0, &dst, 0, 32).unwrap();
    submit(&d, e).await;
    // SAFETY: the only GPU write to dst has completed.
    let map = unsafe { d.map_buffer(&dst, 3..11, MapMode::Read).await }.unwrap();
    assert_eq!(map.bytes(), [11, 12, 13, 14, 15, 16, 17, 18]);
}
