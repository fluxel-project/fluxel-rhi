//! Native WebGPU buffer-usage validation and query-readback regressions.
use super::*;

#[wasm_bindgen_test(async)]
async fn gpu_query_resolve_logical_map_read_uses_staging() {
    let d = open().await;
    if !d.capabilities().timestamp {
        console_log!("SKIP timestamp-query: browser does not expose the feature");
        return;
    }
    let queries = d.queries(QueryKind::Timestamp).unwrap();
    let set = queries
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 2,
        })
        .unwrap();
    // This is logically readable but its native resolve buffer must omit MAP_READ.
    let output = buffer(
        &d,
        256,
        BufferUsage::QUERY_RESOLVE | BufferUsage::MAP_READ,
        MemoryClass::Readback,
    );
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    queries.timestamp(&mut e, &set, 0).unwrap();
    queries.timestamp(&mut e, &set, 1).unwrap();
    queries.resolve(&mut e, &set, 0..2, &output, 0).unwrap();
    let ticket = e.readback_buffer(&output, 0..16).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: the point covers resolve and its private MAP_READ staging copy.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let first = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let second = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    assert!(
        second >= first && second != 0,
        "resolved timestamps {first} -> {second}"
    );
}

#[wasm_bindgen_test(async)]
async fn invalid_native_mapping_combinations_do_not_poison_later_submit() {
    let d = open().await;
    for usage in [
        BufferUsage::MAP_READ | BufferUsage::STORAGE,
        BufferUsage::MAP_WRITE | BufferUsage::COPY_DST,
    ] {
        assert!(matches!(
            d.buffer_support(&BufferDesc {
                label: None,
                size: 16,
                usage,
                memory: MemoryClass::Device
            }),
            Support::Unsupported(_)
        ));
        assert!(
            d.create_buffer(&BufferDesc {
                label: None,
                size: 16,
                usage,
                memory: MemoryClass::Device
            })
            .is_err()
        );
    }
    let source = buffer(
        &d,
        16,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let target = buffer(
        &d,
        16,
        BufferUsage::COPY_DST | BufferUsage::MAP_READ,
        MemoryClass::Readback,
    );
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    e.write_buffer(&source, 0, &[3, 1, 4, 1]).unwrap();
    e.copy_buffer(&source, 0, &target, 0, 4).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: the normal copy submission follows rejected descriptors and is complete.
    let mapped = unsafe { d.map_buffer(&target, 0..4, MapMode::Read).await }.unwrap();
    assert_eq!(mapped.bytes(), [3, 1, 4, 1]);
    let _ = point;
}
