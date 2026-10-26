use super::*;
#[test]
fn serialized_entries_round_trip_and_reject_corruption() {
    let cache = Dx12PipelineCache(Arc::new(PipelineCacheState::default()));
    cache
        .0
        .entries
        .lock()
        .unwrap()
        .insert(b"stable-key".to_vec(), b"native-pso".to_vec());
    let data = <Dx12 as fluxel_rhi_core::backend::CacheBackend>::serialize_cache(&cache).unwrap();
    assert_eq!(
        parse(&data).unwrap().get(b"stable-key".as_slice()).unwrap(),
        b"native-pso"
    );
    let mut corrupted = data.clone();
    corrupted[15] ^= 1;
    assert!(matches!(parse(&corrupted), Err(Error::InvalidState)));
    for end in 0..data.len() {
        assert!(parse(&data[..end]).is_err());
    }
}
#[test]
fn rejects_duplicate_keys_even_with_valid_checksum() {
    let mut data = MAGIC.to_vec();
    data.extend_from_slice(&2u32.to_le_bytes());
    for v in [1u8, 2] {
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&[3, v]);
    }
    data.extend_from_slice(&checksum(&data).to_le_bytes());
    assert!(matches!(parse(&data), Err(Error::InvalidState)));
}
