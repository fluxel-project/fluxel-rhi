use super::*;

#[test]
fn chunk_windows_preserve_wide_resource_offsets_and_bound_shader_addresses() {
    for indexed in [false, true] {
        for stride in [32, 1 << 20, u32::MAX - 3] {
            let count = chunk_records(stride, indexed).unwrap();
            assert!(count > 0 && count.div_ceil(64) <= 65_535);
            // FirstElement itself exceeds u32; only local shader offsets
            // are narrow. No physical multi-gigabyte allocation is needed.
            let offset = (u64::from(u32::MAX) + 1) * 4 + 12;
            let (base, low, elements) = source_window(offset, count, stride, indexed).unwrap();
            assert_eq!(base / 4, u64::from(u32::MAX) + 1);
            assert_eq!(low, 12);
            assert!(u64::from(elements) * 4 <= MAX_VIEW_BYTES);
            let record = if indexed { 20 } else { 16 };
            assert_eq!(
                u64::from(elements) * 4,
                u64::from(low) + u64::from(count - 1) * u64::from(stride) + record
            );
            if count < 65_535 * 64 {
                assert!(source_window(offset, count + 1, stride, indexed).is_err());
            }
        }
    }
    assert!(chunk_records(0, false).is_err());
    assert!(source_window(0, 0, 16, false).is_err());
    assert_eq!(source_window(12, 1, 16, false).unwrap(), (0, 12, 7));
}
