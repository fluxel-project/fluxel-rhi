//! Unit tests for the corresponding implementation module.

use super::*;

#[test]
fn validates_complete_strided_records_and_count_word() {
    let usage = BufferUsage::INDIRECT;
    assert!(validate(usage, 68, 0, usage, 8, 4, 3, 24, true).is_ok());
    assert!(validate(usage, 67, 0, usage, 8, 4, 3, 24, true).is_err());
    assert!(validate(usage, 68, 0, usage, 7, 4, 3, 24, true).is_err());
    assert!(validate(usage, 68, 0, usage, 8, 4, 3, 18, true).is_err());
    assert!(validate(BufferUsage::COPY_SRC, 68, 0, usage, 8, 4, 3, 24, true).is_err());
}

#[test]
fn partitions_at_metal3_limit_without_losing_or_wrapping_commands() {
    assert_eq!(chunks(0).collect::<Vec<_>>(), vec![]);
    assert_eq!(
        chunks(16_385).collect::<Vec<_>>(),
        vec![(0, 16_384), (16_384, 1)]
    );
    let last = chunks(u32::MAX).last().unwrap();
    assert_eq!(u64::from(last.0) + u64::from(last.1), u64::from(u32::MAX));
}

#[test]
fn indexed_base_vertex_words_preserve_signed_bits() {
    for (value, word) in [(i32::MIN, 0x8000_0000), (-1, 0xffff_ffff), (0, 0), (1, 1)] {
        assert_eq!(u32::from_ne_bytes(value.to_ne_bytes()), word);
        assert_eq!(i32::from_ne_bytes(word.to_ne_bytes()), value);
    }
}

#[test]
fn recording_requires_active_raster_pipeline_render_and_index_when_indexed() {
    assert!(valid_recording_state(true, true, false, true, false, false));
    assert!(valid_recording_state(true, true, false, true, true, true));
    assert!(!valid_recording_state(
        false, true, false, true, false, false
    ));
    assert!(!valid_recording_state(true, true, true, true, false, false));
    assert!(!valid_recording_state(
        true, true, false, false, false, false
    ));
    assert!(!valid_recording_state(true, true, false, true, true, false));
}
