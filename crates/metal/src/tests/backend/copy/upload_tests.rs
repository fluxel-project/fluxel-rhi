//! Unit tests for the corresponding implementation module.

use super::{align_up, repack_rows};
use fluxel_rhi_core::api::TexelCopyBufferLayout;

#[test]
fn upload_repacking_applies_offset_and_strips_source_padding() {
    let source = [99, 1, 2, 3, 4, 88, 5, 6, 7, 8, 77];
    let packed = repack_rows(
        &source,
        TexelCopyBufferLayout {
            offset: 1,
            bytes_per_row: 5,
            rows_per_image: 2,
        },
        4,
        2,
        8,
    )
    .unwrap();
    assert_eq!(packed, [1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0]);
}

#[test]
fn upload_repacking_rejects_truncated_source_and_overflowing_stride() {
    assert!(
        repack_rows(
            &[1, 2, 3],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 4,
                rows_per_image: 1,
            },
            4,
            1,
            4,
        )
        .is_err()
    );
    assert!(
        repack_rows(
            &[1, 2, 3, 4],
            TexelCopyBufferLayout {
                offset: u64::MAX,
                bytes_per_row: u32::MAX,
                rows_per_image: 2,
            },
            4,
            2,
            4,
        )
        .is_err()
    );
}

#[test]
fn row_alignment_rounds_without_overflow() {
    assert_eq!(align_up(513, 256), Some(768));
    assert_eq!(align_up(usize::MAX, 256), None);
}
