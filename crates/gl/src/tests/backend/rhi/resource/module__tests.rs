
use super::lower_buffer_binding_range;
use fluxel_rhi_core::api::Error;

#[test]
fn buffer_binding_lowering_preserves_offsets_above_four_gib() {
    let start = 4_u64 * 1024 * 1024 * 1024 + 256;
    assert_eq!(
        lower_buffer_binding_range(start, start + 512, start + 1024, 512),
        Ok((start, 512))
    );
    assert_eq!(
        lower_buffer_binding_range(start, start + 511, start + 1024, 512),
        Err(Error::InvalidRange)
    );
}
