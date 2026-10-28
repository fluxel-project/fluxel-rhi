use super::{GlStorageBufferLimits, GlStorageBufferRange, GlStorageBufferUsage};
use crate::backend::api::{BufferId, ContextEpoch, ContextStamp, DeviceIdentity};
#[test]
fn rejects_unaligned_storage_buffer_range() {
    let stamp = ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL);
    let range = GlStorageBufferRange {
        buffer: BufferId::new(stamp, 1, 1),
        offset: 3,
        size: 4,
        usage: GlStorageBufferUsage::ReadOnly,
    };
    assert!(
        range
            .validate(
                0,
                GlStorageBufferLimits {
                    max_bindings: 1,
                    max_block_size: 4,
                    offset_alignment: 4
                }
            )
            .is_err()
    );
}
