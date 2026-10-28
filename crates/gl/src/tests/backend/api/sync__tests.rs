use super::{GlFenceLeaseBook, GlMemoryBarrier};
use crate::backend::api::{ContextEpoch, ContextStamp, DeviceIdentity, SyncId};
#[test]
fn revoked_lease_is_rejected() {
    let s = ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL);
    let mut book = GlFenceLeaseBook::default();
    let lease = book.issue(SyncId::new(s, 1, 1)).unwrap();
    book.revoke(lease);
    assert!(book.validate(lease).is_err());
}
#[test]
fn by_region_refuses_non_local_visibility_classes() {
    assert!(GlMemoryBarrier::TEXTURE_FETCH.validate_by_region().is_ok());
    assert!(GlMemoryBarrier::COMMAND.validate_by_region().is_err());
}
