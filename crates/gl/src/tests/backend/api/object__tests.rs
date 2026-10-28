
use core::any::TypeId;

use super::{
    BufferId, ContextEpoch, ContextStamp, DeviceIdentity, FramebufferId, ObjectIdentity, SamplerId,
    ShaderId, SurfaceImageId, SyncId, TextureId, VertexArrayId,
};

fn stamp() -> ContextStamp {
    ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL)
}

#[test]
fn object_kinds_have_distinct_static_identity_types() {
    assert_ne!(TypeId::of::<BufferId>(), TypeId::of::<TextureId>());
    assert_ne!(TypeId::of::<SamplerId>(), TypeId::of::<ShaderId>());
    assert_ne!(TypeId::of::<VertexArrayId>(), TypeId::of::<FramebufferId>());
    assert_ne!(TypeId::of::<SyncId>(), TypeId::of::<SurfaceImageId>());
}

#[test]
fn typed_identities_keep_only_fluxel_owned_metadata() {
    let context = stamp();
    let sampler = SamplerId::new(context, 7, 3);

    assert_eq!(sampler.context, context);
    assert_eq!(sampler.slot, 7);
    assert_eq!(sampler.generation, 3);

    fn require_surface_image(_: SurfaceImageId) {}
    require_surface_image(ObjectIdentity::new(context, 2, 1));
}
