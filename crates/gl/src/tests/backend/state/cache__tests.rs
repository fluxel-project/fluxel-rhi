use super::*;
fn texture() -> ResourceRef {
    use crate::backend::api::{ContextEpoch, ContextStamp, DeviceIdentity, TextureId};
    ResourceRef::Texture(TextureId::new(
        ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL),
        1,
        1,
    ))
}
#[test]
fn lru_and_dependency_invalidation_are_deterministic() {
    let mut c = StructuralCache::new(CacheBudget::new(2, 20));
    let mut n = CacheCounters::default();
    c.insert(1, 1, 8, DependencySet::new(), &mut n);
    c.insert(2, 2, 8, DependencySet::new(), &mut n);
    c.get(&1, &mut n);
    assert_eq!(
        c.insert(3, 3, 8, DependencySet::new(), &mut n).removed,
        vec![(2, 2)]
    );
    let mut d = DependencySet::new();
    d.insert(texture());
    c.insert(4, 4, 8, d, &mut n);
    assert_eq!(c.invalidate_resource(texture(), &mut n), vec![(4, 4)]);
}
