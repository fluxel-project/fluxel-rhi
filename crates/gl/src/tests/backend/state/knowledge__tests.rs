use super::*;
#[test]
fn unknown_never_agrees() {
    let mut value = DriverKnowledge::Unknown;
    assert!(!value.agrees(&7));
    value.set(7);
    assert!(value.agrees(&7));
    value.invalidate();
    assert!(!value.agrees(&7));
}
#[test]
fn domains_have_unique_bits() {
    let mut bits = 0;
    for d in StateDomain::ALL {
        assert_eq!(bits & d.bit(), 0);
        bits |= d.bit();
    }
    assert_eq!(bits, DirtyDomains::ALL.0);
}
