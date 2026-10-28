use super::indirect_cleanup_support;

#[test]
fn indirect_cleanup_never_names_unsupported_targets() {
    assert_eq!(
        indirect_cleanup_support(true, 3, 0, false, false),
        (false, false)
    );
    assert_eq!(
        indirect_cleanup_support(true, 3, 1, false, false),
        (true, true)
    );
    assert_eq!(
        indirect_cleanup_support(false, 4, 0, false, false),
        (true, false)
    );
    assert_eq!(
        indirect_cleanup_support(false, 4, 2, false, false),
        (true, false)
    );
    assert_eq!(
        indirect_cleanup_support(false, 4, 3, false, false),
        (true, true)
    );
    assert_eq!(
        indirect_cleanup_support(false, 3, 3, true, false),
        (true, false)
    );
    assert_eq!(
        indirect_cleanup_support(false, 3, 3, false, true),
        (false, true)
    );
}
