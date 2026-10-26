//! Unit tests for the corresponding implementation module.

#[test]
fn exact_frequency_only() {
    assert_eq!(super::integer_frequency(10.0), Some(100_000_000));
    assert_eq!(super::integer_frequency(3.0), None);
    assert_eq!(super::integer_frequency(0.0), None);
    assert_eq!(super::integer_frequency(f32::MIN_POSITIVE), None);
}
