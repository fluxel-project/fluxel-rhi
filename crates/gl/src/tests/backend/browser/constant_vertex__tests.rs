
use super::half;

#[test]
fn half_float_boundaries() {
    assert_eq!(half(0x0001), 2_f32.powi(-24));
    assert_eq!(half(0x03ff), 1023. * 2_f32.powi(-24));
    assert_eq!(half(0x0400), 2_f32.powi(-14));
    assert_eq!(half(0x3c00), 1.);
    assert_eq!(half(0xfc00), f32::NEG_INFINITY);
    assert!(half(0x7e00).is_nan());
}
