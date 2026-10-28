use super::*;
#[test]
fn packed_float_and_srgb_clear_encoding() {
    assert_eq!(small_float(1., 10, true), 0x3c00);
    assert_eq!(small_float(2_f32.powi(-24), 10, true), 1);
    assert_eq!(small_float(2_f32.powi(-25), 10, true), 0);
    assert_eq!(small_float(65504., 10, true), 0x7bff);
    assert_eq!(small_float(1., 6, false), 0x3c0);
    assert_eq!(rgb9e5([1., 0.5, 0., 0.]), 0x80010100);
    assert_eq!(
        encode(F::Rgba8Srgb, ClearColor::Float([0.5, 0., 1., 0.5])).unwrap(),
        [188, 0, 255, 128]
    );
    assert_eq!(
        encode(F::Rg8Snorm, ClearColor::Float([-1., 0.5, 0., 0.])).unwrap(),
        [129, 64]
    );
}
