//! Surface drawable fact validation tests.

use crate::backend::surface::acquire::drawable_matches_configuration;
use fluxel_rhi_core::api::TextureFormat;
use objc2_metal::MTLPixelFormat;

#[test]
fn drawable_fact_check_rejects_out_of_band_extent_and_format_changes() {
    let extent = [640, 480];
    assert!(drawable_matches_configuration(
        extent,
        MTLPixelFormat::BGRA8Unorm,
        extent,
        TextureFormat::Bgra8Unorm,
    ));
    assert!(!drawable_matches_configuration(
        [641, 480],
        MTLPixelFormat::BGRA8Unorm,
        extent,
        TextureFormat::Bgra8Unorm,
    ));
    assert!(!drawable_matches_configuration(
        extent,
        MTLPixelFormat::BGRA8Unorm_sRGB,
        extent,
        TextureFormat::Bgra8Unorm,
    ));
}

#[test]
fn drawable_fact_check_accepts_each_advertised_surface_format_only() {
    for (actual, expected) in [
        (MTLPixelFormat::BGRA8Unorm, TextureFormat::Bgra8Unorm),
        (
            MTLPixelFormat::BGRA8Unorm_sRGB,
            TextureFormat::Bgra8UnormSrgb,
        ),
        (MTLPixelFormat::RGBA16Float, TextureFormat::Rgba16Float),
        (MTLPixelFormat::RGB10A2Unorm, TextureFormat::Rgb10a2Unorm),
    ] {
        assert!(drawable_matches_configuration(
            [1, 1],
            actual,
            [1, 1],
            expected
        ));
    }
}
