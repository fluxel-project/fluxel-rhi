//! Regression checks for the owning private GL boundary.
use super::*;

#[test]
fn rect_clear_selects_the_attachment_numeric_clear_entrypoint() {
    assert!(color_clear_matches(
        GlFormat::Rgba8Unorm,
        ClearColor::Float([0.0; 4])
    ));
    assert!(!color_clear_matches(
        GlFormat::Rgba8Unorm,
        ClearColor::Uint([0; 4])
    ));
    assert!(color_clear_matches(
        GlFormat::Rgba32Uint,
        ClearColor::Uint([0; 4])
    ));
    assert!(!color_clear_matches(
        GlFormat::Rgba32Uint,
        ClearColor::Sint([0; 4])
    ));
    assert!(color_clear_matches(
        GlFormat::Rgba16Sint,
        ClearColor::Sint([0; 4])
    ));
}

#[test]
fn rect_clear_rejects_coordinates_outside_gl_integer_range() {
    assert!(
        clear_native_rect(Rect {
            x: u32::MAX,
            y: 0,
            width: 1,
            height: 1,
        })
        .is_err()
    );
}
