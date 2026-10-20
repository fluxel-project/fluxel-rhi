//! Standard error integration and flag combination behavior.
use super::*;

#[test]
fn error_implements_standard_error_and_reports_details() {
    fn accepts_error(_: &dyn std::error::Error) {}
    let error = Error::Backend("native failure".into());
    accepts_error(&error);
    assert_eq!(error.to_string(), "backend error: native failure");
    assert_eq!(
        Error::WrongDevice.to_string(),
        "resource belongs to a different device"
    );
}

#[test]
fn flags_support_or_and_or_assign() {
    macro_rules! check {
        ($a:expr, $b:expr) => {{
            let expected = $a.0 | $b.0;
            assert_eq!(($a | $b).0, expected);
            let mut value = $a;
            value |= $b;
            assert_eq!(value.0, expected);
        }};
    }
    check!(BufferUsage::VERTEX, BufferUsage::COPY_DST);
    check!(TextureUsage::COPY_SRC, TextureUsage::TEXTURE_BINDING);
    check!(TextureAspects::DEPTH, TextureAspects::STENCIL);
    check!(SurfaceUsage::RENDER_ATTACHMENT, SurfaceUsage::COPY_SRC);
    check!(StageMask::VERTEX, StageMask::FRAGMENT);
    check!(AccessMask::SHADER_READ, AccessMask::SHADER_WRITE);
    const USAGE: BufferUsage = BufferUsage::VERTEX.union(BufferUsage::COPY_DST);
    assert_eq!(USAGE, BufferUsage::VERTEX | BufferUsage::COPY_DST);
}
