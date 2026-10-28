
use super::*;
#[test]
fn masks_validate_and_all_is_the_supported_resource_domain() {
    assert_eq!(
        scope(api::StageMask::ALL_COMMANDS, api::AccessMask::ALL, BUFFER).unwrap(),
        BUFFER
    );
    assert!(scope(api::StageMask(1 << 32), api::AccessMask::EMPTY, BUFFER).is_err());
    assert!(scope(api::StageMask::VERTEX, api::AccessMask(1 << 45), BUFFER).is_err());
    assert!(scope(api::StageMask::EMPTY, api::AccessMask::SHADER_READ, BUFFER).is_err());
    assert!(scope(api::StageMask::RAY, api::AccessMask::SHADER_READ, TEXTURE).is_err());
}
#[test]
fn consumers_choose_visibility_for_the_actual_gl_operation() {
    use api::{AccessMask as A, ImageLayout as L};
    assert_eq!(
        destination(A::SHADER_READ.0, true, Some(L::ShaderReadOnly)),
        Bits::TEXTURE_FETCH
    );
    assert!(
        destination(A::SHADER_READ.0, true, Some(L::General))
            .contains(Bits::TEXTURE_FETCH.union(Bits::SHADER_IMAGE_ACCESS))
    );
    assert!(
        destination(A::COPY_READ.0, true, Some(L::CopySrc))
            .contains(Bits::FRAMEBUFFER.union(Bits::TEXTURE_UPDATE))
    );
    assert!(
        destination(A::COPY_READ.0, false, None)
            .contains(Bits::PIXEL_BUFFER.union(Bits::BUFFER_UPDATE))
    );
    assert_eq!(destination(A::INDIRECT_READ.0, false, None), Bits::COMMAND);
}
