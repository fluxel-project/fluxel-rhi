
use super::*;
fn desc() -> GlSamplerDesc {
    GlSamplerDesc {
        address_mode_u: GlAddressMode::ClampToEdge,
        address_mode_v: GlAddressMode::ClampToEdge,
        address_mode_w: GlAddressMode::ClampToEdge,
        mag_filter: GlFilterMode::Linear,
        min_filter: GlFilterMode::Linear,
        mipmap_filter: GlMipmapFilterMode::Linear,
        lod_min_bits: 0.0f32.to_bits(),
        lod_max_bits: 1.0f32.to_bits(),
        compare: None,
        max_anisotropy_bits: None,
    }
}
#[test]
fn rejects_nan_lod() {
    let mut d = desc();
    d.lod_min_bits = f32::NAN.to_bits();
    assert_eq!(d.validate(), Err(GlSamplerValidationError::NonFiniteLod));
}
#[test]
fn preserves_anisotropy_float_validation() {
    let mut d = desc();
    d.max_anisotropy_bits = Some(0.5f32.to_bits());
    assert_eq!(
        d.validate(),
        Err(GlSamplerValidationError::InvalidAnisotropy)
    );
}
#[test]
fn anisotropy_uses_the_acquirable_typed_extension() {
    assert_eq!(
        GlKnownExtension::ExtTextureFilterAnisotropic.raw_name(),
        "EXT_texture_filter_anisotropic"
    );
}
