
use super::*;
#[test]
fn core_and_extension_evidence_is_format_specific() {
    let gl40 = GlFamilyProfile::Desktop { major: 4, minor: 0 };
    let gl42 = GlFamilyProfile::Desktop { major: 4, minor: 2 };
    let es31 = GlFamilyProfile::Embedded { major: 3, minor: 1 };
    let es32 = GlFamilyProfile::Embedded { major: 3, minor: 2 };
    let astc = GlFormat::Astc {
        block: GlAstcBlock::B4x4,
        color_space: GlCompressedColorSpace::Linear,
    };
    assert!(GlFormat::Bc4RUnorm.is_core_compressed_for(gl40));
    assert!(!GlFormat::Bc7RgbaUnorm.is_core_compressed_for(gl40));
    assert!(GlFormat::Bc7RgbaUnorm.is_core_compressed_for(gl42));
    assert!(!astc.is_core_compressed_for(es31));
    assert!(astc.is_core_compressed_for(es32));
    assert!(astc.is_supplied_by_extension(GlKnownExtension::KhrTextureCompressionAstcHdr));
    let hdr = GlFormat::Astc {
        block: GlAstcBlock::B4x4,
        color_space: GlCompressedColorSpace::Hdr,
    };
    assert!(!hdr.is_supplied_by_extension(GlKnownExtension::KhrTextureCompressionAstcLdr));
    assert!(
        GlFormat::Etc2Rgb8Unorm
            .is_supplied_by_extension(GlKnownExtension::OesCompressedEtc2Rgb8Texture)
    );
    assert!(
        !GlFormat::EacR11Unorm
            .is_supplied_by_extension(GlKnownExtension::OesCompressedEtc2Rgb8Texture)
    );
}
