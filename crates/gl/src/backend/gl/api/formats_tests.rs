//! Tests for the private formats.rs contract.
use super::{
    GlAstcBlock, GlCompressedColorSpace, GlCompressedSizeError, GlFamilyProfile, GlFormat,
    GlFormatCapabilities, GlFormatEvidence, GlFormatResourceKind, GlFormatTable,
    GlFormatTableError,
};
use crate::backend::gl::api::GlLimits;

fn baseline(table: &mut GlFormatTable) {
    for format in [
        GlFormat::Rgba8Unorm,
        GlFormat::Rgba8Srgb,
        GlFormat::Depth32Float,
    ] {
        table
            .record(GlFormatCapabilities {
                format,
                resource_kind: GlFormatResourceKind::Texture,
                sample_count: 1,
                evidence: GlFormatEvidence::CoreGuaranteed,
                sampled: true,
                filterable: format != GlFormat::Depth32Float,
                renderable: true,
                blendable: format != GlFormat::Depth32Float,
                storage_read: false,
                storage_write: false,
                copy_source: format != GlFormat::Depth32Float,
                copy_destination: format != GlFormat::Depth32Float,
            })
            .expect("baseline fact");
    }
}

#[test]
fn static_guarantees_cannot_claim_storage_images() {
    let mut table = GlFormatTable::default();
    let error = table
        .record(GlFormatCapabilities {
            format: GlFormat::Rgba8Unorm,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: GlFormatEvidence::CoreGuaranteed,
            sampled: true,
            filterable: true,
            renderable: true,
            blendable: true,
            storage_read: true,
            storage_write: true,
            copy_source: true,
            copy_destination: true,
        })
        .expect_err("storage requires an operation probe");
    assert!(matches!(
        error,
        GlFormatTableError::StorageRequiresOperationProbe { .. }
    ));
}

#[test]
fn renderbuffer_multisampling_does_not_prove_texture_multisampling() {
    let mut table = GlFormatTable::default();
    baseline(&mut table);
    table
        .record(GlFormatCapabilities {
            format: GlFormat::Rgba8Unorm,
            resource_kind: GlFormatResourceKind::Renderbuffer,
            sample_count: 4,
            evidence: GlFormatEvidence::OperationProbed,
            sampled: false,
            filterable: false,
            renderable: true,
            blendable: true,
            storage_read: false,
            storage_write: false,
            copy_source: false,
            copy_destination: false,
        })
        .expect("renderbuffer operation probe");
    let mut limits = GlLimits::unavailable();
    limits.max_samples = 4;
    assert!(table.validate_for_limits(&limits).is_ok());
    assert!(
        table
            .get_for(GlFormatResourceKind::Texture, GlFormat::Rgba8Unorm, 4)
            .is_none()
    );
}

#[test]
fn compressed_formats_preserve_exact_block_layout_and_color_space() {
    let bc1 = GlFormat::Bc1RgbaSrgb
        .compressed_info()
        .expect("BC1 is compressed");
    assert_eq!(
        (bc1.block_width, bc1.block_height, bc1.block_bytes),
        (4, 4, 8)
    );
    assert_eq!(bc1.color_space, GlCompressedColorSpace::Srgb);
    assert_eq!(
        GlFormat::Bc1RgbSrgb
            .compressed_info()
            .expect("sRGB BC1 RGB")
            .color_space,
        GlCompressedColorSpace::Srgb
    );
    let astc = GlFormat::Astc {
        block: GlAstcBlock::B10x6,
        color_space: GlCompressedColorSpace::Linear,
    }
    .compressed_info()
    .expect("ASTC is compressed");
    assert_eq!(
        (astc.block_width, astc.block_height, astc.block_bytes),
        (10, 6, 16)
    );
}

#[test]
fn compressed_family_metadata_cannot_enable_a_different_exact_format() {
    let bc1 = GlFormat::Bc1RgbaUnorm.compressed_info();
    let bc3 = GlFormat::Bc3RgbaUnorm.compressed_info();
    assert_ne!(
        bc1, bc3,
        "one BC family bit cannot replace exact format facts"
    );
    let mut table = GlFormatTable::default();
    let error = table
        .record(GlFormatCapabilities {
            format: GlFormat::Etc2Rgb8Unorm,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: GlFormatEvidence::OperationProbed,
            sampled: true,
            filterable: true,
            renderable: true,
            blendable: false,
            storage_read: false,
            storage_write: false,
            copy_source: false,
            copy_destination: false,
        })
        .expect_err("compressed formats cannot become render targets");
    assert!(matches!(
        error,
        GlFormatTableError::InvalidCompressedUsage { .. }
    ));
}

#[test]
fn gles3_core_guarantee_is_exactly_etc2_and_eac() {
    let gles3 = GlFamilyProfile::Embedded { major: 3, minor: 0 };
    assert!(GlFormat::Etc2Rgba8Srgb.is_core_compressed_for(gles3));
    assert!(GlFormat::EacRg11Unorm.is_core_compressed_for(gles3));
    assert!(GlFormat::Etc2Rgb8A1Srgb.is_core_compressed_for(gles3));
    assert!(GlFormat::EacR11Snorm.is_core_compressed_for(gles3));
    assert!(
        !GlFormat::Astc {
            block: GlAstcBlock::B4x4,
            color_space: GlCompressedColorSpace::Linear,
        }
        .is_core_compressed_for(gles3)
    );
}

#[test]
fn compressed_encoded_sizes_use_exact_blocks_and_reject_zero_extents() {
    let bc3 = GlFormat::Bc3RgbaUnorm.compressed_info().expect("BC3 info");
    assert_eq!(bc3.checked_encoded_size(5, 5), Ok(64));
    assert_eq!(
        bc3.checked_encoded_size(0, 8),
        Err(GlCompressedSizeError::ZeroExtent)
    );
}
