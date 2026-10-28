
use super::*;
#[test]
fn readback_size_includes_last_image_only_once() {
    let context = super::super::ContextStamp::new(
        super::super::DeviceIdentity::new(1).unwrap(),
        super::super::ContextEpoch::INITIAL,
    );
    let region = GlTextureRegion {
        subresource: super::super::GlTextureSubresource {
            texture: super::super::TextureId::new(context, 0, 0),
            aspect: super::super::GlTextureAspect::Color,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        },
        origin: [0; 3],
        extent: super::super::GlExtent3d {
            width: 2,
            height: 2,
            depth_or_layers: 2,
        },
    };
    let layout = GlPixelLayout {
        format: GlPixelFormat::Rgba8,
        bytes_per_row: 8,
        rows_per_image: 2,
        offset: 0,
        alignment: 4,
        repack: GlRepackPolicy::Disallow,
    };
    assert_eq!(layout.required_bytes(region), Ok(32));
}
#[test]
fn compressed_layout_counts_block_rows_and_last_partial_block() {
    let mut region = test_region();
    region.extent.width = 7;
    region.extent.height = 5;
    region.extent.depth_or_layers = 2;
    let layout = GlPixelLayout {
        format: GlPixelFormat::Compressed {
            block_width: 4,
            block_height: 4,
            block_bytes: 16,
        },
        bytes_per_row: 48,
        rows_per_image: 3,
        offset: 16,
        alignment: 8,
        repack: GlRepackPolicy::Disallow,
    };
    assert_eq!(layout.required_bytes(region), Ok(240));
    assert_eq!(
        GlPixelLayout {
            rows_per_image: 1,
            ..layout
        }
        .required_bytes(region),
        Err(GlCopyValidationError::ImageTooShort)
    );
    assert_eq!(
        GlPixelLayout {
            bytes_per_row: 16,
            ..layout
        }
        .required_bytes(region),
        Err(GlCopyValidationError::RowTooShort)
    );
}
#[test]
fn pixel_store_alignments_are_checked() {
    let mut state = GlPixelStoreState::DEFAULT;
    state.pack_alignment = 3;
    assert_eq!(
        state.validate(),
        Err(GlCopyValidationError::InvalidPixelAlignment)
    );
}
#[test]
fn unmappable_rows_need_an_explicit_bounded_repack() {
    let region = test_region();
    let layout = GlPixelLayout {
        format: GlPixelFormat::Rgba8,
        bytes_per_row: 9,
        rows_per_image: 2,
        offset: 0,
        alignment: 4,
        repack: GlRepackPolicy::Disallow,
    };
    assert_eq!(
        layout.validate_for(region),
        Err(GlCopyValidationError::UnmappableRowStride)
    );
}
/// The layout the two directions disagreed about: `bytes_per_row` is a
/// multiple of the pixel size but not of the declared alignment, so GL
/// cannot express the stride and the payload is staged tightly.  The
/// parameters have to describe the staging rather than the caller's buffer,
/// because describing the caller's buffer walks GL a full `bytes_per_row`
/// per row through a staging allocation that only holds `width * bpp`.
#[test]
fn an_unexpressible_stride_is_described_as_tight_not_as_the_callers() {
    let layout = GlPixelLayout {
        format: GlPixelFormat::Rgba8,
        bytes_per_row: 12,
        rows_per_image: 2,
        offset: 0,
        alignment: 8,
        repack: GlRepackPolicy::Bounded { max_bytes: 64 },
    };
    assert!(!layout.stride_is_mappable());
    assert_eq!(layout.staged_row_parameters(), (1, 0));
}

#[test]
fn an_expressible_stride_is_described_as_the_callers_own_pitch() {
    let layout = GlPixelLayout {
        format: GlPixelFormat::Rgba8,
        bytes_per_row: 16,
        rows_per_image: 2,
        offset: 0,
        alignment: 4,
        repack: GlRepackPolicy::Disallow,
    };
    assert!(layout.stride_is_mappable());
    assert_eq!(layout.staged_row_parameters(), (4, 4));
}

/// The invariant the two tests above are instances of, so a later edit to
/// either arm has to keep it: a stride GL cannot express never carries a
/// row length, and an expressible one always does.
#[test]
fn the_row_length_is_present_exactly_when_the_stride_is_mappable() {
    for bytes_per_row in [3u32, 4, 5, 8, 12, 16, 17, 24] {
        for alignment in [1u8, 2, 4, 8] {
            let layout = GlPixelLayout {
                format: GlPixelFormat::Rgba8,
                bytes_per_row,
                rows_per_image: 2,
                offset: 0,
                alignment,
                repack: GlRepackPolicy::Bounded { max_bytes: 4_096 },
            };
            let (told_alignment, row_length) = layout.staged_row_parameters();
            assert_eq!(row_length != 0, layout.stride_is_mappable(), "{layout:?}");
            // Tight staging is alignment one, which is what makes a row
            // exactly `width * bpp` with nothing padding it.
            if row_length == 0 {
                assert_eq!(told_alignment, 1, "{layout:?}");
            }
        }
    }
}

/// A zero extent used to reach `required_bytes_unchecked`, which subtracts
/// one from the height and from the layer count: a panic in a debug build,
/// a `u32::MAX` wrap reported as a size error in a release one.  The guard
/// that existed for it was in a function nothing called.
#[test]
fn a_zero_extent_is_refused_as_a_shape_rather_than_underflowing() {
    let region = GlTextureRegion {
        subresource: test_region().subresource,
        origin: [0; 3],
        extent: super::super::GlExtent3d {
            width: 2,
            height: 0,
            depth_or_layers: 1,
        },
    };
    let layout = GlPixelLayout {
        format: GlPixelFormat::Rgba8,
        bytes_per_row: 8,
        rows_per_image: 0,
        offset: 0,
        alignment: 4,
        repack: GlRepackPolicy::Bounded { max_bytes: 4_096 },
    };
    assert_eq!(
        layout.validate_for(region),
        Err(GlCopyValidationError::ZeroTransferExtent)
    );
    // `required_bytes` validates first, so the subtraction it would
    // otherwise do on this height is never reached.
    assert_eq!(
        layout.required_bytes(region),
        Err(GlCopyValidationError::ZeroTransferExtent)
    );
}

#[test]
fn readback_requires_the_exact_returned_length() {
    let region = test_region();
    let layout = GlPixelLayout {
        format: GlPixelFormat::Rgba8,
        bytes_per_row: 8,
        rows_per_image: 2,
        offset: 0,
        alignment: 4,
        repack: GlRepackPolicy::Disallow,
    };
    assert_eq!(
        GlReadback {
            layout,
            bytes: vec![0; 31]
        }
        .validate_for(region),
        Err(GlCopyValidationError::ReadbackLengthMismatch)
    );
}
fn test_region() -> GlTextureRegion {
    let context = super::super::ContextStamp::new(
        super::super::DeviceIdentity::new(1).unwrap(),
        super::super::ContextEpoch::INITIAL,
    );
    GlTextureRegion {
        subresource: super::super::GlTextureSubresource {
            texture: super::super::TextureId::new(context, 0, 0),
            aspect: super::super::GlTextureAspect::Color,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        },
        origin: [0; 3],
        extent: super::super::GlExtent3d {
            width: 2,
            height: 2,
            depth_or_layers: 1,
        },
    }
}
