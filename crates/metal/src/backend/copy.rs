//! Shared validation for direct Metal blit operations.
//!
//! Native operations are split into image-to-image/upload and buffer transfers;
//! Core continues to provide resource-state ordering.

use super::{Metal, MetalTexture};
use fluxel_rhi_core::api::{self, TextureRegion};

mod buffer;
mod texture;

pub(super) use buffer::{buffer_to_texture, texture_to_buffer};
pub(super) use texture::{copy_texture, write_texture};

pub(super) fn validate_region(
    texture: &MetalTexture,
    region: &TextureRegion<'_, Metal>,
) -> api::Result<()> {
    validate_region_fields(
        texture.desc(),
        region.mip,
        region.origin,
        region.extent,
        region.aspect,
    )
}

fn validate_region_fields(
    desc: &api::TextureDesc,
    mip: u32,
    origin: [u32; 3],
    extent: [u32; 3],
    aspect: api::TextureAspects,
) -> api::Result<()> {
    let expected_aspect = if desc.format == api::TextureFormat::Depth32Float {
        api::TextureAspects::DEPTH
    } else {
        api::TextureAspects::COLOR
    };
    let mip_extent = [
        (desc.extent[0] >> mip.min(31)).max(1),
        (desc.extent[1] >> mip.min(31)).max(1),
    ];
    if desc.dimension != api::TextureDimension::D2
        || aspect != expected_aspect
        || mip >= desc.mip_levels
        || origin[2] >= desc.array_layers
        || extent[0] == 0
        || extent[1] == 0
        || extent[2] != 1
        || origin[0]
            .checked_add(extent[0])
            .is_none_or(|end| end > mip_extent[0])
        || origin[1]
            .checked_add(extent[1])
            .is_none_or(|end| end > mip_extent[1])
    {
        return Err(api::Error::InvalidRange);
    }
    Ok(())
}

pub(super) fn bytes_per_texel(format: api::TextureFormat) -> Option<u32> {
    Some(match format {
        api::TextureFormat::R8Unorm => 1,
        api::TextureFormat::Rg8Unorm => 2,
        api::TextureFormat::Rgba8Unorm
        | api::TextureFormat::Rgba8UnormSrgb
        | api::TextureFormat::Bgra8Unorm
        | api::TextureFormat::Bgra8UnormSrgb => 4,
        api::TextureFormat::Rgba16Float => 8,
        _ => return None,
    })
}

pub(super) fn validate_copy_layout(
    layout: api::TexelCopyBufferLayout,
    tight_row: u32,
    rows: u32,
    buffer_size: u64,
) -> api::Result<()> {
    if rows == 0 || layout.bytes_per_row < tight_row || layout.rows_per_image < rows {
        return Err(api::Error::InvalidRange);
    }
    let end = u64::from(layout.bytes_per_row)
        .checked_mul(u64::from(rows - 1))
        .and_then(|offset| layout.offset.checked_add(offset))
        .and_then(|offset| offset.checked_add(u64::from(tight_row)))
        .ok_or(api::Error::InvalidRange)?;
    if end > buffer_size {
        return Err(api::Error::InvalidRange);
    }
    Ok(())
}

#[cfg(any(target_vendor = "apple", test))]
pub(super) fn align_up(value: usize, alignment: usize) -> Option<usize> {
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

#[cfg(any(target_vendor = "apple", test))]
pub(super) fn repack_rows(
    source: &[u8],
    layout: api::TexelCopyBufferLayout,
    tight_row: usize,
    rows: usize,
    target_row: usize,
) -> api::Result<Vec<u8>> {
    let source_rows_offset = u64::from(layout.bytes_per_row)
        .checked_mul(u64::try_from(rows.saturating_sub(1)).map_err(|_| api::Error::InvalidRange)?)
        .ok_or(api::Error::InvalidRange)?;
    let source_end = layout
        .offset
        .checked_add(source_rows_offset)
        .and_then(|end| end.checked_add(u64::try_from(tight_row).ok()?))
        .ok_or(api::Error::InvalidRange)?;
    if source_end > u64::try_from(source.len()).map_err(|_| api::Error::InvalidRange)?
        || u64::from(layout.bytes_per_row)
            < u64::try_from(tight_row).map_err(|_| api::Error::InvalidRange)?
        || target_row < tight_row
    {
        return Err(api::Error::InvalidRange);
    }
    let target_len = target_row
        .checked_mul(rows)
        .ok_or(api::Error::InvalidRange)?;
    let mut result = vec![0; target_len];
    for row in 0..rows {
        let row_offset = u64::from(layout.bytes_per_row)
            .checked_mul(u64::try_from(row).map_err(|_| api::Error::InvalidRange)?)
            .ok_or(api::Error::InvalidRange)?;
        let source_start = usize::try_from(
            layout
                .offset
                .checked_add(row_offset)
                .ok_or(api::Error::InvalidRange)?,
        )
        .map_err(|_| api::Error::InvalidRange)?;
        let target_start = row
            .checked_mul(target_row)
            .ok_or(api::Error::InvalidRange)?;
        result[target_start..target_start + tight_row]
            .copy_from_slice(&source[source_start..source_start + tight_row]);
    }
    Ok(result)
}

#[cfg(test)]
mod upload_tests {
    use super::{align_up, repack_rows};
    use fluxel_rhi_core::api::TexelCopyBufferLayout;

    #[test]
    fn upload_repacking_applies_offset_and_strips_source_padding() {
        let source = [99, 1, 2, 3, 4, 88, 5, 6, 7, 8, 77];
        let packed = repack_rows(
            &source,
            TexelCopyBufferLayout {
                offset: 1,
                bytes_per_row: 5,
                rows_per_image: 2,
            },
            4,
            2,
            8,
        )
        .unwrap();
        assert_eq!(packed, [1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0]);
    }

    #[test]
    fn upload_repacking_rejects_truncated_source_and_overflowing_stride() {
        assert!(
            repack_rows(
                &[1, 2, 3],
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: 4,
                    rows_per_image: 1,
                },
                4,
                1,
                4,
            )
            .is_err()
        );
        assert!(
            repack_rows(
                &[1, 2, 3, 4],
                TexelCopyBufferLayout {
                    offset: u64::MAX,
                    bytes_per_row: u32::MAX,
                    rows_per_image: 2,
                },
                4,
                2,
                4,
            )
            .is_err()
        );
    }

    #[test]
    fn row_alignment_rounds_without_overflow() {
        assert_eq!(align_up(513, 256), Some(768));
        assert_eq!(align_up(usize::MAX, 256), None);
    }
}

#[cfg(target_vendor = "apple")]
pub(super) fn same_texture(left: &MetalTexture, right: &MetalTexture) -> bool {
    std::ptr::eq(&*left.raw, &*right.raw)
}

#[cfg(not(target_vendor = "apple"))]
pub(super) fn same_texture(_: &MetalTexture, _: &MetalTexture) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::{validate_copy_layout, validate_region_fields};
    use fluxel_rhi_core::api::{
        TexelCopyBufferLayout, TextureAspects, TextureDesc, TextureDimension, TextureFormat,
        TextureUsage,
    };

    fn desc() -> TextureDesc {
        TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [64, 32, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 7,
            samples: 1,
            usage: TextureUsage(TextureUsage::COPY_SRC.0 | TextureUsage::COPY_DST.0),
            view_formats: vec![TextureFormat::Rgba8Unorm],
        }
    }

    #[test]
    fn texture_copy_region_accepts_valid_mip_and_array_slice() {
        assert!(
            validate_region_fields(&desc(), 2, [3, 2, 1], [8, 4, 1], TextureAspects::COLOR,)
                .is_ok()
        );
    }

    #[test]
    fn texture_copy_region_rejects_out_of_bounds_and_wrong_aspect() {
        let desc = desc();
        assert!(
            validate_region_fields(&desc, 0, [63, 0, 0], [2, 1, 1], TextureAspects::COLOR,)
                .is_err()
        );
        assert!(
            validate_region_fields(&desc, 0, [0, 0, 2], [1, 1, 1], TextureAspects::COLOR,).is_err()
        );
        assert!(
            validate_region_fields(&desc, 0, [0, 0, 0], [1, 1, 1], TextureAspects::DEPTH,).is_err()
        );
    }

    #[test]
    fn buffer_texture_layout_checks_row_pitch_offset_and_final_row_bounds() {
        let valid = api_layout(0, 4, 2);
        assert!(validate_copy_layout(valid, 4, 2, 9).is_ok());
        assert!(validate_copy_layout(valid, 5, 2, 9).is_err());
        assert!(validate_copy_layout(api_layout(8, 1, 4), 4, 2, 11).is_err());
        assert!(validate_copy_layout(api_layout(u64::MAX, 4, 2), 4, 2, u64::MAX).is_err());
    }

    fn api_layout(offset: u64, bytes_per_row: u32, rows_per_image: u32) -> TexelCopyBufferLayout {
        TexelCopyBufferLayout {
            offset,
            bytes_per_row,
            rows_per_image,
        }
    }
}
