//! Shared validation for direct Metal blit operations.
//!
//! Native operations are split into image-to-image/upload and buffer transfers;
//! Core continues to provide resource-state ordering.

use super::{Metal, MetalTexture};
use fluxel_rhi_core::api::{self, TextureRegion};

mod buffer;
#[cfg(target_vendor = "apple")]
mod byte_copy;
#[cfg(target_vendor = "apple")]
pub(super) use byte_copy::copy_bytes;
#[cfg(target_vendor = "apple")]
mod multisample_plane;
mod texture;

pub(super) use buffer::{buffer_to_texture, texture_to_buffer};
pub(super) use texture::{copy_texture, resolve, write_texture};

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
    if mip >= desc.mip_levels || extent.contains(&0) {
        return Err(api::Error::InvalidRange);
    }
    let (bw, bh, _) = block(desc.format, aspect)?;
    let bounds = [
        (desc.extent[0] >> mip.min(31)).max(1),
        (desc.extent[1] >> mip.min(31)).max(1),
        if desc.dimension == api::TextureDimension::D3 {
            (desc.extent[2] >> mip.min(31)).max(1)
        } else {
            desc.array_layers
        },
    ];
    if (0..3).any(|i| {
        origin[i]
            .checked_add(extent[i])
            .is_none_or(|end| end > bounds[i])
    }) {
        return Err(api::Error::InvalidRange);
    }
    for (i, unit) in [(0, bw), (1, bh)] {
        if origin[i] % unit != 0 || (extent[i] % unit != 0 && origin[i] + extent[i] != bounds[i]) {
            return Err(api::Error::InvalidRange);
        }
    }
    Ok(())
}

pub(super) fn block(
    format: api::TextureFormat,
    aspect: api::TextureAspects,
) -> api::Result<(u32, u32, u32)> {
    use api::TextureFormat::*;
    let bytes = match (format, aspect) {
        (Stencil8 | Depth24PlusStencil8 | Depth32FloatStencil8, api::TextureAspects::STENCIL) => 1,
        (Depth16Unorm, api::TextureAspects::DEPTH) => 2,
        (Depth32Float | Depth32FloatStencil8, api::TextureAspects::DEPTH) => 4,
        // The Vulkan formats chosen for Depth24Plus use a 32-bit packed depth
        // plane. Other backends may choose a different native representation.
        (Depth24Plus | Depth24PlusStencil8, api::TextureAspects::DEPTH) => 4,
        (R8Unorm | R8Snorm | R8Uint | R8Sint, api::TextureAspects::COLOR) => 1,
        (
            Rg8Unorm | Rg8Snorm | Rg8Uint | Rg8Sint | R16Unorm | R16Snorm | R16Uint | R16Sint
            | R16Float,
            api::TextureAspects::COLOR,
        ) => 2,
        (
            Rgba8Unorm | Rgba8UnormSrgb | Rgba8Snorm | Rgba8Uint | Rgba8Sint | Bgra8Unorm
            | Bgra8UnormSrgb | Rg16Unorm | Rg16Snorm | Rg16Uint | Rg16Sint | Rg16Float | R32Uint
            | R32Sint | R32Float | Rgb10a2Unorm | Rgb10a2Uint | Rg11b10Ufloat | Rgb9e5Ufloat,
            api::TextureAspects::COLOR,
        ) => 4,
        (
            Rgba16Unorm | Rgba16Snorm | Rgba16Uint | Rgba16Sint | Rgba16Float | Rg32Uint | Rg32Sint
            | Rg32Float | R64Uint,
            api::TextureAspects::COLOR,
        ) => 8,
        (Rgba32Uint | Rgba32Sint | Rgba32Float | Rg64Uint, api::TextureAspects::COLOR) => 16,
        (
            Bc1RgbaUnorm | Bc1RgbaUnormSrgb | Bc4RUnorm | Bc4RSnorm | Etc2Rgb8Unorm
            | Etc2Rgb8UnormSrgb | Etc2Rgb8A1Unorm | Etc2Rgb8A1UnormSrgb | EacR11Unorm | EacR11Snorm,
            api::TextureAspects::COLOR,
        ) => return Ok((4, 4, 8)),
        (
            Bc2RgbaUnorm | Bc2RgbaUnormSrgb | Bc3RgbaUnorm | Bc3RgbaUnormSrgb | Bc5RgUnorm
            | Bc5RgSnorm | Bc6hRgbUfloat | Bc6hRgbFloat | Bc7RgbaUnorm | Bc7RgbaUnormSrgb
            | Etc2Rgba8Unorm | Etc2Rgba8UnormSrgb | EacRg11Unorm | EacRg11Snorm,
            api::TextureAspects::COLOR,
        ) => return Ok((4, 4, 16)),
        (Astc { block, .. }, api::TextureAspects::COLOR) if block[0] != 0 && block[1] != 0 => {
            return Ok((block[0] as u32, block[1] as u32, 16));
        }
        _ => return Err(api::Error::InvalidRange),
    };
    Ok((1, 1, bytes))
}

// Apple Metal feature tables list resolve separately from render and MSAA.
// Integer formats do not have fixed-function resolve, while 32-bit float and
// 16-bit normalized formats acquire it on newer Apple families or Mac GPUs.
pub(super) fn color_resolve_supported(
    format: api::TextureFormat,
    apple4: bool,
    apple9: bool,
    mac: bool,
) -> bool {
    use api::TextureFormat::*;
    match format {
        R8Unorm | R8Snorm | Rg8Unorm | Rg8Snorm | Rgba8Unorm | Rgba8UnormSrgb | Rgba8Snorm
        | Bgra8Unorm | Bgra8UnormSrgb | R16Float | Rg16Float | Rgba16Float | Rgb10a2Unorm
        | Rg11b10Ufloat => true,
        R16Unorm | R16Snorm | Rg16Unorm | Rg16Snorm | Rgba16Unorm | Rgba16Snorm => apple4 || mac,
        R32Float | Rg32Float | Rgba32Float => apple9 || mac,
        Rgb9e5Ufloat => apple4 || !mac,
        _ => false,
    }
}

#[cfg(target_vendor = "apple")]
pub(super) fn device_color_resolve_supported(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
    format: api::TextureFormat,
) -> bool {
    use objc2::runtime::NSObjectProtocol as _;
    use objc2_metal::{MTLDevice as _, MTLGPUFamily};
    let float32_resolve = device.supportsFamily(MTLGPUFamily::Apple9)
        || (device.respondsToSelector(objc2::sel!(supports32BitFloatFiltering))
            && device.supports32BitFloatFiltering());
    color_resolve_supported(
        format,
        device.supportsFamily(MTLGPUFamily::Apple4),
        float32_resolve,
        device.supportsFamily(MTLGPUFamily::Mac1),
    )
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

#[cfg(target_vendor = "apple")]
pub(super) fn blit_options(
    format: api::TextureFormat,
    aspect: api::TextureAspects,
) -> objc2_metal::MTLBlitOption {
    if matches!(
        format,
        api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8
    ) {
        if aspect == api::TextureAspects::DEPTH {
            objc2_metal::MTLBlitOption::DepthFromDepthStencil
        } else {
            objc2_metal::MTLBlitOption::StencilFromDepthStencil
        }
    } else {
        objc2_metal::MTLBlitOption::None
    }
}

pub(super) fn validate_copy_images(
    layout: api::TexelCopyBufferLayout,
    tight_row: u32,
    rows: u32,
    images: u32,
    size: u64,
) -> api::Result<()> {
    if images == 0 {
        return Err(api::Error::InvalidRange);
    }
    let stride = u64::from(layout.bytes_per_row)
        .checked_mul(u64::from(layout.rows_per_image))
        .ok_or(api::Error::InvalidRange)?;
    let offset = layout
        .offset
        .checked_add(
            stride
                .checked_mul(u64::from(images - 1))
                .ok_or(api::Error::InvalidRange)?,
        )
        .ok_or(api::Error::InvalidRange)?;
    validate_copy_layout(
        api::TexelCopyBufferLayout { offset, ..layout },
        tight_row,
        rows,
        size,
    )
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

    #[test]
    fn resolve_capability_distinguishes_integer_and_family_limits() {
        use super::color_resolve_supported as supports;
        assert!(!supports(TextureFormat::Rgba8Uint, true, true, true));
        assert!(supports(TextureFormat::Rgba8Unorm, false, false, false));
        assert!(!supports(TextureFormat::R32Float, true, false, false));
        assert!(supports(TextureFormat::R32Float, true, true, false));
        assert!(!supports(TextureFormat::Rg16Unorm, false, false, false));
        assert!(supports(TextureFormat::Rg16Unorm, true, false, false));
        assert!(supports(TextureFormat::Rgb9e5Ufloat, true, false, true));
        assert!(!supports(TextureFormat::Rgb9e5Ufloat, false, false, true));
    }

    #[test]
    fn validates_compressed_blocks_and_multilayer_footprints() {
        let mut texture = desc();
        texture.format = TextureFormat::Bc1RgbaUnorm;
        assert!(
            validate_region_fields(&texture, 0, [4, 4, 0], [8, 8, 2], TextureAspects::COLOR)
                .is_ok()
        );
        assert!(
            validate_region_fields(&texture, 0, [1, 0, 0], [4, 4, 1], TextureAspects::COLOR)
                .is_err()
        );
        let layout = api_layout(0, 16, 3);
        assert!(super::validate_copy_images(layout, 16, 2, 2, 80).is_ok());
        assert!(super::validate_copy_images(layout, 16, 2, 2, 79).is_err());
        texture.dimension = TextureDimension::D3;
        texture.extent[2] = 8;
        texture.array_layers = 1;
        assert!(
            validate_region_fields(&texture, 1, [0, 0, 1], [4, 4, 3], TextureAspects::COLOR)
                .is_ok()
        );
        assert!(
            validate_region_fields(&texture, 1, [0, 0, 2], [4, 4, 3], TextureAspects::COLOR)
                .is_err()
        );
    }

    fn api_layout(offset: u64, bytes_per_row: u32, rows_per_image: u32) -> TexelCopyBufferLayout {
        TexelCopyBufferLayout {
            offset,
            bytes_per_row,
            rows_per_image,
        }
    }
}
