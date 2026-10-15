//! Native Metal image copy and host upload operations.

#[cfg(target_vendor = "apple")]
use super::{align_up, repack_rows};
use super::{bytes_per_texel, same_texture, validate_copy_layout, validate_region};
use crate::backend::{Metal, MetalCommandEncoder};
use fluxel_rhi_core::api::{self, TextureRegion};

pub(in crate::backend) fn copy_texture(
    encoder: &mut MetalCommandEncoder,
    src: TextureRegion<'_, Metal>,
    dst: TextureRegion<'_, Metal>,
) -> api::Result<()> {
    // SAFETY: the typed command API checked device identity before entering
    // this backend, and both texture handles remain borrowed through encoding.
    let source = unsafe { src.texture.native_ref() };
    // SAFETY: same owner-device guarantee as the source region above.
    let target = unsafe { dst.texture.native_ref() };
    validate_region(source, &src)?;
    validate_region(target, &dst)?;

    let source_desc = source.desc();
    let target_desc = target.desc();
    if !source_desc.usage.contains(api::TextureUsage::COPY_SRC)
        || !target_desc.usage.contains(api::TextureUsage::COPY_DST)
        || source_desc.format != target_desc.format
        || source_desc.samples != 1
        || target_desc.samples != 1
        || src.aspect != dst.aspect
        || src.extent != dst.extent
        || same_texture(source, target)
    {
        return Err(api::Error::Unsupported("Metal texture copy shape"));
    }

    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::{
            MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLOrigin,
            MTLSize,
        };

        let blit = encoder
            .command
            .blitCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: validate_region checked mip, array slice, origin and extent;
        // source and destination are distinct, single-sample textures with
        // matching formats/usages. The command buffer retains encoded objects.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &source.raw,
                src.origin[2] as usize,
                src.mip as usize,
                MTLOrigin {
                    x: src.origin[0] as usize,
                    y: src.origin[1] as usize,
                    z: 0,
                },
                MTLSize {
                    width: src.extent[0] as usize,
                    height: src.extent[1] as usize,
                    depth: 1,
                },
                &target.raw,
                dst.origin[2] as usize,
                dst.mip as usize,
                MTLOrigin {
                    x: dst.origin[0] as usize,
                    y: dst.origin[1] as usize,
                    z: 0,
                },
            );
        }
        blit.endEncoding();
        Ok(())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = encoder;
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

pub(in crate::backend) fn write_texture(
    encoder: &mut MetalCommandEncoder,
    dst: TextureRegion<'_, Metal>,
    layout: api::TexelCopyBufferLayout,
    bytes: &[u8],
) -> api::Result<()> {
    // SAFETY: Core checked that the destination texture belongs to the
    // encoder's device; the borrowed handle remains live through encoding.
    let target = unsafe { dst.texture.native_ref() };
    validate_region(target, &dst)?;
    let desc = target.desc();
    if !desc.usage.contains(api::TextureUsage::COPY_DST)
        || desc.samples != 1
        || dst.aspect != api::TextureAspects::COLOR
    {
        return Err(api::Error::Unsupported("Metal texture upload shape"));
    }
    let pixel_bytes = bytes_per_texel(desc.format)
        .ok_or(api::Error::Unsupported("Metal texture upload format"))?;
    let tight_row = dst.extent[0]
        .checked_mul(pixel_bytes)
        .ok_or(api::Error::InvalidRange)?;
    validate_copy_layout(
        layout,
        tight_row,
        dst.extent[1],
        u64::try_from(bytes.len()).map_err(|_| api::Error::InvalidRange)?,
    )?;

    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::{
            MTLBlitCommandEncoder as _, MTLBuffer as _, MTLCommandBuffer as _,
            MTLCommandEncoder as _, MTLDevice as _, MTLOrigin, MTLResourceOptions, MTLSize,
        };

        let pixel_format = crate::backend::texture::metal_format(desc.format)
            .ok_or(api::Error::Unsupported("Metal texture upload format"))?;
        let alignment = encoder
            .command
            .device()
            .minimumLinearTextureAlignmentForPixelFormat(pixel_format)
            .max(1);
        let native_row = align_up(tight_row as usize, alignment).ok_or(api::Error::InvalidRange)?;
        let staging_bytes = native_row
            .checked_mul(dst.extent[1] as usize)
            .ok_or(api::Error::InvalidRange)?;
        let staging_data = repack_rows(
            bytes,
            layout,
            tight_row as usize,
            dst.extent[1] as usize,
            native_row,
        )?;
        let staging = encoder
            .command
            .device()
            .newBufferWithLength_options(staging_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: the shared staging allocation has `staging_bytes` capacity,
        // equal to the fully initialized repacked vector length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                staging_data.as_ptr(),
                staging.contents().cast::<u8>().as_ptr(),
                staging_bytes,
            );
        }
        let blit = encoder
            .command
            .blitCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: region validation bounds the image coordinates, the staging
        // buffer is fully populated, and row/image strides satisfy Metal's
        // pixel-format alignment requirement. The command buffer retains both.
        unsafe {
            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &staging,
                0,
                native_row,
                staging_bytes,
                MTLSize {
                    width: dst.extent[0] as usize,
                    height: dst.extent[1] as usize,
                    depth: 1,
                },
                &target.raw,
                dst.origin[2] as usize,
                dst.mip as usize,
                MTLOrigin {
                    x: dst.origin[0] as usize,
                    y: dst.origin[1] as usize,
                    z: 0,
                },
            );
        }
        blit.endEncoding();
        Ok(())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (encoder, bytes);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}
