//! Native Metal buffer-to-image and image-to-buffer transfer operations.

#[cfg(target_vendor = "apple")]
use super::align_up;
use super::{bytes_per_texel, validate_copy_layout, validate_region};
use crate::backend::{Metal, MetalBuffer, MetalCommandEncoder};
use fluxel_rhi_core::api::{self, TextureRegion};

pub(in crate::backend) fn buffer_to_texture(
    encoder: &mut MetalCommandEncoder,
    src: &MetalBuffer,
    layout: api::TexelCopyBufferLayout,
    dst: TextureRegion<'_, Metal>,
) -> api::Result<()> {
    // SAFETY: Core checked the destination texture belongs to this device; the
    // borrowed source and target stay alive while native commands are encoded.
    let target = unsafe { dst.texture.native_ref() };
    validate_region(target, &dst)?;
    let desc = target.desc();
    if !desc.usage.contains(api::TextureUsage::COPY_DST)
        || desc.samples != 1
        || dst.aspect != api::TextureAspects::COLOR
    {
        return Err(api::Error::Unsupported("Metal buffer-to-texture shape"));
    }
    let pixel_bytes = bytes_per_texel(desc.format)
        .ok_or(api::Error::Unsupported("Metal buffer-to-texture format"))?;
    let tight_row = dst.extent[0]
        .checked_mul(pixel_bytes)
        .ok_or(api::Error::InvalidRange)?;
    validate_copy_layout(layout, tight_row, dst.extent[1], src.size())?;

    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::{
            MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _,
            MTLDevice as _, MTLOrigin, MTLResourceOptions, MTLSize,
        };
        if !src.usage.contains(api::BufferUsage::COPY_SRC) {
            return Err(api::Error::Unsupported(
                "Metal buffer-to-texture requires COPY_SRC",
            ));
        }
        let pixel_format = crate::backend::texture::metal_format(desc.format)
            .ok_or(api::Error::Unsupported("Metal buffer-to-texture format"))?;
        let alignment = encoder
            .command
            .device()
            .minimumLinearTextureAlignmentForPixelFormat(pixel_format)
            .max(1);
        let native_row = align_up(tight_row as usize, alignment).ok_or(api::Error::InvalidRange)?;
        let staging_bytes = native_row
            .checked_mul(dst.extent[1] as usize)
            .ok_or(api::Error::InvalidRange)?;
        let staging = encoder
            .command
            .device()
            .newBufferWithLength_options(staging_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(api::Error::OutOfMemory)?;
        let blit = encoder
            .command
            .blitCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        // First remove the caller's row padding into a Metal-aligned staging
        // layout. Each range was checked by validate_copy_layout above.
        for row in 0..dst.extent[1] as usize {
            let source_offset = layout.offset as usize + row * layout.bytes_per_row as usize;
            let staging_offset = row * native_row;
            // SAFETY: both source and staging row ranges fit their allocations;
            // the blit encoder retains the source and staging resources.
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &src.raw,
                    source_offset,
                    &staging,
                    staging_offset,
                    tight_row as usize,
                );
            }
        }
        // SAFETY: the staging allocation has aligned row/image strides, every
        // texel is initialized, and validate_region bounded the destination.
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
        let _ = (encoder, src);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

pub(in crate::backend) fn texture_to_buffer(
    encoder: &mut MetalCommandEncoder,
    src: TextureRegion<'_, Metal>,
    dst: &MetalBuffer,
    layout: api::TexelCopyBufferLayout,
) -> api::Result<()> {
    // SAFETY: Core checked source texture provenance; both resources stay
    // borrowed while the native command is encoded.
    let source = unsafe { src.texture.native_ref() };
    validate_region(source, &src)?;
    let desc = source.desc();
    if !desc.usage.contains(api::TextureUsage::COPY_SRC)
        || desc.samples != 1
        || src.aspect != api::TextureAspects::COLOR
    {
        return Err(api::Error::Unsupported("Metal texture readback shape"));
    }
    #[cfg(target_vendor = "apple")]
    if !dst.usage.contains(api::BufferUsage::COPY_DST) {
        return Err(api::Error::Unsupported(
            "Metal texture readback buffer usage",
        ));
    }
    let pixel_bytes = bytes_per_texel(desc.format)
        .ok_or(api::Error::Unsupported("Metal texture readback format"))?;
    let tight_row = src.extent[0]
        .checked_mul(pixel_bytes)
        .ok_or(api::Error::InvalidRange)?;
    validate_copy_layout(layout, tight_row, src.extent[1], dst.size())?;

    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::{
            MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _,
            MTLDevice as _, MTLOrigin, MTLResourceOptions, MTLSize,
        };

        let pixel_format = crate::backend::texture::metal_format(desc.format)
            .ok_or(api::Error::Unsupported("Metal texture readback format"))?;
        let alignment = encoder
            .command
            .device()
            .minimumLinearTextureAlignmentForPixelFormat(pixel_format)
            .max(1);
        let native_row = align_up(tight_row as usize, alignment).ok_or(api::Error::InvalidRange)?;
        let staging_bytes = native_row
            .checked_mul(src.extent[1] as usize)
            .ok_or(api::Error::InvalidRange)?;
        let staging = encoder
            .command
            .device()
            .newBufferWithLength_options(staging_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(api::Error::OutOfMemory)?;
        let blit = encoder
            .command
            .blitCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: image region bounds, slice, row stride and staging size were
        // checked above; the command buffer retains both Metal resources.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
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
                &staging,
                0,
                native_row,
                staging_bytes,
            );
            for row in 0..src.extent[1] as usize {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &staging,
                    row * native_row,
                    &dst.raw,
                    layout.offset as usize + row * layout.bytes_per_row as usize,
                    tight_row as usize,
                );
            }
        }
        blit.endEncoding();
        Ok(())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (encoder, dst);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}
