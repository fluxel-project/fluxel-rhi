//! Native Metal buffer-to-image and image-to-buffer transfer operations.

#[cfg(target_vendor = "apple")]
use super::align_up;
use super::{block, validate_copy_images, validate_region};
use crate::backend::{Metal, MetalBuffer, MetalCommandEncoder};
use fluxel_rhi_core::api::{self, TextureRegion};

/// Upload host bytes through an encoder-local slice of a shared staging page.
/// The caller's `UploadSlice` retention becomes command-buffer completion
/// retention when recording is finished.
pub(in crate::backend) fn write_buffer(
    encoder: &mut MetalCommandEncoder,
    dst: &MetalBuffer,
    offset: u64,
    bytes: &[u8],
) -> api::Result<()> {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::{
            MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _,
        };
        if encoder.render.is_some() || encoder.compute.is_some() {
            return Err(api::Error::InvalidState);
        }
        if !dst.usage.contains(api::BufferUsage::COPY_DST)
            || offset
                .checked_add(bytes.len() as u64)
                .is_none_or(|end| end > dst.size())
        {
            return Err(api::Error::InvalidRange);
        }
        if bytes.is_empty() {
            return Ok(());
        }
        let staging = encoder.upload_pool.write(
            &encoder.command.device(),
            &mut encoder.upload_slices,
            bytes,
            4,
        )?;
        if offset & 3 != 0 || bytes.len() & 3 != 0 {
            let destination = usize::try_from(offset).map_err(|_| api::Error::InvalidRange)?;
            return super::byte_copy::copy_bytes(
                &encoder.command,
                &staging.buffer,
                staging.offset,
                &dst.raw,
                destination,
                bytes.len(),
            );
        }
        let blit = encoder
            .command
            .blitCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: the retained page range and destination range were validated.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &staging.buffer,
                staging.offset,
                &dst.raw,
                offset as usize,
                bytes.len(),
            );
        }
        blit.endEncoding();
        Ok(())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (encoder, dst, offset, bytes);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

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
    if !desc.usage.contains(api::TextureUsage::COPY_DST) || desc.samples != 1 {
        return Err(api::Error::Unsupported("Metal buffer-to-texture shape"));
    }
    let (bw, bh, pixel_bytes) = block(desc.format, dst.aspect)?;
    let tight_row = dst.extent[0]
        .div_ceil(bw)
        .checked_mul(pixel_bytes)
        .ok_or(api::Error::InvalidRange)?;
    validate_copy_images(
        layout,
        tight_row,
        dst.extent[1].div_ceil(bh),
        dst.extent[2],
        src.size(),
    )?;

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
        let rows = dst.extent[1].div_ceil(bh) as usize;
        let staging_bytes = native_row
            .checked_mul(rows)
            .ok_or(api::Error::InvalidRange)?;
        for layer in 0..dst.extent[2] {
            let staging = encoder
                .command
                .device()
                .newBufferWithLength_options(staging_bytes, MTLResourceOptions::StorageModeShared)
                .ok_or(api::Error::OutOfMemory)?;
            let source_offset = layout.offset as usize
                + layer as usize * layout.rows_per_image as usize * layout.bytes_per_row as usize;
            super::byte_copy::copy_rows(
                &encoder.command,
                &src.raw,
                source_offset,
                layout.bytes_per_row as usize,
                &staging,
                0,
                native_row,
                tight_row as usize,
                rows,
            )?;
            let blit = encoder
                .command
                .blitCommandEncoder()
                .ok_or(api::Error::OutOfMemory)?; // SAFETY: the staging allocation has aligned row/image strides, every
            // texel is initialized, and validate_region bounded the destination.
            unsafe {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
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
                if desc.dimension == api::TextureDimension::D3 { 0 } else { (dst.origin[2] + layer) as usize },
                dst.mip as usize,
                MTLOrigin {
                    x: dst.origin[0] as usize,
                    y: dst.origin[1] as usize,
                    z: if desc.dimension == api::TextureDimension::D3 { (dst.origin[2] + layer) as usize } else { 0 },
                },
            super::blit_options(desc.format, dst.aspect),
            );
            }
            blit.endEncoding();
        }
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
    if !desc.usage.contains(api::TextureUsage::COPY_SRC) || desc.samples != 1 {
        return Err(api::Error::Unsupported("Metal texture readback shape"));
    }
    #[cfg(target_vendor = "apple")]
    if !dst.usage.contains(api::BufferUsage::COPY_DST) {
        return Err(api::Error::Unsupported(
            "Metal texture readback buffer usage",
        ));
    }
    let (bw, bh, pixel_bytes) = block(desc.format, src.aspect)?;
    let tight_row = src.extent[0]
        .div_ceil(bw)
        .checked_mul(pixel_bytes)
        .ok_or(api::Error::InvalidRange)?;
    validate_copy_images(
        layout,
        tight_row,
        src.extent[1].div_ceil(bh),
        src.extent[2],
        dst.size(),
    )?;

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
        let rows = src.extent[1].div_ceil(bh) as usize;
        let staging_bytes = native_row
            .checked_mul(rows)
            .ok_or(api::Error::InvalidRange)?;
        // SAFETY: image region bounds, slice, row stride and staging size were
        // checked above; the command buffer retains both Metal resources.
        for layer in 0..src.extent[2] {
            let staging = encoder
                .command
                .device()
                .newBufferWithLength_options(staging_bytes, MTLResourceOptions::StorageModeShared)
                .ok_or(api::Error::OutOfMemory)?;
            let blit = encoder
                .command
                .blitCommandEncoder()
                .ok_or(api::Error::OutOfMemory)?;
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                &source.raw,
                if desc.dimension == api::TextureDimension::D3 { 0 } else { (src.origin[2] + layer) as usize },
                src.mip as usize,
                MTLOrigin {
                    x: src.origin[0] as usize,
                    y: src.origin[1] as usize,
                    z: if desc.dimension == api::TextureDimension::D3 { (src.origin[2] + layer) as usize } else { 0 },
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
            super::blit_options(desc.format, src.aspect),
            );
            }
            blit.endEncoding();
            let destination = layout.offset as usize
                + layer as usize * layout.rows_per_image as usize * layout.bytes_per_row as usize;
            super::byte_copy::copy_rows(
                &encoder.command,
                &staging,
                0,
                native_row,
                &dst.raw,
                destination,
                layout.bytes_per_row as usize,
                tight_row as usize,
                rows,
            )?;
        }
        Ok(())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (encoder, dst);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}
