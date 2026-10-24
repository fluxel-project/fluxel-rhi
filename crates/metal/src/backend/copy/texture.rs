//! Native Metal image copy and host upload operations.

#[cfg(target_vendor = "apple")]
use super::{align_up, repack_rows};
use super::{block, same_texture, validate_copy_images, validate_region};
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
        || source_desc.samples != target_desc.samples
        || src.aspect != dst.aspect
        || src.extent != dst.extent
        || (same_texture(source, target)
            && src.mip == dst.mip
            && (0..3).all(|i| {
                src.origin[i] < dst.origin[i] + dst.extent[i]
                    && dst.origin[i] < src.origin[i] + src.extent[i]
            }))
    {
        return Err(api::Error::Unsupported("Metal texture copy shape"));
    }

    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::{
            MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLOrigin,
            MTLSize,
        };

        if matches!(
            source_desc.format,
            api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8
        ) {
            return copy_combined_plane(encoder, source, target, &src, &dst);
        }
        let blit = encoder
            .command
            .blitCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: validate_region checked mip, array slice, origin and extent;
        // matching formats and sample counts; same-texture regions do not
        // overlap. The command buffer retains encoded objects.
        for layer in 0..src.extent[2] {
            let source_3d = source_desc.dimension == api::TextureDimension::D3;
            let target_3d = target_desc.dimension == api::TextureDimension::D3;
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &source.raw,
                if source_3d { 0 } else { (src.origin[2] + layer) as usize },
                src.mip as usize,
                MTLOrigin {
                    x: src.origin[0] as usize,
                    y: src.origin[1] as usize,
                    z: if source_3d { (src.origin[2] + layer) as usize } else { 0 },
                },
                MTLSize {
                    width: src.extent[0] as usize,
                    height: src.extent[1] as usize,
                    depth: 1,
                },
                &target.raw,
                if target_3d { 0 } else { (dst.origin[2] + layer) as usize },
                dst.mip as usize,
                MTLOrigin {
                    x: dst.origin[0] as usize,
                    y: dst.origin[1] as usize,
                    z: if target_3d { (dst.origin[2] + layer) as usize } else { 0 },
                },
            );
            }
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

// Metal texture-to-texture blits copy both planes of a combined format. Use
// the buffer blit aspect option to preserve the destination's other plane.
#[cfg(target_vendor = "apple")]
fn copy_combined_plane(
    encoder: &mut MetalCommandEncoder,
    source: &crate::backend::MetalTexture,
    target: &crate::backend::MetalTexture,
    src: &TextureRegion<'_, Metal>,
    dst: &TextureRegion<'_, Metal>,
) -> api::Result<()> {
    use objc2_metal::{
        MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLDevice as _,
        MTLOrigin, MTLResourceOptions, MTLSize,
    };
    if source.desc().samples != 1 {
        return super::multisample_plane::copy(encoder, source, target, src, dst);
    }
    let (_, _, bytes) = block(source.desc().format, src.aspect)?;
    let row = (src.extent[0] as usize)
        .checked_mul(bytes as usize)
        .ok_or(api::Error::InvalidRange)?;
    let pixel_format = crate::backend::texture::metal_format(source.desc().format)
        .ok_or(api::Error::InvalidRange)?;
    let row = align_up(
        row,
        encoder
            .command
            .device()
            .minimumLinearTextureAlignmentForPixelFormat(pixel_format)
            .max(1),
    )
    .ok_or(api::Error::InvalidRange)?;
    let size = row
        .checked_mul(src.extent[1] as usize)
        .ok_or(api::Error::InvalidRange)?;
    let staging = encoder
        .command
        .device()
        .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
        .ok_or(api::Error::OutOfMemory)?;
    let blit = encoder
        .command
        .blitCommandEncoder()
        .ok_or(api::Error::OutOfMemory)?;
    let option = super::blit_options(source.desc().format, src.aspect);
    for layer in 0..src.extent[2] {
        let sd = source.desc().dimension == api::TextureDimension::D3;
        let dd = target.desc().dimension == api::TextureDimension::D3;
        let so = MTLOrigin {
            x: src.origin[0] as usize,
            y: src.origin[1] as usize,
            z: if sd {
                (src.origin[2] + layer) as usize
            } else {
                0
            },
        };
        let dest = MTLOrigin {
            x: dst.origin[0] as usize,
            y: dst.origin[1] as usize,
            z: if dd {
                (dst.origin[2] + layer) as usize
            } else {
                0
            },
        };
        let extent = MTLSize {
            width: src.extent[0] as usize,
            height: src.extent[1] as usize,
            depth: 1,
        };
        // SAFETY: validated subresources and staging footprint bound both transfers;
        // ordered blit commands preserve the selected plane and retain resources.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(&source.raw, if sd {0} else {(src.origin[2]+layer) as usize}, src.mip as usize, so, extent, &staging, 0, row, size, option);
            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(&staging, 0, row, size, extent, &target.raw, if dd {0} else {(dst.origin[2]+layer) as usize}, dst.mip as usize, dest, option);
        }
    }
    blit.endEncoding();
    Ok(())
}

/// Resolve a color region through a temporary full-sized attachment. Metal
/// resolves whole attachments, so the final blit preserves pixels outside the
/// caller's destination region. A temporary multisample source preserves the
/// original even on GPU families lacking store-and-resolve.
pub(in crate::backend) fn resolve(
    encoder: &mut MetalCommandEncoder,
    src: TextureRegion<'_, Metal>,
    dst: TextureRegion<'_, Metal>,
) -> api::Result<()> {
    // SAFETY: Core verified device identity and both handles remain borrowed.
    let (source, target) = unsafe { (src.texture.native_ref(), dst.texture.native_ref()) };
    validate_region(source, &src)?;
    validate_region(target, &dst)?;
    let sd = source.desc();
    let dd = target.desc();
    if encoder.lane != 0
        || sd.dimension != api::TextureDimension::D2
        || dd.dimension != api::TextureDimension::D2
        || sd.samples <= 1
        || dd.samples != 1
        || sd.format != dd.format
        || src.aspect != api::TextureAspects::COLOR
        || dst.aspect != src.aspect
        || src.extent != dst.extent
        || !(sd.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
            || sd.usage.contains(api::TextureUsage::COPY_SRC))
        || !(dd.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
            || dd.usage.contains(api::TextureUsage::COPY_DST))
    {
        return Err(api::Error::InvalidRange);
    }
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::{
            MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _,
            MTLDevice as _, MTLLoadAction, MTLOrigin, MTLRenderPassDescriptor, MTLSize,
            MTLStorageMode, MTLStoreAction, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
        };
        if encoder.render.is_some() || encoder.compute.is_some() {
            return Err(api::Error::InvalidState);
        }
        let device = encoder.command.device();
        if !super::device_color_resolve_supported(&device, sd.format) {
            return Err(api::Error::Unsupported(
                "Metal device does not support resolve for this color format",
            ));
        }
        let format =
            crate::backend::texture::metal_format(sd.format).ok_or(api::Error::InvalidRange)?;
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setPixelFormat(format);
        descriptor.setTextureType(MTLTextureType::Type2DMultisample);
        // SAFETY: dimensions and sample count have already been validated at creation.
        unsafe {
            descriptor.setWidth(sd.extent[0] as usize);
            descriptor.setHeight(sd.extent[1] as usize);
            descriptor.setSampleCount(sd.samples as usize);
        }
        descriptor.setStorageMode(MTLStorageMode::Private);
        descriptor.setUsage(MTLTextureUsage::RenderTarget);
        let msaa = device
            .newTextureWithDescriptor(&descriptor)
            .ok_or(api::Error::OutOfMemory)?;
        descriptor.setTextureType(MTLTextureType::Type2D);
        // SAFETY: one sample is a valid single-sample texture descriptor.
        unsafe {
            descriptor.setSampleCount(1);
        }
        let resolved = device
            .newTextureWithDescriptor(&descriptor)
            .ok_or(api::Error::OutOfMemory)?;
        for layer in 0..src.extent[2] {
            let blit = encoder
                .command
                .blitCommandEncoder()
                .ok_or(api::Error::OutOfMemory)?;
            // SAFETY: the source slice and full extent are valid and scratch has the same format and sample count.
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(&source.raw, (src.origin[2] + layer) as usize, src.mip as usize, MTLOrigin {x:0,y:0,z:0}, MTLSize {width: sd.extent[0] as usize,height: sd.extent[1] as usize,depth:1}, &msaa,0,0,MTLOrigin {x:0,y:0,z:0});
            }
            blit.endEncoding();
            let pass = MTLRenderPassDescriptor::renderPassDescriptor();
            // SAFETY: Metal exposes eight color attachment slots.
            let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            color.setTexture(Some(&msaa));
            color.setResolveTexture(Some(&resolved));
            color.setLoadAction(MTLLoadAction::Load);
            color.setStoreAction(MTLStoreAction::MultisampleResolve);
            let render = encoder
                .command
                .renderCommandEncoderWithDescriptor(&pass)
                .ok_or(api::Error::OutOfMemory)?;
            render.endEncoding();
            let blit = encoder
                .command
                .blitCommandEncoder()
                .ok_or(api::Error::OutOfMemory)?;
            // SAFETY: validated rectangles fit both scratch and destination, whose formats match.
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(&resolved,0,0,MTLOrigin {x:src.origin[0] as usize,y:src.origin[1] as usize,z:0}, MTLSize {width:src.extent[0] as usize,height:src.extent[1] as usize,depth:1}, &target.raw,(dst.origin[2]+layer) as usize,dst.mip as usize,MTLOrigin {x:dst.origin[0] as usize,y:dst.origin[1] as usize,z:0});
            }
            blit.endEncoding();
        }
        Ok(())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (encoder, source, target);
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
    if !desc.usage.contains(api::TextureUsage::COPY_DST) || desc.samples != 1 {
        return Err(api::Error::Unsupported("Metal texture upload shape"));
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
        u64::try_from(bytes.len()).map_err(|_| api::Error::InvalidRange)?,
    )?;

    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::{
            MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _,
            MTLDevice as _, MTLOrigin, MTLSize,
        };

        let pixel_format = crate::backend::texture::metal_format(desc.format)
            .ok_or(api::Error::Unsupported("Metal texture upload format"))?;
        let alignment = encoder
            .command
            .device()
            .minimumLinearTextureAlignmentForPixelFormat(pixel_format)
            .max(1)
            .checked_next_power_of_two()
            .ok_or(api::Error::InvalidRange)?;
        let native_row = align_up(tight_row as usize, alignment).ok_or(api::Error::InvalidRange)?;
        let rows = dst.extent[1].div_ceil(bh) as usize;
        let staging_bytes = native_row
            .checked_mul(rows)
            .ok_or(api::Error::InvalidRange)?;
        let blit = encoder
            .command
            .blitCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        for layer in 0..dst.extent[2] {
            let image_offset = u64::from(layer)
                * u64::from(layout.bytes_per_row)
                * u64::from(layout.rows_per_image);
            let image_layout = api::TexelCopyBufferLayout {
                offset: layout.offset + image_offset,
                ..layout
            };
            let staging_data =
                repack_rows(bytes, image_layout, tight_row as usize, rows, native_row)?;
            let staging = encoder.upload_pool.write(
                &encoder.command.device(),
                &mut encoder.upload_slices,
                &staging_data,
                alignment,
            )?;
            // SAFETY: region validation bounds the image coordinates, the staging
            // buffer is fully populated, and row/image strides satisfy Metal's
            // pixel-format alignment requirement. The command buffer retains both.
            unsafe {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                &staging.buffer,
                staging.offset,
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
