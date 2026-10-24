//! Native image copies and staging uploads. The caller owns explicit layouts.

mod region;

use super::super::{VulkanBuffer, VulkanEncoder, resource, upload};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use region::{buffer_region, extent, image_region, layers, offset};

pub(crate) fn copy_texture(
    e: &mut VulkanEncoder,
    src: api::TextureRegion<'_, super::super::Vulkan>,
    dst: api::TextureRegion<'_, super::super::Vulkan>,
) -> Result<()> {
    // SAFETY: Core validates ownership and the encoder retains both images.
    let (s, d) = unsafe { (src.texture.native_ref(), dst.texture.native_ref()) };
    image_region(s, &src)?;
    image_region(d, &dst)?;
    let sb = region::block(s.desc().format, src.aspect)?;
    let db = region::block(d.desc().format, dst.aspect)?;
    let formats_compatible = s.desc().format == d.desc().format
        || (src.aspect == api::TextureAspects::COLOR
            && dst.aspect == src.aspect
            && sb.2 == db.2
            && ((sb.0 == 1 && sb.1 == 1)
                || (db.0 == 1 && db.1 == 1)
                || (sb.0 == db.0 && sb.1 == db.1)));
    let extents_compatible = src.extent[0].div_ceil(sb.0) == dst.extent[0].div_ceil(db.0)
        && src.extent[1].div_ceil(sb.1) == dst.extent[1].div_ceil(db.1)
        && src.extent[2] == dst.extent[2]
        && ((sb.0, sb.1) != (db.0, db.1) || src.extent == dst.extent);
    let dimensions_compatible = s.desc().dimension == d.desc().dimension
        || e.device.maintenance5
        || matches!(
            (s.desc().dimension, d.desc().dimension),
            (api::TextureDimension::D2, api::TextureDimension::D3)
                | (api::TextureDimension::D3, api::TextureDimension::D2)
        );
    if !dimensions_compatible {
        return Err(Error::Unsupported("Vulkan maintenance5 image type copy"));
    }
    if !s.desc().usage.contains(api::TextureUsage::COPY_SRC)
        || !d.desc().usage.contains(api::TextureUsage::COPY_DST)
        || !formats_compatible
        || s.desc().samples != d.desc().samples
        || !extents_compatible
        || src.aspect != dst.aspect
        || same_subresource(s, &src, d, &dst)
    {
        return Err(Error::InvalidRange);
    }
    let region = vk::ImageCopy::default()
        .src_subresource(layers(s, &src))
        .src_offset(offset(s, &src))
        .dst_subresource(layers(d, &dst))
        .dst_offset(offset(d, &dst))
        .extent(
            if s.desc().dimension != d.desc().dimension
                && (s.desc().dimension == api::TextureDimension::D3
                    || d.desc().dimension == api::TextureDimension::D3)
            {
                vk::Extent3D {
                    width: src.extent[0],
                    height: src.extent[1],
                    depth: src.extent[2],
                }
            } else {
                extent(s, &src)
            },
        );
    // Vulkan requires depth one whenever either image is 1D. Map each array
    // layer to a separate 3D slice when maintenance5 permits this pairing.
    let split_1d_3d = (s.desc().dimension == api::TextureDimension::D1
        && d.desc().dimension == api::TextureDimension::D3)
        || (s.desc().dimension == api::TextureDimension::D3
            && d.desc().dimension == api::TextureDimension::D1);
    let mut regions = Vec::new();
    if split_1d_3d {
        for slice in 0..src.extent[2] {
            let mut part = region;
            part.extent.depth = 1;
            if s.desc().dimension == api::TextureDimension::D3 {
                part.src_offset.z += slice as i32;
                part.dst_subresource.base_array_layer += slice;
                part.dst_subresource.layer_count = 1;
            } else {
                part.dst_offset.z += slice as i32;
                part.src_subresource.base_array_layer += slice;
                part.src_subresource.layer_count = 1;
            }
            regions.push(part);
        }
    } else {
        regions.push(region);
    }
    // SAFETY: checked regions and explicit transfer layouts satisfy the native copy contract.
    unsafe {
        e.device.device.cmd_copy_image(
            e.command,
            s.native(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            d.native(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
    }
    e.textures.extend([s.clone(), d.clone()]);
    Ok(())
}

pub(crate) fn buffer_to_texture(
    e: &mut VulkanEncoder,
    src: &VulkanBuffer,
    layout: api::TexelCopyBufferLayout,
    dst: api::TextureRegion<'_, super::super::Vulkan>,
) -> Result<()> {
    // SAFETY: Core validates device identity; the encoder retains both objects.
    let d = unsafe { dst.texture.native_ref() };
    if !src.0.usage.contains(api::BufferUsage::COPY_SRC)
        || !d.desc().usage.contains(api::TextureUsage::COPY_DST)
        || d.desc().samples != 1
    {
        return Err(Error::InvalidRange);
    }
    let copy = buffer_region(d, &dst, layout, src.0.size)?;
    // SAFETY: buffer footprint and region were checked; caller selected CopyDst.
    unsafe {
        e.device.device.cmd_copy_buffer_to_image(
            e.command,
            src.0.buffer,
            d.native(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy],
        );
    }
    e.buffers.push(src.clone());
    e.textures.push(d.clone());
    Ok(())
}

pub(crate) fn texture_to_buffer(
    e: &mut VulkanEncoder,
    src: api::TextureRegion<'_, super::super::Vulkan>,
    dst: &VulkanBuffer,
    layout: api::TexelCopyBufferLayout,
) -> Result<()> {
    // SAFETY: Core validates device identity; the encoder retains both objects.
    let s = unsafe { src.texture.native_ref() };
    if !s.desc().usage.contains(api::TextureUsage::COPY_SRC)
        || !dst.0.usage.contains(api::BufferUsage::COPY_DST)
        || s.desc().samples != 1
    {
        return Err(Error::InvalidRange);
    }
    let copy = buffer_region(s, &src, layout, dst.0.size)?;
    // SAFETY: validated footprint and caller-selected CopySrc layout.
    unsafe {
        e.device.device.cmd_copy_image_to_buffer(
            e.command,
            s.native(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst.0.buffer,
            &[copy],
        );
    }
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(dst.0.buffer)
        .offset(layout.offset)
        .size(vk::WHOLE_SIZE);
    // SAFETY: same command buffer, after the copy; this makes host reads visible.
    unsafe {
        e.device.device.cmd_pipeline_barrier(
            e.command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
    e.textures.push(s.clone());
    e.buffers.push(dst.clone());
    Ok(())
}

pub(crate) fn write_texture(
    e: &mut VulkanEncoder,
    dst: api::TextureRegion<'_, super::super::Vulkan>,
    layout: api::TexelCopyBufferLayout,
    bytes: &[u8],
) -> Result<()> {
    // SAFETY: Core validates ownership and the image is retained until completion.
    let d = unsafe { dst.texture.native_ref() };
    if !d.desc().usage.contains(api::TextureUsage::COPY_DST) || d.desc().samples != 1 {
        return Err(Error::InvalidRange);
    }
    let family = if e.lane.ordinal == 1 {
        e.device.compute_family
    } else {
        e.device.graphics_family
    };
    let upload = e.device.upload_pool.write(
        &e.device,
        family,
        &mut e.upload_slices,
        bytes,
        upload::copy_offset_alignment(&e.device),
        layout.offset,
    )?;
    let copy_layout = api::TexelCopyBufferLayout {
        offset: upload
            .offset
            .checked_add(layout.offset)
            .ok_or(Error::InvalidRange)?,
        ..layout
    };
    let copy = buffer_region(
        d,
        &dst,
        copy_layout,
        upload
            .offset
            .checked_add(bytes.len() as u64)
            .ok_or(Error::InvalidRange)?,
    )?;
    let l = layers(d, &dst);
    let coupled = matches!(
        d.desc().format,
        api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8
    ) && !e.device.separate_depth_stencil_layouts;
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(if coupled {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        } else {
            l.aspect_mask
        })
        .base_mip_level(l.mip_level)
        .level_count(1)
        .base_array_layer(l.base_array_layer)
        .layer_count(if d.desc().dimension == api::TextureDimension::D3 {
            vk::REMAINING_ARRAY_LAYERS
        } else {
            l.layer_count
        });
    let barrier = |old, new, from, to| {
        vk::ImageMemoryBarrier::default()
            .src_access_mask(from)
            .dst_access_mask(to)
            .old_layout(old)
            .new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(d.native())
            .subresource_range(range)
    };
    let full = dst.origin[0] == 0
        && dst.origin[1] == 0
        && dst.extent[0] == (d.desc().extent[0] >> dst.mip).max(1)
        && dst.extent[1] == (d.desc().extent[1] >> dst.mip).max(1)
        && (d.desc().dimension != api::TextureDimension::D3
            || (dst.origin[2] == 0 && dst.extent[2] == (d.desc().extent[2] >> dst.mip).max(1)));
    let to_copy = barrier(
        if full && !coupled {
            vk::ImageLayout::UNDEFINED
        } else {
            vk::ImageLayout::GENERAL
        },
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        vk::AccessFlags::TRANSFER_WRITE,
    );
    let to_idle = barrier(
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::GENERAL,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
    );
    // SAFETY: full uploads discard the selected subresource; partial uploads
    // require its GENERAL idle layout to preserve untouched texels.
    unsafe {
        e.device.device.cmd_pipeline_barrier(
            e.command,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_copy],
        );
        e.device.device.cmd_copy_buffer_to_image(
            e.command,
            upload.buffer,
            d.native(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy],
        );
        e.device.device.cmd_pipeline_barrier(
            e.command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_idle],
        );
    }
    e.textures.push(d.clone());
    Ok(())
}

pub(crate) fn blit(
    e: &mut VulkanEncoder,
    src: api::TextureRegion<'_, super::super::Vulkan>,
    dst: api::TextureRegion<'_, super::super::Vulkan>,
    filter: api::FilterMode,
) -> Result<()> {
    // SAFETY: Core validates ownership and the encoder retains the images.
    let (s, d) = unsafe { (src.texture.native_ref(), dst.texture.native_ref()) };
    image_region(s, &src)?;
    image_region(d, &dst)?;
    if !s.desc().usage.contains(api::TextureUsage::COPY_SRC)
        || !d.desc().usage.contains(api::TextureUsage::COPY_DST)
        || s.desc().samples != 1
        || d.desc().samples != 1
        || s.desc().dimension != d.desc().dimension
        || src.aspect != dst.aspect
        || same_subresource(s, &src, d, &dst)
        || (s.desc().dimension != api::TextureDimension::D3 && src.extent[2] != dst.extent[2])
        || blit_class(s.desc().format) != blit_class(d.desc().format)
        || (src.aspect != api::TextureAspects::COLOR
            && (s.desc().format != d.desc().format || filter != api::FilterMode::Nearest))
        || (blit_class(s.desc().format) != 0 && filter != api::FilterMode::Nearest)
    {
        return Err(Error::InvalidRange);
    }
    let features = |format| {
        // SAFETY: the instance and physical device remain live with the encoder.
        unsafe {
            e.device
                .instance
                .instance
                .get_physical_device_format_properties(e.device.physical, format)
        }
        .optimal_tiling_features
    };
    let sf = features(
        resource::native_format(&e.device, s.desc().format)
            .ok_or(Error::Unsupported("Vulkan blit source format"))?,
    );
    let df = features(
        resource::native_format(&e.device, d.desc().format)
            .ok_or(Error::Unsupported("Vulkan blit destination format"))?,
    );
    if !sf.contains(vk::FormatFeatureFlags::BLIT_SRC)
        || !df.contains(vk::FormatFeatureFlags::BLIT_DST)
        || (filter == api::FilterMode::Linear
            && !sf.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR))
    {
        return Err(Error::Unsupported("Vulkan format blit capability"));
    }
    let end = |t: &super::super::VulkanTexture,
               r: &api::TextureRegion<'_, super::super::Vulkan>| {
        let o = offset(t, r);
        let x = extent(t, r);
        vk::Offset3D {
            x: o.x + x.width as i32,
            y: o.y + x.height as i32,
            z: o.z + x.depth as i32,
        }
    };
    let region = vk::ImageBlit::default()
        .src_subresource(layers(s, &src))
        .src_offsets([offset(s, &src), end(s, &src)])
        .dst_subresource(layers(d, &dst))
        .dst_offsets([offset(d, &dst), end(d, &dst)]);
    // SAFETY: checked regions and native feature flags; caller selected transfer layouts.
    unsafe {
        e.device.device.cmd_blit_image(
            e.command,
            s.native(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            d.native(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
            match filter {
                api::FilterMode::Nearest => vk::Filter::NEAREST,
                api::FilterMode::Linear => vk::Filter::LINEAR,
            },
        );
    }
    e.textures.extend([s.clone(), d.clone()]);
    Ok(())
}

fn blit_class(f: api::TextureFormat) -> u8 {
    use api::TextureFormat::*;
    match f {
        R8Uint | Rg8Uint | Rgba8Uint | R16Uint | Rg16Uint | Rgba16Uint | R32Uint | Rg32Uint
        | Rgba32Uint | Rgb10a2Uint | R64Uint | Rg64Uint => 1,
        R8Sint | Rg8Sint | Rgba8Sint | R16Sint | Rg16Sint | Rgba16Sint | R32Sint | Rg32Sint
        | Rgba32Sint => 2,
        _ => 0,
    }
}

fn same_subresource(
    s: &super::super::VulkanTexture,
    src: &api::TextureRegion<'_, super::super::Vulkan>,
    d: &super::super::VulkanTexture,
    dst: &api::TextureRegion<'_, super::super::Vulkan>,
) -> bool {
    s.native() == d.native()
        && src.mip == dst.mip
        && (s.desc().dimension == api::TextureDimension::D3
            || (src.origin[2] < dst.origin[2] + dst.extent[2]
                && dst.origin[2] < src.origin[2] + src.extent[2]))
}

pub(crate) fn resolve(
    e: &mut VulkanEncoder,
    src: api::TextureRegion<'_, super::super::Vulkan>,
    dst: api::TextureRegion<'_, super::super::Vulkan>,
) -> Result<()> {
    if e.lane.ordinal != 0 {
        return Err(Error::Unsupported("Vulkan resolve requires graphics queue"));
    }
    // SAFETY: Core validates ownership and both images are retained below.
    let (s, d) = unsafe { (src.texture.native_ref(), dst.texture.native_ref()) };
    image_region(s, &src)?;
    image_region(d, &dst)?;
    if s.desc().samples <= 1
        || d.desc().samples != 1
        || s.desc().format != d.desc().format
        || src.extent != dst.extent
        || src.aspect != api::TextureAspects::COLOR
        || dst.aspect != src.aspect
        || s.desc().dimension != api::TextureDimension::D2
        || d.desc().dimension != api::TextureDimension::D2
        || !(s
            .desc()
            .usage
            .contains(api::TextureUsage::RENDER_ATTACHMENT)
            || s.desc().usage.contains(api::TextureUsage::COPY_SRC))
        || !(d
            .desc()
            .usage
            .contains(api::TextureUsage::RENDER_ATTACHMENT)
            || d.desc().usage.contains(api::TextureUsage::COPY_DST))
        || s.native() == d.native()
    {
        return Err(Error::InvalidRange);
    }
    let region = vk::ImageResolve::default()
        .src_subresource(layers(s, &src))
        .src_offset(offset(s, &src))
        .dst_subresource(layers(d, &dst))
        .dst_offset(offset(d, &dst))
        .extent(extent(s, &src));
    // SAFETY: ResolveSrc/ResolveDst transitions establish GENERAL layout and the
    // required memory dependency. Both images and regions are retained.
    unsafe {
        e.device.device.cmd_resolve_image(
            e.command,
            s.native(),
            vk::ImageLayout::GENERAL,
            d.native(),
            vk::ImageLayout::GENERAL,
            &[region],
        );
    }
    e.textures.extend([s.clone(), d.clone()]);
    Ok(())
}
