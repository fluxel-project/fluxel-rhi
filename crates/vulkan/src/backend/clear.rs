//! Standalone attachment clear using dynamic rendering.
//!
//! The caller transitions each selected subresource to the matching attachment
//! layout before this command. Views survive until encoder completion.
use super::*;
use std::ops::Range;

pub(super) fn buffer(
    e: &mut VulkanEncoder,
    buffer: &VulkanBuffer,
    range: Range<u64>,
    value: u32,
) -> Result<()> {
    if e.raster || e.compute {
        return Err(Error::InvalidState);
    }
    if !buffer.0.usage.contains(api::BufferUsage::COPY_DST)
        || range.start >= range.end
        || range.end > buffer.0.size
        || range.start % 4 != 0
        || (range.end - range.start) % 4 != 0
    {
        return Err(Error::InvalidRange);
    }
    // SAFETY: Core checked device identity; the range and transfer usage are
    // valid, and the caller provides the required transfer write dependency.
    unsafe {
        e.device.device.cmd_fill_buffer(
            e.command,
            buffer.0.buffer,
            range.start,
            range.end - range.start,
            value,
        );
    }
    e.buffers.push(buffer.clone());
    Ok(())
}

pub(super) fn texture(
    e: &mut VulkanEncoder,
    texture: &VulkanTexture,
    mips: Range<u32>,
    layers: Range<u32>,
    value: api::TextureClearValue,
) -> Result<()> {
    let d = texture.desc();
    if e.raster
        || e.compute
        || e.lane.ordinal != 0
        || !d.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
    {
        return Err(Error::Unsupported(
            "Vulkan clear requires an idle encoder and attachment usage",
        ));
    }
    if mips.start >= mips.end
        || layers.start >= layers.end
        || mips.end > d.mip_levels
        || layers.end > d.array_layers
    {
        return Err(Error::InvalidRange);
    }
    let (depth, clear) = match (d.format, value) {
        (
            api::TextureFormat::Depth16Unorm
            | api::TextureFormat::Depth24Plus
            | api::TextureFormat::Depth24PlusStencil8
            | api::TextureFormat::Depth32Float
            | api::TextureFormat::Depth32FloatStencil8,
            api::TextureClearValue::Depth(v),
        ) if v.is_finite() && (0.0..=1.0).contains(&v) => (
            true,
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: v,
                    stencil: 0,
                },
            },
        ),
        (f, api::TextureClearValue::Color(api::ClearColor::Float(v)))
            if matches!(
                f,
                api::TextureFormat::R8Unorm
                    | api::TextureFormat::R8Snorm
                    | api::TextureFormat::Rg8Unorm
                    | api::TextureFormat::Rg8Snorm
                    | api::TextureFormat::Rgba8Unorm
                    | api::TextureFormat::Rgba8UnormSrgb
                    | api::TextureFormat::Rgba8Snorm
                    | api::TextureFormat::Bgra8Unorm
                    | api::TextureFormat::Bgra8UnormSrgb
                    | api::TextureFormat::R16Unorm
                    | api::TextureFormat::R16Snorm
                    | api::TextureFormat::R16Float
                    | api::TextureFormat::Rg16Unorm
                    | api::TextureFormat::Rg16Snorm
                    | api::TextureFormat::Rg16Float
                    | api::TextureFormat::Rgba16Unorm
                    | api::TextureFormat::Rgba16Snorm
                    | api::TextureFormat::Rgba16Float
                    | api::TextureFormat::R32Float
                    | api::TextureFormat::Rg32Float
                    | api::TextureFormat::Rgba32Float
                    | api::TextureFormat::Rgb10a2Unorm
                    | api::TextureFormat::Rg11b10Ufloat
                    | api::TextureFormat::Rgb9e5Ufloat
            ) =>
        {
            (
                false,
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: v },
                },
            )
        }
        (f, api::TextureClearValue::Color(api::ClearColor::Uint(v)))
            if matches!(
                f,
                api::TextureFormat::R8Uint
                    | api::TextureFormat::Rg8Uint
                    | api::TextureFormat::Rgba8Uint
                    | api::TextureFormat::R16Uint
                    | api::TextureFormat::Rg16Uint
                    | api::TextureFormat::Rgba16Uint
                    | api::TextureFormat::R32Uint
                    | api::TextureFormat::Rg32Uint
                    | api::TextureFormat::Rgba32Uint
                    | api::TextureFormat::Rgb10a2Uint
                    | api::TextureFormat::R64Uint
                    | api::TextureFormat::Rg64Uint
            ) =>
        {
            (
                false,
                vk::ClearValue {
                    color: vk::ClearColorValue { uint32: v },
                },
            )
        }
        (f, api::TextureClearValue::Color(api::ClearColor::Sint(v)))
            if matches!(
                f,
                api::TextureFormat::R8Sint
                    | api::TextureFormat::Rg8Sint
                    | api::TextureFormat::Rgba8Sint
                    | api::TextureFormat::R16Sint
                    | api::TextureFormat::Rg16Sint
                    | api::TextureFormat::Rgba16Sint
                    | api::TextureFormat::R32Sint
                    | api::TextureFormat::Rg32Sint
                    | api::TextureFormat::Rgba32Sint
            ) =>
        {
            (
                false,
                vk::ClearValue {
                    color: vk::ClearColorValue { int32: v },
                },
            )
        }
        _ => return Err(Error::Unsupported("Vulkan clear format/value")),
    };
    // Allocate all views before recording to leave the encoder unchanged on failure.
    let mut views = Vec::new();
    for mip in mips.clone() {
        let slices = if d.dimension == api::TextureDimension::D3 {
            0..(d.extent[2] >> mip).max(1)
        } else {
            layers.clone()
        };
        for layer in slices {
            let aspects = if depth {
                api::TextureAspects::DEPTH
            } else {
                api::TextureAspects::COLOR
            };
            let view = if d.dimension == api::TextureDimension::D3 {
                resource::create_slice_view(&e.owner, texture, mip, layer, aspects)?
            } else {
                resource::create_view(
                    &e.owner,
                    texture,
                    &api::TextureViewDesc {
                        format: d.format,
                        dimension: match d.dimension {
                            api::TextureDimension::D1 => api::TextureViewDimension::D1,
                            _ => api::TextureViewDimension::D2,
                        },
                        aspects,
                        mips: mip..mip + 1,
                        layers: layer..layer + 1,
                    },
                )?
            };
            views.push((mip, view));
        }
    }
    for (mip, view) in views {
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(view.native())
            .image_layout(if depth {
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            } else {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            })
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear);
        let rect = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: (d.extent[0] >> mip).max(1),
                height: (d.extent[1] >> mip).max(1),
            },
        };
        let mut render = vk::RenderingInfo::default()
            .render_area(rect)
            .layer_count(1);
        let colors = [attachment];
        if depth {
            render = render.depth_attachment(&colors[0]);
        } else {
            render = render.color_attachments(&colors);
        }
        // SAFETY: the views and rendering info stay live for both synchronous
        // recording calls; caller supplied matching attachment layouts.
        unsafe {
            e.device.device.cmd_begin_rendering(e.command, &render);
            e.device.device.cmd_end_rendering(e.command);
        }
        e.attachments.push(view);
    }
    Ok(())
}
