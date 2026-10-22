//! Standalone attachment clear using dynamic rendering.
//!
//! The caller transitions each selected subresource to the matching attachment
//! layout before this command. Views survive until encoder completion.
use super::*;
use std::ops::Range;

pub(super) fn texture(
    e: &mut VulkanEncoder,
    texture: &VulkanTexture,
    mips: Range<u32>,
    layers: Range<u32>,
    value: api::TextureClearValue,
) -> Result<()> {
    let d = texture.desc();
    if e.raster || e.compute || !d.usage.contains(api::TextureUsage::RENDER_ATTACHMENT) {
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
        (api::TextureFormat::Depth32Float, api::TextureClearValue::Depth(v))
            if v.is_finite() && (0.0..=1.0).contains(&v) =>
        {
            (
                true,
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: v,
                        stencil: 0,
                    },
                },
            )
        }
        (
            api::TextureFormat::Rgba8Unorm
            | api::TextureFormat::Bgra8Unorm
            | api::TextureFormat::Rgba8UnormSrgb
            | api::TextureFormat::Bgra8UnormSrgb,
            api::TextureClearValue::Color(api::ClearColor::Float(v)),
        ) => (
            false,
            vk::ClearValue {
                color: vk::ClearColorValue { float32: v },
            },
        ),
        _ => return Err(Error::Unsupported("Vulkan clear format/value")),
    };
    // Allocate all views before recording to leave the encoder unchanged on failure.
    let mut views = Vec::new();
    for layer in layers {
        for mip in mips.clone() {
            let view = resource::create_view(
                &e.owner,
                texture,
                &api::TextureViewDesc {
                    format: d.format,
                    dimension: api::TextureViewDimension::D2,
                    aspects: if depth {
                        api::TextureAspects::DEPTH
                    } else {
                        api::TextureAspects::COLOR
                    },
                    mips: mip..mip + 1,
                    layers: layer..layer + 1,
                },
            )?;
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
