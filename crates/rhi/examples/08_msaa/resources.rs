//! Image decoding and attachment/resource construction for the Voyager scene.
//!
//! The module creates durable scene resources and transient resize targets but
//! does not record or submit frame commands.

use super::MsaaBackend;
use fluxel_rhi::api::*;
use std::io::Cursor;

pub(super) fn decode_material_image(image: &super::super::gltf::EmbeddedImage) -> Result<Vec<u8>> {
    if image.mime_type != "image/png" {
        return Err(Error::Unsupported("08_msaa requires embedded PNG images"));
    }
    let mut decoder = png::Decoder::new(Cursor::new(&image.bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|error| Error::Backend(error.to_string()))?;
    let mut pixels = vec![0; reader.output_buffer_size()];
    let frame = reader
        .next_frame(&mut pixels)
        .map_err(|error| Error::Backend(error.to_string()))?;
    if frame.width != 1024 || frame.height != 1024 || frame.bit_depth != png::BitDepth::Eight {
        return Err(Error::Unsupported(
            "08_msaa PNG image dimensions or bit depth",
        ));
    }
    pixels.truncate(frame.buffer_size());
    match frame.color_type {
        png::ColorType::Rgba => Ok(pixels),
        png::ColorType::Rgb => {
            let mut rgba = Vec::with_capacity(1024 * 1024 * 4);
            for rgb in pixels.chunks_exact(3) {
                rgba.extend_from_slice(rgb);
                rgba.push(255);
            }
            Ok(rgba)
        }
        _ => Err(Error::Unsupported("08_msaa PNG color type")),
    }
}

pub(super) fn supported_samples<B: MsaaBackend>(
    device: &Device<B>,
    format: TextureFormat,
) -> Result<u32> {
    for samples in [8, 4, 2] {
        let color = TextureDesc {
            label: Some("08 MSAA color support".into()),
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        };
        let depth = TextureDesc {
            label: Some("08 MSAA depth support".into()),
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 1,
            format: TextureFormat::Depth32Float,
            mip_levels: 1,
            samples,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        };
        if matches!(device.texture_support(&color), Support::Supported(info) if info.sample_counts.contains(&samples))
            && matches!(device.texture_support(&depth), Support::Supported(info) if info.sample_counts.contains(&samples))
        {
            return Ok(samples);
        }
    }
    Err(Error::Unsupported(
        "08_msaa requires a common color/depth sample count greater than one",
    ))
}

pub(super) fn buffer<B: MsaaBackend>(
    device: &Device<B>,
    label: &str,
    size: u64,
    usage: BufferUsage,
) -> Result<Buffer<B>> {
    device.create_buffer(&BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    })
}
pub(super) fn layout_entry(
    binding: u32,
    visibility: Vec<ShaderStage>,
    ty: BindingType,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}
pub(super) fn material_texture<B: MsaaBackend>(
    device: &Device<B>,
    label: &str,
) -> Result<Texture<B>> {
    device.create_texture(&TextureDesc {
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [1024, 1024, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_DST.union(TextureUsage::TEXTURE_BINDING),
        view_formats: vec![],
    })
}
pub(super) fn texture_view<B: MsaaBackend>(
    device: &Device<B>,
    texture: &Texture<B>,
) -> Result<TextureView<B>> {
    device.create_view(
        texture,
        &TextureViewDesc {
            format: TextureFormat::Rgba8Unorm,
            dimension: TextureViewDimension::D2,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        },
    )
}
pub(super) fn color_target<B: MsaaBackend>(
    device: &Device<B>,
    extent: [u32; 2],
    format: TextureFormat,
    samples: u32,
) -> Result<(Texture<B>, TextureView<B>)> {
    let texture = device.create_texture(&TextureDesc {
        label: Some("08 MSAA color".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format,
        mip_levels: 1,
        samples,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    })?;
    let view = device.create_view(
        &texture,
        &TextureViewDesc {
            format,
            dimension: TextureViewDimension::D2,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        },
    )?;
    Ok((texture, view))
}
pub(super) fn depth_target<B: MsaaBackend>(
    device: &Device<B>,
    extent: [u32; 2],
    samples: u32,
) -> Result<(Texture<B>, TextureView<B>)> {
    let texture = device.create_texture(&TextureDesc {
        label: Some("08 MSAA depth".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    })?;
    let view = device.create_view(
        &texture,
        &TextureViewDesc {
            format: TextureFormat::Depth32Float,
            dimension: TextureViewDimension::D2,
            aspects: TextureAspects::DEPTH,
            mips: 0..1,
            layers: 0..1,
        },
    )?;
    Ok((texture, view))
}
