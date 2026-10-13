//! Direct `GPUCommandEncoder` and `GPURenderPassEncoder` lowering.
//!
//! `pass` is the encoder state machine: `None` permits copies and a new pass;
//! `Some` permits only that pass's commands. `keep_alive` owns cloned JS objects
//! referenced by the native command buffer until queue completion; it is not a
//! portable command cache or replay stream.

use fluxel_rhi_core::api::{self, Error, Result};
use js_sys::{Object, Uint8Array};
use wasm_bindgen::JsValue;

use super::{
    instance::{WebGpu, WebGpuDevice},
    js,
    resource::WebGpuBuffer,
};

pub struct WebGpuEncoder {
    pub(super) device: WebGpuDevice,
    pub(super) raw: JsValue,
    pub(super) pass: Option<JsValue>,
    /// WebGPU fixes this set in the render-pass descriptor.  Keep the native
    /// identity while the pass is open so `beginOcclusionQuery` cannot target
    /// a different Core query set.
    pub(super) occlusion_query_set: Option<JsValue>,
    pub(super) occlusion_active: bool,
    pub(super) keep_alive: Vec<JsValue>,
}

pub(super) fn error(value: JsValue) -> Error {
    Error::Backend(js::message(&value))
}

pub(super) fn object() -> JsValue {
    Object::new().into()
}

pub(super) fn field(object: &JsValue, name: &str, value: JsValue) -> Result<()> {
    js::set(object, name, &value).map_err(error)
}

pub(super) fn number(value: impl Into<f64>) -> JsValue {
    JsValue::from_f64(value.into())
}

pub(super) fn pass(encoder: &WebGpuEncoder) -> Result<&JsValue> {
    encoder.pass.as_ref().ok_or(Error::InvalidState)
}

fn copy_origin(origin: [u32; 3]) -> Result<JsValue> {
    let value = object();
    field(&value, "x", number(origin[0]))?;
    field(&value, "y", number(origin[1]))?;
    field(&value, "z", number(origin[2]))?;
    Ok(value)
}

fn copy_extent(extent: [u32; 3]) -> Result<JsValue> {
    let value = object();
    field(&value, "width", number(extent[0]))?;
    field(&value, "height", number(extent[1]))?;
    field(&value, "depthOrArrayLayers", number(extent[2]))?;
    Ok(value)
}

fn copy_texture_desc(region: &api::TextureRegion<'_, WebGpu>) -> Result<JsValue> {
    let value = object();
    // SAFETY: every caller is a CommandEncoder entry point that checked the
    // region's DeviceId against the encoder; the JS clone is retained by it.
    let texture = unsafe { region.texture.native_ref() }.raw.clone();
    field(&value, "texture", texture)?;
    field(&value, "mipLevel", number(region.mip))?;
    field(&value, "origin", copy_origin(region.origin)?)?;
    field(
        &value,
        "aspect",
        JsValue::from_str(if region.aspect == api::TextureAspects::COLOR {
            "all"
        } else if region.aspect == api::TextureAspects::DEPTH {
            "depth-only"
        } else if region.aspect == api::TextureAspects::STENCIL {
            "stencil-only"
        } else {
            return Err(Error::Unsupported("WebGPU texture copy aspect"));
        }),
    )?;
    Ok(value)
}

fn validate_texel_buffer(
    buffer: &WebGpuBuffer,
    layout: api::TexelCopyBufferLayout,
    region: &api::TextureRegion<'_, WebGpu>,
) -> Result<()> {
    // SAFETY: callers first check this region's DeviceId against the encoder;
    // this only reads immutable descriptor metadata during that call.
    let format = unsafe { region.texture.native_ref() }.desc.format;
    let bytes_per_texel = match format {
        api::TextureFormat::R8Unorm => 1_u64,
        api::TextureFormat::Rg8Unorm => 2,
        api::TextureFormat::Rgba8Unorm
        | api::TextureFormat::Rgba8UnormSrgb
        | api::TextureFormat::Bgra8Unorm
        | api::TextureFormat::Bgra8UnormSrgb
        | api::TextureFormat::R32Float
        | api::TextureFormat::Depth32Float => 4,
        api::TextureFormat::Rgba16Float => 8,
        _ => return Err(Error::Unsupported("WebGPU texel copy format")),
    };
    let [width, height, layers] = region.extent;
    let row_bytes = u64::from(width)
        .checked_mul(bytes_per_texel)
        .ok_or(Error::InvalidRange)?;
    let stride = u64::from(layout.bytes_per_row);
    let rows = u64::from(layout.rows_per_image);
    if width == 0
        || height == 0
        || layers == 0
        || layout.offset % 4 != 0
        || stride < row_bytes
        || stride % 256 != 0
        || rows < u64::from(height)
    {
        return Err(Error::InvalidRange);
    }
    let layer_offset = u64::from(layers - 1)
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(stride))
        .ok_or(Error::InvalidRange)?;
    let last_row = u64::from(height - 1)
        .checked_mul(stride)
        .ok_or(Error::InvalidRange)?;
    let end = layout
        .offset
        .checked_add(layer_offset)
        .and_then(|n| n.checked_add(last_row))
        .and_then(|n| n.checked_add(row_bytes))
        .ok_or(Error::InvalidRange)?;
    if end > buffer.size {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

impl api::CommandBackend for WebGpu {
    fn begin_encoder(device: &WebGpuDevice, lane: api::LaneId) -> Result<WebGpuEncoder> {
        if device.0.lost.get() {
            return Err(Error::DeviceLost);
        }
        if let Some(message) = device.0.validation_error.borrow().clone() {
            return Err(Error::Backend(message));
        }
        if lane.device != device.0.id || lane.ordinal != 0 {
            return Err(Error::WrongLane);
        }
        let raw = js::call1(&device.0.device, "createCommandEncoder", &object()).map_err(error)?;
        Ok(WebGpuEncoder {
            device: device.clone(),
            raw,
            pass: None,
            occlusion_query_set: None,
            occlusion_active: false,
            keep_alive: Vec::new(),
        })
    }

    fn transition(_: &mut WebGpuEncoder, _: &[api::ResourceTransition<'_, Self>]) -> Result<()> {
        // WebGPU resolves native usage transitions at command execution. The
        // renderer still owns the ordering and hazards expressed by its plan.
        Ok(())
    }

    fn copy_buffer(
        encoder: &mut WebGpuEncoder,
        src: &WebGpuBuffer,
        src_offset: u64,
        dst: &WebGpuBuffer,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<()> {
        if src_offset
            .checked_add(bytes)
            .is_none_or(|end| end > src.size)
            || dst_offset
                .checked_add(bytes)
                .is_none_or(|end| end > dst.size)
            || src_offset % 4 != 0
            || dst_offset % 4 != 0
            || bytes % 4 != 0
            || !src.usage.contains(api::BufferUsage::COPY_SRC)
            || !dst.usage.contains(api::BufferUsage::COPY_DST)
        {
            return Err(Error::InvalidRange);
        }
        js::call(
            &encoder.raw,
            "copyBufferToBuffer",
            &[
                src.raw.clone(),
                number(src_offset as f64),
                dst.raw.clone(),
                number(dst_offset as f64),
                number(bytes as f64),
            ],
        )
        .map_err(error)?;
        encoder
            .keep_alive
            .extend([src.raw.clone(), dst.raw.clone()]);
        Ok(())
    }

    fn copy_texture(
        encoder: &mut WebGpuEncoder,
        src: api::TextureRegion<'_, Self>,
        dst: api::TextureRegion<'_, Self>,
    ) -> Result<()> {
        if src.extent != dst.extent {
            return Err(Error::InvalidRange);
        }
        // SAFETY: `CommandEncoder::copy_texture` checked both DeviceIds against
        // this encoder; their cloned JS values are retained below.
        let source = unsafe { src.texture.native_ref() };
        let target = unsafe { dst.texture.native_ref() };
        if !source.desc.usage.contains(api::TextureUsage::COPY_SRC)
            || !target.desc.usage.contains(api::TextureUsage::COPY_DST)
            || source.desc.format != target.desc.format
        {
            return Err(Error::Unsupported("WebGPU texture copy format or usage"));
        }
        js::call(
            &encoder.raw,
            "copyTextureToTexture",
            &[
                copy_texture_desc(&src)?,
                copy_texture_desc(&dst)?,
                copy_extent(src.extent)?,
            ],
        )
        .map_err(error)?;
        encoder
            .keep_alive
            .extend([source.raw.clone(), target.raw.clone()]);
        Ok(())
    }

    fn buffer_to_texture(
        encoder: &mut WebGpuEncoder,
        src: &WebGpuBuffer,
        layout: api::TexelCopyBufferLayout,
        dst: api::TextureRegion<'_, Self>,
    ) -> Result<()> {
        // SAFETY: `CommandEncoder::buffer_to_texture` checked the source and
        // destination DeviceIds against this encoder before this SPI call.
        let destination = unsafe { dst.texture.native_ref() };
        if !src.usage.contains(api::BufferUsage::COPY_SRC)
            || !destination.desc.usage.contains(api::TextureUsage::COPY_DST)
        {
            return Err(Error::Unsupported("WebGPU buffer-to-texture copy usage"));
        }
        validate_texel_buffer(src, layout, &dst)?;
        let source = object();
        field(&source, "buffer", src.raw.clone())?;
        field(&source, "offset", number(layout.offset as f64))?;
        field(&source, "bytesPerRow", number(layout.bytes_per_row))?;
        field(&source, "rowsPerImage", number(layout.rows_per_image))?;
        let texture = destination.raw.clone();
        js::call(
            &encoder.raw,
            "copyBufferToTexture",
            &[source, copy_texture_desc(&dst)?, copy_extent(dst.extent)?],
        )
        .map_err(error)?;
        encoder.keep_alive.extend([src.raw.clone(), texture]);
        Ok(())
    }

    fn texture_to_buffer(
        encoder: &mut WebGpuEncoder,
        src: api::TextureRegion<'_, Self>,
        dst: &WebGpuBuffer,
        layout: api::TexelCopyBufferLayout,
    ) -> Result<()> {
        // SAFETY: `CommandEncoder::texture_to_buffer` checked the source and
        // destination DeviceIds against this encoder before this SPI call.
        let source = unsafe { src.texture.native_ref() };
        if !source.desc.usage.contains(api::TextureUsage::COPY_SRC)
            || !dst.usage.contains(api::BufferUsage::COPY_DST)
        {
            return Err(Error::Unsupported("WebGPU texture-to-buffer copy usage"));
        }
        validate_texel_buffer(dst, layout, &src)?;
        let target = object();
        field(&target, "buffer", dst.raw.clone())?;
        field(&target, "offset", number(layout.offset as f64))?;
        field(&target, "bytesPerRow", number(layout.bytes_per_row))?;
        field(&target, "rowsPerImage", number(layout.rows_per_image))?;
        let texture = source.raw.clone();
        js::call(
            &encoder.raw,
            "copyTextureToBuffer",
            &[copy_texture_desc(&src)?, target, copy_extent(src.extent)?],
        )
        .map_err(error)?;
        encoder.keep_alive.extend([texture, dst.raw.clone()]);
        Ok(())
    }

    fn write_buffer(
        encoder: &mut WebGpuEncoder,
        dst: &WebGpuBuffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()> {
        let end = offset
            .checked_add(bytes.len() as u64)
            .ok_or(Error::InvalidRange)?;
        if end > dst.size || offset % 4 != 0 || !dst.usage.contains(api::BufferUsage::COPY_DST) {
            return Err(Error::InvalidRange);
        }
        if bytes.is_empty() {
            return Ok(());
        }
        if end != dst.size && bytes.len() % 4 != 0 {
            return Err(Error::Unsupported(
                "unaligned interior WebGPU buffer upload",
            ));
        }
        let padded_size = (bytes.len() as u64)
            .checked_add(3)
            .ok_or(Error::InvalidRange)?
            & !3;
        let desc = object();
        field(&desc, "size", number(padded_size as f64))?;
        field(&desc, "usage", number(4_u32))?;
        field(&desc, "mappedAtCreation", JsValue::TRUE)?;
        let staging = js::call1(&encoder.device.0.device, "createBuffer", &desc).map_err(error)?;
        let mapped = js::call0(&staging, "getMappedRange").map_err(error)?;
        let mut padded = vec![0_u8; padded_size as usize];
        padded[..bytes.len()].copy_from_slice(bytes);
        Uint8Array::new(&mapped).copy_from(&padded);
        js::call0(&staging, "unmap").map_err(error)?;
        js::call(
            &encoder.raw,
            "copyBufferToBuffer",
            &[
                staging.clone(),
                number(0_u32),
                dst.raw.clone(),
                number(offset as f64),
                number(padded_size as f64),
            ],
        )
        .map_err(error)?;
        encoder.keep_alive.extend([staging, dst.raw.clone()]);
        Ok(())
    }

    fn write_texture(
        encoder: &mut WebGpuEncoder,
        dst: api::TextureRegion<'_, Self>,
        layout: api::TexelCopyBufferLayout,
        bytes: &[u8],
    ) -> Result<()> {
        // SAFETY: `CommandEncoder::write_texture` checked the destination
        // DeviceId; the raw texture is retained with the finished command buffer.
        let native = unsafe { dst.texture.native_ref() };
        if native.desc.dimension != api::TextureDimension::D2
            || native.desc.format != api::TextureFormat::Rgba8Unorm
            || dst.aspect != api::TextureAspects::COLOR
            || !native.desc.usage.contains(api::TextureUsage::COPY_DST)
            || dst.extent.contains(&0)
        {
            return Err(Error::Unsupported("WebGPU RGBA8 texture upload"));
        }
        let row_bytes = u64::from(dst.extent[0])
            .checked_mul(4)
            .ok_or(Error::InvalidRange)?;
        let src_stride = u64::from(layout.bytes_per_row);
        let src_rows = u64::from(layout.rows_per_image);
        if src_stride < row_bytes || src_rows < u64::from(dst.extent[1]) {
            return Err(Error::InvalidRange);
        }
        let height = u64::from(dst.extent[1]);
        let layers = u64::from(dst.extent[2]);
        let last_byte = layout
            .offset
            .checked_add(
                (layers - 1)
                    .checked_mul(src_rows)
                    .ok_or(Error::InvalidRange)?
                    .checked_mul(src_stride)
                    .ok_or(Error::InvalidRange)?,
            )
            .and_then(|n| n.checked_add((height - 1) * src_stride))
            .and_then(|n| n.checked_add(row_bytes))
            .ok_or(Error::InvalidRange)?;
        if last_byte > bytes.len() as u64 {
            return Err(Error::InvalidRange);
        }
        let native_stride = row_bytes.checked_add(255).ok_or(Error::InvalidRange)? & !255;
        let native_size = native_stride
            .checked_mul(height)
            .and_then(|n| n.checked_mul(layers))
            .ok_or(Error::InvalidRange)?;
        let size = usize::try_from(native_size).map_err(|_| Error::OutOfMemory)?;
        let mut packed = Vec::new();
        packed
            .try_reserve_exact(size)
            .map_err(|_| Error::OutOfMemory)?;
        packed.resize(size, 0);
        for layer in 0..layers {
            for row in 0..height {
                let src = layout.offset + (layer * src_rows + row) * src_stride;
                let dst_offset = (layer * height + row) * native_stride;
                let count = row_bytes as usize;
                packed[dst_offset as usize..dst_offset as usize + count]
                    .copy_from_slice(&bytes[src as usize..src as usize + count]);
            }
        }
        let desc = object();
        field(&desc, "size", number(native_size as f64))?;
        field(&desc, "usage", number(4_u32))?;
        field(&desc, "mappedAtCreation", JsValue::TRUE)?;
        let staging = js::call1(&encoder.device.0.device, "createBuffer", &desc).map_err(error)?;
        let mapped = js::call0(&staging, "getMappedRange").map_err(error)?;
        Uint8Array::new(&mapped).copy_from(&packed);
        js::call0(&staging, "unmap").map_err(error)?;
        let source = object();
        field(&source, "buffer", staging.clone())?;
        field(&source, "offset", number(0_u32))?;
        field(&source, "bytesPerRow", number(native_stride as f64))?;
        field(&source, "rowsPerImage", number(height as f64))?;
        js::call(
            &encoder.raw,
            "copyBufferToTexture",
            &[source, copy_texture_desc(&dst)?, copy_extent(dst.extent)?],
        )
        .map_err(error)?;
        encoder.keep_alive.extend([staging, native.raw.clone()]);
        Ok(())
    }

    fn blit(
        encoder: &mut WebGpuEncoder,
        src: api::TextureRegion<'_, Self>,
        dst: api::TextureRegion<'_, Self>,
        filter: api::FilterMode,
    ) -> Result<()> {
        super::blit::blit(encoder, src, dst, filter)
    }
}
