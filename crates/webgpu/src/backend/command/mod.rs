//! Direct `GPUCommandEncoder` and `GPURenderPassEncoder` lowering.
//!
//! `pass` is the encoder state machine: `None` permits copies and a new pass;
//! `Some` permits only that pass's commands. `keep_alive` owns cloned JS objects
//! referenced by the native command buffer until queue completion; it is not a
//! portable command cache or replay stream.

use fluxel_rhi_core::api::{self, Error, Result};
use js_sys::{Array, Object, Uint8Array};
use wasm_bindgen::JsValue;

#[cfg(test)]
thread_local! { static UPLOAD_PAGE_CREATES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
pub(super) fn reset_upload_page_creates() {
    UPLOAD_PAGE_CREATES.with(|count| count.set(0));
}
#[cfg(test)]
pub(super) fn upload_page_creates() -> u32 {
    UPLOAD_PAGE_CREATES.with(|count| count.get())
}

#[path = "../clear_fallback.rs"]
mod clear_fallback;
#[path = "../depth_upload.rs"]
mod depth_upload;
#[path = "../stencil_copy.rs"]
mod stencil_copy;
#[path = "../unaligned.rs"]
pub(super) mod unaligned;
#[path = "upload.rs"]
mod upload;
pub(super) use upload::{UploadPage, UploadPagePool};

use super::{
    instance::{WebGpu, WebGpuDevice},
    js,
    resource::{WebGpuBuffer, WebGpuTexture},
};

pub struct WebGpuEncoder {
    pub(super) device: WebGpuDevice,
    pub(super) raw: JsValue,
    pub(super) pass: Option<JsValue>,
    pub(super) render_area: Option<api::Rect>,
    pub(super) partial_resolves: Vec<super::partial::PartialResolve>,
    pub(super) partial_timestamp_end: Option<(super::query::WebGpuQuerySet, u32)>,
    /// WebGPU fixes this set in the render-pass descriptor.  Keep the native
    /// identity while the pass is open so `beginOcclusionQuery` cannot target
    /// a different Core query set.
    pub(super) occlusion_query_set: Option<JsValue>,
    pub(super) occlusion_logical_set: Option<JsValue>,
    pub(super) occlusion_next_index: u32,
    pub(super) occlusion_active_query: Option<super::query_segments::QuerySegments>,
    pub(super) occlusion_pending: Vec<super::query_segments::QuerySegments>,
    pub(super) occlusion_active: bool,
    pub(super) keep_alive: Vec<JsValue>,
    pub(super) upload_pages: Vec<upload::UploadPage>,
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

#[path = "../resolve.rs"]
mod resolve;

#[path = "../command_texture.rs"]
mod texture_ops;
pub(super) use texture_ops::texel_block;
use texture_ops::{
    copy_extent, copy_texture_desc, copy_unaligned_rows, validate_texel_buffer,
    validate_texture_region,
};

impl fluxel_rhi_core::backend::CommandBackend for WebGpu {
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
            render_area: None,
            partial_resolves: Vec::new(),
            partial_timestamp_end: None,
            occlusion_query_set: None,
            occlusion_logical_set: None,
            occlusion_next_index: 0,
            occlusion_active_query: None,
            occlusion_pending: Vec::new(),
            occlusion_active: false,
            keep_alive: Vec::new(),
            upload_pages: Vec::new(),
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
            || !src.usage.contains(api::BufferUsage::COPY_SRC)
            || !dst.usage.contains(api::BufferUsage::COPY_DST)
        {
            return Err(Error::InvalidRange);
        }
        if bytes == 0 {
            return Ok(());
        }
        if src_offset % 4 != 0
            || dst_offset % 4 != 0
            || bytes % 4 != 0
            || js_sys::Object::is(&src.raw, &dst.raw)
        {
            if dst.usage.contains(api::BufferUsage::MAP_READ) {
                return Err(Error::Unsupported(
                    "WebGPU mapped destination cannot preserve unaligned bytes",
                ));
            }
            if js_sys::Object::is(&src.raw, &dst.raw)
                && src_offset < dst_offset + bytes
                && dst_offset < src_offset + bytes
            {
                return Err(Error::InvalidRange);
            }
            let mut done = 0u64;
            while done < bytes {
                if (src_offset + done) % 4 == 0 && (dst_offset + done) % 4 == 0 {
                    let aligned = (bytes - done) & !3;
                    if aligned != 0 && !js_sys::Object::is(&src.raw, &dst.raw) {
                        js::call(
                            &encoder.raw,
                            "copyBufferToBuffer",
                            &[
                                src.raw.clone(),
                                number((src_offset + done) as f64),
                                dst.raw.clone(),
                                number((dst_offset + done) as f64),
                                number(aligned as f64),
                            ],
                        )
                        .map_err(error)?;
                        encoder
                            .keep_alive
                            .extend([src.raw.clone(), dst.raw.clone()]);
                        done += aligned;
                        continue;
                    }
                }
                let n = (1024 * 1024 - ((dst_offset + done) % 4)).min(bytes - done);
                unaligned::copy_word(
                    encoder,
                    src,
                    src_offset + done,
                    dst,
                    dst_offset + done,
                    n as usize,
                )?;
                done += n;
            }
            return Ok(());
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
        if encoder.pass.is_some() {
            return Err(Error::InvalidState);
        }
        if src.extent != dst.extent {
            return Err(Error::InvalidRange);
        }
        // SAFETY: `CommandEncoder::copy_texture` checked both DeviceIds against
        // this encoder; their cloned JS values are retained below.
        let source = unsafe { src.texture.native_ref() };
        let target = unsafe { dst.texture.native_ref() };
        validate_texture_region(&src)?;
        validate_texture_region(&dst)?;
        if source.desc.samples != target.desc.samples || src.aspect != dst.aspect {
            return Err(Error::InvalidRange);
        }
        if js_sys::Object::is(&source.raw, &target.raw)
            && src.mip == dst.mip
            && (source.desc.dimension == api::TextureDimension::D3
                || (src.origin[2] < dst.origin[2] + dst.extent[2]
                    && dst.origin[2] < src.origin[2] + src.extent[2]))
        {
            return Err(Error::InvalidRange);
        }
        if !source.desc.usage.contains(api::TextureUsage::COPY_SRC)
            || !target.desc.usage.contains(api::TextureUsage::COPY_DST)
            || !super::resource_validation::compatible_view_format(
                source.desc.format,
                target.desc.format,
            )
        {
            return Err(Error::Unsupported("WebGPU texture copy format or usage"));
        }
        let combined = matches!(
            source.desc.format,
            api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8
        );
        if combined && src.aspect == api::TextureAspects::DEPTH {
            return super::blit::blit(encoder, src, dst, api::FilterMode::Nearest);
        }
        if combined && src.aspect == api::TextureAspects::STENCIL {
            return stencil_copy::copy(encoder, src, dst);
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

    fn resolve(
        encoder: &mut WebGpuEncoder,
        src: api::TextureRegion<'_, Self>,
        dst: api::TextureRegion<'_, Self>,
    ) -> Result<()> {
        resolve::record(encoder, src, dst)
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
        if encoder.pass.is_some() {
            return Err(Error::InvalidState);
        }
        validate_texture_region(&dst)?;
        if destination.desc.samples != 1 {
            return Err(Error::InvalidRange);
        }
        if !src.usage.contains(api::BufferUsage::COPY_SRC)
            || !destination.desc.usage.contains(api::TextureUsage::COPY_DST)
        {
            return Err(Error::Unsupported("WebGPU buffer-to-texture copy usage"));
        }
        validate_texel_buffer(src, layout, &dst)?;
        if dst.aspect == api::TextureAspects::DEPTH
            && matches!(
                destination.desc.format,
                api::TextureFormat::Depth32Float | api::TextureFormat::Depth32FloatStencil8
            )
        {
            return depth_upload::upload(
                encoder,
                &src.raw,
                layout.offset,
                u64::from(layout.bytes_per_row),
                u64::from(layout.rows_per_image),
                &dst,
            );
        }
        let (_, _, block_bytes) = texel_block(destination.desc.format, dst.aspect)?;
        let native_alignment = if dst.aspect == api::TextureAspects::COLOR {
            block_bytes
        } else {
            4
        };
        if layout.bytes_per_row % 256 != 0 || layout.offset % native_alignment != 0 {
            return copy_unaligned_rows(encoder, src, layout, &dst, true);
        }
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
        if encoder.pass.is_some() {
            return Err(Error::InvalidState);
        }
        validate_texture_region(&src)?;
        if source.desc.samples != 1 {
            return Err(Error::InvalidRange);
        }
        if !source.desc.usage.contains(api::TextureUsage::COPY_SRC)
            || !dst.usage.contains(api::BufferUsage::COPY_DST)
        {
            return Err(Error::Unsupported("WebGPU texture-to-buffer copy usage"));
        }
        validate_texel_buffer(dst, layout, &src)?;
        let (_, _, block_bytes) = texel_block(source.desc.format, src.aspect)?;
        let native_alignment = if src.aspect == api::TextureAspects::COLOR {
            block_bytes
        } else {
            4
        };
        if layout.bytes_per_row % 256 != 0 || layout.offset % native_alignment != 0 {
            return copy_unaligned_rows(encoder, dst, layout, &src, false);
        }
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
        if encoder.pass.is_some() {
            return Err(Error::InvalidState);
        }
        if end > dst.size || !dst.usage.contains(api::BufferUsage::COPY_DST) {
            return Err(Error::InvalidRange);
        }
        if bytes.is_empty() {
            return Ok(());
        }
        if offset % 4 != 0 || (end != dst.size && bytes.len() % 4 != 0) {
            if dst.usage.contains(api::BufferUsage::MAP_READ)
                && !dst.usage.contains(api::BufferUsage::QUERY_RESOLVE)
            {
                return Err(Error::Unsupported(
                    "WebGPU mapped buffer cannot provide native COPY_SRC for unaligned write",
                ));
            }
            let mut position = 0usize;
            let mut current = offset;
            if current % 4 != 0 {
                let first = (current % 4) as usize;
                let count = (4 - first).min(bytes.len());
                unaligned::patch(encoder, dst, current & !3, first, &bytes[..count])?;
                current += count as u64;
                position += count;
            }
            let remaining = bytes.len() - position;
            let middle = remaining & !3;
            if middle != 0 {
                Self::write_buffer(encoder, dst, current, &bytes[position..position + middle])?;
                current += middle as u64;
                position += middle;
            }
            if position < bytes.len() {
                unaligned::patch(encoder, dst, current, 0, &bytes[position..])?;
            }
            return Ok(());
        }
        let (staging, source_offset, padded_size) = upload::write(encoder, bytes)?;
        js::call(
            &encoder.raw,
            "copyBufferToBuffer",
            &[
                staging.clone(),
                number(source_offset as f64),
                dst.raw.clone(),
                number(offset as f64),
                number(padded_size as f64),
            ],
        )
        .map_err(error)?;
        encoder.keep_alive.push(dst.raw.clone());
        Ok(())
    }

    fn clear_buffer(
        encoder: &mut WebGpuEncoder,
        dst: &WebGpuBuffer,
        range: std::ops::Range<u64>,
        value: u32,
    ) -> Result<()> {
        if encoder.pass.is_some() {
            return Err(Error::InvalidState);
        }
        if range.start >= range.end
            || range.end > dst.size
            || range.start % 4 != 0
            || (range.end - range.start) % 4 != 0
            || !dst.usage.contains(api::BufferUsage::COPY_DST)
        {
            return Err(Error::InvalidRange);
        }
        if value == 0 {
            js::call(
                &encoder.raw,
                "clearBuffer",
                &[
                    dst.raw.clone(),
                    number(range.start as f64),
                    number((range.end - range.start) as f64),
                ],
            )
            .map_err(error)?;
            encoder.keep_alive.push(dst.raw.clone());
            return Ok(());
        }
        // Upload one repeated pattern, then record native copies from that
        // encoder-local page. This avoids a staging allocation per chunk.
        let bytes = value.to_le_bytes();
        let chunk = bytes.repeat(16384);
        let (staging, source_offset, _) = upload::write(encoder, &chunk)?;
        let mut offset = range.start;
        while offset < range.end {
            let length = (range.end - offset).min(chunk.len() as u64) as usize;
            js::call(
                &encoder.raw,
                "copyBufferToBuffer",
                &[
                    staging.clone(),
                    number(source_offset as f64),
                    dst.raw.clone(),
                    number(offset as f64),
                    number(length as f64),
                ],
            )
            .map_err(error)?;
            offset += length as u64;
        }
        encoder.keep_alive.push(dst.raw.clone());
        Ok(())
    }

    fn clear_texture(
        encoder: &mut WebGpuEncoder,
        dst: &WebGpuTexture,
        mips: std::ops::Range<u32>,
        layers: std::ops::Range<u32>,
        value: api::TextureClearValue,
    ) -> Result<()> {
        if encoder.pass.is_some() {
            return Err(Error::InvalidState);
        }
        let desc = &dst.desc;
        if desc.dimension == api::TextureDimension::D1
            || !super::resource_validation::capabilities(&encoder.device, desc.format).3
        {
            return clear_fallback::clear(encoder, dst, mips, layers, value);
        }
        if desc.dimension == api::TextureDimension::D1
            || !(desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
                || desc.usage.contains(api::TextureUsage::COPY_DST))
            || !super::resource_validation::capabilities(&encoder.device, desc.format).3
        {
            return Err(Error::Unsupported(
                "WebGPU texture clear requires a 2D render attachment",
            ));
        }
        if mips.start >= mips.end
            || mips.end > desc.mip_levels
            || layers.start >= layers.end
            || layers.end > desc.array_layers
        {
            return Err(Error::InvalidRange);
        }
        let format = super::resource::texture_format(desc.format)?;
        let is_depth = format.starts_with("depth");
        let clear = match value {
            api::TextureClearValue::Depth(v) if is_depth && (0.0..=1.0).contains(&v) => None,
            api::TextureClearValue::Color(api::ClearColor::Float(v))
                if !is_depth
                    && format != "stencil8"
                    && !format.ends_with("uint")
                    && !format.ends_with("sint") =>
            {
                Some(v.map(f64::from))
            }
            api::TextureClearValue::Color(api::ClearColor::Uint(v)) if format.ends_with("uint") => {
                Some(v.map(f64::from))
            }
            api::TextureClearValue::Color(api::ClearColor::Sint(v)) if format.ends_with("sint") => {
                Some(v.map(f64::from))
            }
            _ => return Err(Error::InvalidRange),
        };
        for mip in mips {
            let slices = if desc.dimension == api::TextureDimension::D3 {
                0..(desc.extent[2].checked_shr(mip).unwrap_or(0).max(1))
            } else {
                layers.clone()
            };
            for layer in slices {
                let view_desc = object();
                field(
                    &view_desc,
                    "dimension",
                    JsValue::from_str(if desc.dimension == api::TextureDimension::D3 {
                        "3d"
                    } else {
                        "2d"
                    }),
                )?;
                field(&view_desc, "baseMipLevel", number(mip))?;
                field(&view_desc, "mipLevelCount", number(1u32))?;
                if desc.dimension == api::TextureDimension::D2 {
                    field(&view_desc, "baseArrayLayer", number(layer))?;
                    field(&view_desc, "arrayLayerCount", number(1u32))?;
                }
                let view = js::call1(&dst.raw, "createView", &view_desc).map_err(error)?;
                let attachment = object();
                field(&attachment, "view", view.clone())?;
                if desc.dimension == api::TextureDimension::D3 {
                    field(&attachment, "depthSlice", number(layer))?;
                }
                let pass_desc = object();
                if let Some(values) = clear {
                    field(&attachment, "loadOp", JsValue::from_str("clear"))?;
                    field(&attachment, "storeOp", JsValue::from_str("store"))?;
                    let color = object();
                    for (key, component) in ["r", "g", "b", "a"].into_iter().zip(values) {
                        field(&color, key, JsValue::from_f64(component))?;
                    }
                    field(&attachment, "clearValue", color)?;
                    let colors = Array::new();
                    colors.push(&attachment);
                    field(&pass_desc, "colorAttachments", colors.into())?;
                } else {
                    let api::TextureClearValue::Depth(depth) = value else {
                        return Err(Error::InvalidRange);
                    };
                    field(&attachment, "depthLoadOp", JsValue::from_str("clear"))?;
                    field(&attachment, "depthStoreOp", JsValue::from_str("store"))?;
                    field(
                        &attachment,
                        "depthClearValue",
                        JsValue::from_f64(f64::from(depth)),
                    )?;
                    if format.ends_with("stencil8") {
                        field(&attachment, "stencilReadOnly", JsValue::TRUE)?;
                    }
                    field(&pass_desc, "depthStencilAttachment", attachment)?;
                    field(&pass_desc, "colorAttachments", Array::new().into())?;
                }
                let pass = js::call1(&encoder.raw, "beginRenderPass", &pass_desc).map_err(error)?;
                js::call0(&pass, "end").map_err(error)?;
                encoder.keep_alive.push(view);
            }
        }
        encoder.keep_alive.push(dst.raw.clone());
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
        if encoder.pass.is_some() {
            return Err(Error::InvalidState);
        }
        validate_texture_region(&dst)?;
        if native.desc.samples != 1 {
            return Err(Error::InvalidRange);
        }
        if !native.desc.usage.contains(api::TextureUsage::COPY_DST) || dst.extent.contains(&0) {
            return Err(Error::InvalidRange);
        }
        let (block_w, block_h, block_bytes) = texel_block(native.desc.format, dst.aspect)?;
        let row_bytes = u64::from(dst.extent[0].div_ceil(block_w))
            .checked_mul(block_bytes)
            .ok_or(Error::InvalidRange)?;
        let src_stride = u64::from(layout.bytes_per_row);
        let src_rows = u64::from(layout.rows_per_image);
        if src_stride < row_bytes || src_rows < u64::from(dst.extent[1].div_ceil(block_h)) {
            return Err(Error::InvalidRange);
        }
        let height = u64::from(dst.extent[1].div_ceil(block_h));
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
        if dst.aspect == api::TextureAspects::DEPTH
            && matches!(
                native.desc.format,
                api::TextureFormat::Depth32Float | api::TextureFormat::Depth32FloatStencil8
            )
        {
            return depth_upload::upload(encoder, &staging, 0, native_stride, height, &dst);
        }
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
