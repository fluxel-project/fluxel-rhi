//! Completion-gated WebGPU readback buffer mapping.
//!
//! Mapping begins only after the caller supplies a completion point covering the
//! encoded copy. Query resolve sources use private MAP_READ staging because the
//! WebGPU usage rules forbid MAP_READ and QUERY_RESOLVE on one buffer.

use std::{future::Future, ops::Range, pin::Pin};

use fluxel_rhi_core::api::{self, Error, ReadbackData, Result};
use js_sys::{Object, Promise, Uint8Array};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

use super::{
    command::WebGpuEncoder,
    instance::{WebGpu, WebGpuDevice},
    js,
    resource::WebGpuBuffer,
};

/// Ownership of a buffer range retained until the caller supplies completion.
pub struct WebGpuReadback {
    buffer: WebGpuBuffer,
    range: Range<u64>,
    crop: Range<usize>,
}

struct ReadbackMapping(WebGpuBuffer);
impl Drop for ReadbackMapping {
    fn drop(&mut self) {
        let _ = js::call0(&self.0.raw, "unmap");
        self.0.mapped.set(false);
    }
}

fn error(value: JsValue) -> Error {
    Error::Backend(js::message(&value))
}

fn field(object: &JsValue, name: &str, value: JsValue) -> Result<()> {
    js::set(object, name, &value).map_err(error)
}

impl api::ReadbackBackend for WebGpu {
    type NativeReadback = WebGpuReadback;
    type ReadbackFuture<'a>
        = Pin<Box<dyn Future<Output = Result<ReadbackData>> + 'a>>
    where
        Self: 'a;

    fn encode_readback(
        encoder: &mut WebGpuEncoder,
        src: &WebGpuBuffer,
        range: Range<u64>,
    ) -> Result<WebGpuReadback> {
        let size = range
            .end
            .checked_sub(range.start)
            .ok_or(Error::InvalidRange)?;
        if range.end > src.size || size == 0 {
            return Err(Error::InvalidRange);
        }
        if !src.usage.contains(api::BufferUsage::MAP_READ) {
            if !src.usage.contains(api::BufferUsage::COPY_SRC)
                && !src.usage.contains(api::BufferUsage::QUERY_RESOLVE)
            {
                return Err(Error::InvalidRange);
            }
            // `MAP_READ | QUERY_RESOLVE` is prohibited by WebGPU.  Resource
            // creation gives the resolve buffer COPY_SRC, then this encoded
            // copy materializes the portable readback ticket in MAP_READ
            // staging memory.
            let aligned_start = range.start & !3;
            let aligned_end = range.end.checked_add(3).ok_or(Error::InvalidRange)? & !3;
            let native_size = aligned_end - aligned_start;
            let crop_start =
                usize::try_from(range.start - aligned_start).map_err(|_| Error::InvalidRange)?;
            let crop_end = crop_start
                .checked_add(usize::try_from(size).map_err(|_| Error::InvalidRange)?)
                .ok_or(Error::InvalidRange)?;
            let descriptor: JsValue = Object::new().into();
            field(&descriptor, "size", JsValue::from_f64(native_size as f64))?;
            // GPUBufferUsage::MAP_READ | COPY_DST
            field(&descriptor, "usage", JsValue::from_f64(9.0))?;
            let staging =
                js::call1(&encoder.device.0.device, "createBuffer", &descriptor).map_err(error)?;
            js::call(
                &encoder.raw,
                "copyBufferToBuffer",
                &[
                    src.raw.clone(),
                    JsValue::from_f64(aligned_start as f64),
                    staging.clone(),
                    JsValue::from_f64(0.0),
                    JsValue::from_f64(native_size as f64),
                ],
            )
            .map_err(error)?;
            encoder
                .keep_alive
                .extend([src.raw.clone(), staging.clone()]);
            return Ok(WebGpuReadback {
                buffer: WebGpuBuffer {
                    raw: staging,
                    size: native_size,
                    usage: api::BufferUsage::MAP_READ.union(api::BufferUsage::COPY_DST),
                    mapped: std::rc::Rc::new(std::cell::Cell::new(false)),
                },
                range: 0..native_size,
                crop: crop_start..crop_end,
            });
        }
        encoder.keep_alive.push(src.raw.clone());
        let aligned_start = range.start & !7;
        let aligned_end = range.end.checked_add(3).ok_or(Error::InvalidRange)? & !3;
        let crop_start =
            usize::try_from(range.start - aligned_start).map_err(|_| Error::InvalidRange)?;
        let crop_end = crop_start
            .checked_add(usize::try_from(size).map_err(|_| Error::InvalidRange)?)
            .ok_or(Error::InvalidRange)?;
        Ok(WebGpuReadback {
            buffer: src.clone(),
            range: aligned_start..aligned_end,
            crop: crop_start..crop_end,
        })
    }

    fn map_readback<'a>(_: &'a WebGpuDevice, ticket: WebGpuReadback) -> Self::ReadbackFuture<'a> {
        Box::pin(async move {
            if ticket.buffer.mapped.replace(true) {
                return Err(Error::InvalidState);
            }
            let _mapping = ReadbackMapping(ticket.buffer.clone());
            let start = ticket.range.start;
            let size = ticket.range.end - start;
            let promise = js::call(
                &ticket.buffer.raw,
                "mapAsync",
                &[
                    // GPUMapMode.READ
                    JsValue::from_f64(1.0),
                    JsValue::from_f64(start as f64),
                    JsValue::from_f64(size as f64),
                ],
            )
            .map_err(error)?;
            JsFuture::from(Promise::from(promise))
                .await
                .map_err(error)?;
            let mapped = js::call(
                &ticket.buffer.raw,
                "getMappedRange",
                &[
                    JsValue::from_f64(start as f64),
                    JsValue::from_f64(size as f64),
                ],
            )
            .map_err(error)?;
            let mapped_bytes = Uint8Array::new(&mapped).to_vec();
            let bytes = mapped_bytes
                .get(ticket.crop)
                .ok_or(Error::InvalidRange)?
                .to_vec();
            Ok(ReadbackData {
                row_pitch: bytes.len() as u64,
                slice_pitch: bytes.len() as u64,
                bytes,
            })
        })
    }
}
