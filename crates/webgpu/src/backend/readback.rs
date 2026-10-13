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
        if range.end > src.size || range.start % 8 != 0 || size == 0 || size % 4 != 0 {
            return Err(Error::InvalidRange);
        }
        if src.usage.contains(api::BufferUsage::QUERY_RESOLVE) {
            // `MAP_READ | QUERY_RESOLVE` is prohibited by WebGPU.  Resource
            // creation gives the resolve buffer COPY_SRC, then this encoded
            // copy materializes the portable readback ticket in MAP_READ
            // staging memory.
            let descriptor: JsValue = Object::new().into();
            field(&descriptor, "size", JsValue::from_f64(size as f64))?;
            // GPUBufferUsage::MAP_READ | COPY_DST
            field(&descriptor, "usage", JsValue::from_f64(9.0))?;
            let staging =
                js::call1(&encoder.device.0.device, "createBuffer", &descriptor).map_err(error)?;
            js::call(
                &encoder.raw,
                "copyBufferToBuffer",
                &[
                    src.raw.clone(),
                    JsValue::from_f64(range.start as f64),
                    staging.clone(),
                    JsValue::from_f64(0.0),
                    JsValue::from_f64(size as f64),
                ],
            )
            .map_err(error)?;
            encoder
                .keep_alive
                .extend([src.raw.clone(), staging.clone()]);
            return Ok(WebGpuReadback {
                buffer: WebGpuBuffer {
                    raw: staging,
                    size,
                    usage: api::BufferUsage::MAP_READ.union(api::BufferUsage::COPY_DST),
                },
                range: 0..size,
            });
        }
        if !src.usage.contains(api::BufferUsage::MAP_READ) {
            return Err(Error::InvalidRange);
        }
        encoder.keep_alive.push(src.raw.clone());
        Ok(WebGpuReadback {
            buffer: src.clone(),
            range,
        })
    }

    fn map_readback<'a>(_: &'a WebGpuDevice, ticket: WebGpuReadback) -> Self::ReadbackFuture<'a> {
        Box::pin(async move {
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
            let bytes = Uint8Array::new(&mapped).to_vec();
            js::call0(&ticket.buffer.raw, "unmap").map_err(error)?;
            Ok(ReadbackData {
                row_pitch: size,
                slice_pitch: size,
                bytes,
            })
        })
    }
}
