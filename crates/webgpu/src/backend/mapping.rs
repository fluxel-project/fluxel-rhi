//! WebGPU asynchronous buffer mapping with cancellation-safe ownership.
//!
//! A map holds exclusive ownership from mapAsync until unmap. The native
//! aligned range may include padding around the Core-visible byte range.

use super::{
    instance::{WebGpu, WebGpuDevice},
    js,
    resource::WebGpuBuffer,
};
use fluxel_rhi_core::api::{self, Error, Result};
use js_sys::{Promise, Uint8Array};
use std::{future::Future, ops::Range, pin::Pin};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

pub struct WebGpuMap {
    buffer: WebGpuBuffer,
    mapped: Uint8Array,
    bytes: Vec<u8>,
    offset: usize,
    mode: api::MapMode,
}

struct Pending {
    buffer: WebGpuBuffer,
    armed: bool,
}
impl Drop for Pending {
    fn drop(&mut self) {
        if self.armed {
            let _ = js::call0(&self.buffer.raw, "unmap");
            self.buffer.mapped.set(false);
        }
    }
}

fn failure(value: JsValue) -> Error {
    Error::Backend(js::message(&value))
}
fn checked_range(range: Range<u64>, len: usize) -> Result<Range<usize>> {
    let start = usize::try_from(range.start).map_err(|_| Error::InvalidRange)?;
    let end = usize::try_from(range.end).map_err(|_| Error::InvalidRange)?;
    if start > end || end > len {
        return Err(Error::InvalidRange);
    }
    Ok(start..end)
}
fn copy_to_native(map: &WebGpuMap, range: Range<usize>) {
    let start = map.offset + range.start;
    map.mapped
        .subarray(start as u32, (start + range.len()) as u32)
        .copy_from(&map.bytes[range]);
}

impl api::MappingBackend for WebGpu {
    type NativeMap = WebGpuMap;
    type MapFuture<'a> = Pin<Box<dyn Future<Output = Result<WebGpuMap>> + 'a>>;

    fn map<'a>(
        device: &'a WebGpuDevice,
        buffer: &'a WebGpuBuffer,
        range: Range<u64>,
        mode: api::MapMode,
    ) -> Self::MapFuture<'a> {
        Box::pin(async move {
            if device.0.lost.get() {
                return Err(Error::DeviceLost);
            }
            let usage = match mode {
                api::MapMode::Read => api::BufferUsage::MAP_READ,
                api::MapMode::Write => api::BufferUsage::MAP_WRITE,
            };
            let native_bit = match mode {
                api::MapMode::Read => 1,
                api::MapMode::Write => 2,
            };
            if !buffer.usage.contains(usage)
                || buffer.native_usage & native_bit == 0
                || range.start >= range.end
                || range.end > buffer.size
            {
                return Err(Error::InvalidRange);
            }
            if buffer.mapped.replace(true) {
                return Err(Error::InvalidState);
            }
            let mut pending = Pending {
                buffer: buffer.clone(),
                armed: true,
            };
            let start = range.start & !7;
            let end = range.end.checked_add(3).ok_or(Error::InvalidRange)? & !3;
            let mapped_size = end - start;
            let crop_start =
                usize::try_from(range.start - start).map_err(|_| Error::InvalidRange)?;
            let length =
                usize::try_from(range.end - range.start).map_err(|_| Error::InvalidRange)?;
            let promise = js::call(
                &buffer.raw,
                "mapAsync",
                &[
                    JsValue::from_f64(if mode == api::MapMode::Read { 1.0 } else { 2.0 }),
                    JsValue::from_f64(start as f64),
                    JsValue::from_f64(mapped_size as f64),
                ],
            )
            .map_err(failure)?;
            JsFuture::from(Promise::from(promise))
                .await
                .map_err(failure)?;
            if device.0.lost.get() {
                return Err(Error::DeviceLost);
            }
            let mapped = js::call(
                &buffer.raw,
                "getMappedRange",
                &[
                    JsValue::from_f64(start as f64),
                    JsValue::from_f64(mapped_size as f64),
                ],
            )
            .map_err(failure)?;
            let mapped = Uint8Array::new(&mapped);
            let end = crop_start.checked_add(length).ok_or(Error::InvalidRange)?;
            if end > mapped.length() as usize {
                return Err(Error::InvalidRange);
            }
            let bytes = mapped.subarray(crop_start as u32, end as u32).to_vec();
            pending.armed = false;
            Ok(WebGpuMap {
                buffer: buffer.clone(),
                mapped,
                bytes,
                offset: crop_start,
                mode,
            })
        })
    }
    fn bytes(map: &WebGpuMap) -> &[u8] {
        &map.bytes
    }
    fn bytes_mut(map: &mut WebGpuMap) -> &mut [u8] {
        &mut map.bytes
    }
    fn flush(map: &mut WebGpuMap, bytes: Range<u64>) -> Result<()> {
        if map.mode != api::MapMode::Write {
            return Err(Error::InvalidState);
        }
        let range = checked_range(bytes, map.bytes.len())?;
        copy_to_native(map, range);
        Ok(())
    }
    fn invalidate(map: &mut WebGpuMap, bytes: Range<u64>) -> Result<()> {
        if map.mode != api::MapMode::Read {
            return Err(Error::InvalidState);
        }
        let range = checked_range(bytes, map.bytes.len())?;
        let start = map.offset + range.start;
        map.mapped
            .subarray(start as u32, (start + range.len()) as u32)
            .copy_to(&mut map.bytes[range]);
        Ok(())
    }
    fn unmap(map: WebGpuMap) {
        if map.mode == api::MapMode::Write {
            copy_to_native(&map, 0..map.bytes.len());
        }
        let _ = js::call0(&map.buffer.raw, "unmap");
        map.buffer.mapped.set(false);
    }
}
