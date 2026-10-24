//! Encoder-local WebGPU upload pages.
//!
//! `queue.writeBuffer` fills only a private staging page.  The ordered native
//! copy remains in the owning `GPUCommandEncoder`, so this never changes the
//! ordering of Core commands across encoders.

use js_sys::Uint8Array;
use wasm_bindgen::JsValue;

use super::{WebGpuEncoder, error, field, number, object};
use fluxel_rhi_core::api::{Error, Result};

const PAGE_BYTES: u64 = 1024 * 1024;
const MAX_JS_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;

pub(crate) struct UploadPage {
    raw: JsValue,
    cursor: u64,
    size: u64,
    retained: bool,
}

pub(super) fn write(encoder: &mut WebGpuEncoder, bytes: &[u8]) -> Result<(JsValue, u64, u64)> {
    let bytes_len = u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?;
    let size = bytes_len.checked_add(3).ok_or(Error::InvalidRange)? & !3;
    if bytes_len > u64::from(u32::MAX) || size > MAX_JS_SAFE_INTEGER {
        return Err(Error::InvalidRange);
    }
    let index = encoder.upload_pages.iter().position(|page| {
        page.cursor
            .checked_add(size)
            .is_some_and(|end| end <= page.size)
    });
    let index = match index {
        Some(index) => index,
        None => {
            let page_size = PAGE_BYTES.max(size);
            if page_size
                > super::super::resource::device_limit(&encoder.device.0.device, "maxBufferSize")
                || page_size > MAX_JS_SAFE_INTEGER
            {
                return Err(Error::InvalidRange);
            }
            let desc = object();
            field(&desc, "size", number(page_size as f64))?;
            field(&desc, "usage", number(4_u32 | 8_u32))?;
            let raw = super::super::js::call1(&encoder.device.0.device, "createBuffer", &desc)
                .map_err(error)?;
            #[cfg(test)]
            super::UPLOAD_PAGE_CREATES.with(|count| count.set(count.get() + 1));
            encoder.upload_pages.push(UploadPage {
                raw,
                cursor: 0,
                size: page_size,
                retained: false,
            });
            encoder.upload_pages.len() - 1
        }
    };
    let page = &mut encoder.upload_pages[index];
    let offset = page.cursor;
    page.cursor += size;
    let data = Uint8Array::new_with_length(u32::try_from(size).map_err(|_| Error::InvalidRange)?);
    data.subarray(
        0,
        u32::try_from(bytes_len).map_err(|_| Error::InvalidRange)?,
    )
    .copy_from(bytes);
    super::super::js::call(
        &encoder.device.0.queue,
        "writeBuffer",
        &[page.raw.clone(), number(offset as f64), data.into()],
    )
    .map_err(error)?;
    if !page.retained {
        encoder.keep_alive.push(page.raw.clone());
        page.retained = true;
    }
    Ok((page.raw.clone(), offset, size))
}
