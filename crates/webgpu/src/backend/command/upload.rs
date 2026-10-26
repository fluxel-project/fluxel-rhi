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
/// Keep staging allocations bounded even when an application has many short
/// encoders in flight. Oversized pages are deliberately one-shot.
const MAX_CACHED_PAGE_BYTES: u64 = 32 * PAGE_BYTES;
const MAX_JS_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;

pub(crate) struct UploadPage {
    raw: JsValue,
    cursor: u64,
    size: u64,
    retained: bool,
}

/// Device-local pages which are known to be past their queue completion
/// boundary.  Pages enter this pool only from `submit::retire`; an encoder
/// never borrows an in-flight page.
pub(crate) struct UploadPagePool {
    pages: Vec<UploadPage>,
    bytes: u64,
}

impl UploadPagePool {
    pub(crate) const fn new() -> Self {
        Self {
            pages: Vec::new(),
            bytes: 0,
        }
    }

    fn take(&mut self, needed: u64) -> Option<UploadPage> {
        let index = self.pages.iter().position(|page| page.size >= needed)?;
        let mut page = self.pages.swap_remove(index);
        self.bytes -= page.size;
        page.cursor = 0;
        page.retained = false;
        Some(page)
    }

    pub(crate) fn recycle(&mut self, mut page: UploadPage) {
        page.cursor = 0;
        page.retained = false;
        if page.size > PAGE_BYTES || self.bytes.saturating_add(page.size) > MAX_CACHED_PAGE_BYTES {
            // Releasing the JS reference is sufficient for eventual browser
            // collection. `destroy` releases the native allocation promptly
            // when the page falls outside the bounded reusable arena.
            let _ = super::super::js::call0(&page.raw, "destroy");
            return;
        }
        self.bytes += page.size;
        self.pages.push(page);
    }
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
            let page = encoder
                .device
                .0
                .upload_page_pool
                .borrow_mut()
                .take(page_size);
            let page = match page {
                Some(page) => page,
                None => {
                    let desc = object();
                    field(&desc, "size", number(page_size as f64))?;
                    field(&desc, "usage", number(4_u32 | 8_u32))?;
                    let raw =
                        super::super::js::call1(&encoder.device.0.device, "createBuffer", &desc)
                            .map_err(error)?;
                    #[cfg(test)]
                    super::UPLOAD_PAGE_CREATES.with(|count| count.set(count.get() + 1));
                    UploadPage {
                        raw,
                        cursor: 0,
                        size: page_size,
                        retained: false,
                    }
                }
            };
            encoder.upload_pages.push(page);
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
