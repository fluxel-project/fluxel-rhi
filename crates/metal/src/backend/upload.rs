//! Persistently allocated Metal staging pages.
//!
//! Idle pages contain only native `MTLBuffer` objects, never an `Arc` back to
//! `DeviceState`. An encoder owns slices while recording; submitted work moves
//! those slices into a command-buffer completion handler, so a page cannot
//! re-enter the idle pool before Metal reports terminal completion.

#[cfg(target_vendor = "apple")]
use std::sync::{Arc, Mutex, Weak};

#[cfg(target_vendor = "apple")]
const DEFAULT_PAGE_BYTES: usize = 4 * 1024 * 1024;

#[cfg(target_vendor = "apple")]
#[derive(Clone, Default)]
pub(super) struct UploadPool(Arc<PoolInner>);

#[cfg(not(target_vendor = "apple"))]
#[derive(Clone, Default)]
pub(super) struct UploadPool;

#[cfg(target_vendor = "apple")]
struct PoolInner {
    idle: Mutex<Vec<RawPage>>,
}

#[cfg(target_vendor = "apple")]
impl Default for PoolInner {
    fn default() -> Self {
        Self {
            idle: Mutex::new(Vec::new()),
        }
    }
}

#[cfg(target_vendor = "apple")]
struct RawPage {
    buffer: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    size: usize,
}

// Metal resource objects may be created, encoded, and released from distinct
// threads. Pool access is serialized by its mutex; page contents are written
// only while an encoder exclusively owns a lease, before native submission.
unsafe impl Send for RawPage {}
unsafe impl Sync for RawPage {}

#[cfg(target_vendor = "apple")]
struct Lease {
    pool: Weak<PoolInner>,
    page: Mutex<Option<RawPage>>,
    cursor: Mutex<usize>,
}

#[cfg(target_vendor = "apple")]
impl Drop for Lease {
    fn drop(&mut self) {
        let Some(page) = self.page.lock().unwrap_or_else(|p| p.into_inner()).take() else {
            return;
        };
        if let Some(pool) = self.pool.upgrade() {
            pool.idle
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(page);
        }
    }
}

/// A recorded region of one staging page. Its lease returns the raw page only
/// after every recorded reference, including the native completion handler,
/// has gone away.
#[cfg(target_vendor = "apple")]
#[derive(Clone)]
pub(crate) struct UploadSlice {
    lease: Arc<Lease>,
    pub(crate) buffer:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    pub(crate) offset: usize,
}

#[cfg(target_vendor = "apple")]
impl UploadPool {
    /// Releases only idle native pages. Device teardown invokes this before
    /// its queue/device owners are dropped; submitted leases remain held by
    /// their command-buffer completion handlers.
    pub(super) fn clear(&self) {
        self.0
            .idle
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
    }

    pub(super) fn write(
        &self,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        retained: &mut Vec<UploadSlice>,
        bytes: &[u8],
        alignment: usize,
    ) -> fluxel_rhi_core::api::Result<UploadSlice> {
        if bytes.is_empty() || alignment == 0 || !alignment.is_power_of_two() {
            return Err(fluxel_rhi_core::api::Error::InvalidRange);
        }
        if let Some(last) = retained.last() {
            if let Some(slice) = allocate(&last.lease, bytes.len(), alignment)? {
                copy(&slice, bytes)?;
                retained.push(slice.clone());
                return Ok(slice);
            }
        }
        let minimum = bytes
            .len()
            .checked_add(alignment)
            .ok_or(fluxel_rhi_core::api::Error::InvalidRange)?;
        let page = self.acquire(device, minimum)?;
        let lease = Arc::new(Lease {
            pool: Arc::downgrade(&self.0),
            page: Mutex::new(Some(page)),
            cursor: Mutex::new(0),
        });
        let slice = allocate(&lease, bytes.len(), alignment)?
            .ok_or(fluxel_rhi_core::api::Error::OutOfMemory)?;
        copy(&slice, bytes)?;
        retained.push(slice.clone());
        Ok(slice)
    }

    fn acquire(
        &self,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        minimum: usize,
    ) -> fluxel_rhi_core::api::Result<RawPage> {
        let mut idle = self.0.idle.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(index) = idle.iter().position(|page| page.size >= minimum) {
            return Ok(idle.swap_remove(index));
        }
        drop(idle);
        use objc2_metal::MTLDevice as _;
        let size = DEFAULT_PAGE_BYTES.max(minimum);
        let buffer = device
            .newBufferWithLength_options(size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or(fluxel_rhi_core::api::Error::OutOfMemory)?;
        Ok(RawPage { buffer, size })
    }
}

#[cfg(not(target_vendor = "apple"))]
impl UploadPool {
    pub(super) fn clear(&self) {}
}

#[cfg(target_vendor = "apple")]
fn allocate(
    lease: &Arc<Lease>,
    bytes: usize,
    alignment: usize,
) -> fluxel_rhi_core::api::Result<Option<UploadSlice>> {
    let mut cursor = lease.cursor.lock().unwrap_or_else(|p| p.into_inner());
    let offset = cursor
        .checked_add(alignment - 1)
        .ok_or(fluxel_rhi_core::api::Error::InvalidRange)?
        & !(alignment - 1);
    let end = offset
        .checked_add(bytes)
        .ok_or(fluxel_rhi_core::api::Error::InvalidRange)?;
    let page = lease.page.lock().unwrap_or_else(|p| p.into_inner());
    let Some(page) = page.as_ref() else {
        return Ok(None);
    };
    if end > page.size {
        return Ok(None);
    }
    *cursor = end;
    Ok(Some(UploadSlice {
        lease: lease.clone(),
        buffer: page.buffer.clone(),
        offset,
    }))
}

#[cfg(target_vendor = "apple")]
fn copy(slice: &UploadSlice, bytes: &[u8]) -> fluxel_rhi_core::api::Result<()> {
    use objc2_metal::MTLBuffer as _;
    let page = slice.lease.page.lock().unwrap_or_else(|p| p.into_inner());
    let Some(page) = page.as_ref() else {
        return Err(fluxel_rhi_core::api::Error::InvalidState);
    };
    // SAFETY: `allocate` reserved this page range to the exclusive recording
    // encoder and the shared MTLBuffer contents cover the complete page.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            page.buffer
                .contents()
                .cast::<u8>()
                .as_ptr()
                .add(slice.offset),
            bytes.len(),
        );
    }
    Ok(())
}
