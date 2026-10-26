//! Native Metal pipeline binary archives.
//!
//! Metal consumes and emits archives through file URLs. Each temporary
//! directory is created exclusively, is private to this process, and lives as
//! long as a loaded archive. A read/write lease serializes archive mutation and
//! serialization against native archive use. Pipeline creation retains a read
//! lease through its native completion callback, so it never races a mutable
//! archive operation while the archive mutex itself is not held for compilation.
use super::{Metal, MetalDevice};
use fluxel_rhi_core::api::{Error, Result};
use fluxel_rhi_core::backend::CacheBackend;

/// Cloneable owner of a native Metal binary archive.
#[derive(Clone)]
pub struct MetalPipelineCache {
    #[cfg(target_vendor = "apple")]
    inner: std::sync::Arc<CacheInner>,
    #[cfg(target_vendor = "apple")]
    access: std::sync::Arc<ArchiveAccess>,
}

#[cfg(all(test, target_vendor = "apple", feature = "hardware-tests"))]
#[path = "../tests/backend/cache/archive_prepare_benchmark.rs"]
mod archive_prepare_benchmark;

/// Keeps a binary archive (and its backing input URL) alive after descriptor
/// setup has released the write lease.
///
/// Pipeline creation owns this lease until its native completion callback. The
/// archive itself remains attached to the native descriptor, while retaining
/// the cache here also preserves the temporary source file that Metal may
/// access lazily. The read lease prevents archive mutation or serialization
/// during cold native compilation without holding an exclusive mutex.
#[cfg(target_vendor = "apple")]
pub(super) struct PreparedArchive {
    _cache: MetalPipelineCache,
    _read: ArchiveReadLease,
}

#[cfg(target_vendor = "apple")]
struct CacheInner {
    archive: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBinaryArchive>>,
    // Metal may retain the URL and access archive bytes after construction.
    _input: Option<TemporaryArchive>,
}

/// A small ownership gate around one non-`Send`/non-`Sync` native archive.
///
/// `objc2-metal` deliberately does not declare `MTLBinaryArchive` as
/// `Send + Sync`, and Apple's archive documentation does not promise that
/// adding/serializing can overlap descriptor consumption. The gate therefore
/// treats native creation as a read use and add/serialize as exclusive writes.
/// It is intentionally separate from the native archive so a read lease can
/// span an asynchronous completion without keeping a mutex locked.
#[cfg(target_vendor = "apple")]
struct ArchiveAccess {
    state: std::sync::Mutex<ArchiveAccessState>,
    changed: std::sync::Condvar,
}

#[cfg(target_vendor = "apple")]
struct ArchiveAccessState {
    readers: usize,
    writer: bool,
    next_waiter: u64,
    writers: std::collections::VecDeque<(u64, std::task::Waker)>,
}

#[cfg(target_vendor = "apple")]
struct ArchiveWriteLease(Option<std::sync::Arc<ArchiveAccess>>);

#[cfg(target_vendor = "apple")]
struct ArchiveReadLease(std::sync::Arc<ArchiveAccess>);

#[cfg(target_vendor = "apple")]
struct ArchiveWriteFuture {
    access: std::sync::Arc<ArchiveAccess>,
    registration: Option<u64>,
}

#[cfg(target_vendor = "apple")]
impl ArchiveAccess {
    fn new() -> Self {
        Self {
            state: std::sync::Mutex::new(ArchiveAccessState {
                readers: 0,
                writer: false,
                next_waiter: 0,
                writers: std::collections::VecDeque::new(),
            }),
            changed: std::sync::Condvar::new(),
        }
    }

    fn write(self: &std::sync::Arc<Self>) -> ArchiveWriteFuture {
        ArchiveWriteFuture {
            access: self.clone(),
            registration: None,
        }
    }

    /// Used only by synchronous cache serialization. Public cold-pipeline
    /// admission uses `write` and never parks a
    /// task executor thread.
    fn write_blocking(self: &std::sync::Arc<Self>) -> ArchiveWriteLease {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        while state.writer || state.readers != 0 {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(|poison| poison.into_inner());
        }
        state.writer = true;
        ArchiveWriteLease(Some(self.clone()))
    }

    fn wake_next_writer(&self) {
        let waker = {
            let state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            state.writers.front().map(|(_, waker)| waker.clone())
        };
        if let Some(waker) = waker {
            // A foreign completion callback may drop the final read lease.
            // Never allow an executor's waker panic to unwind through ObjC.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| waker.wake()));
        }
    }
}

#[cfg(target_vendor = "apple")]
impl std::future::Future for ArchiveWriteFuture {
    type Output = Result<ArchiveWriteLease>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let access = self.access.clone();
        let mut state = access
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let registration = match self.registration {
            Some(value) => value,
            None if !state.writer && state.readers == 0 && state.writers.is_empty() => {
                state.writer = true;
                return std::task::Poll::Ready(Ok(ArchiveWriteLease(Some(access.clone()))));
            }
            None => {
                let value = state.next_waiter;
                let Some(next) = value.checked_add(1) else {
                    return std::task::Poll::Ready(Err(Error::InvalidState));
                };
                state.next_waiter = next;
                state.writers.push_back((value, context.waker().clone()));
                self.registration = Some(value);
                value
            }
        };
        if !state.writer
            && state.readers == 0
            && state
                .writers
                .front()
                .is_some_and(|(value, _)| *value == registration)
        {
            state.writers.pop_front();
            state.writer = true;
            self.registration = None;
            return std::task::Poll::Ready(Ok(ArchiveWriteLease(Some(access.clone()))));
        }
        if let Some((_, waker)) = state
            .writers
            .iter_mut()
            .find(|(value, _)| *value == registration)
        {
            if !waker.will_wake(context.waker()) {
                *waker = context.waker().clone();
            }
        }
        std::task::Poll::Pending
    }
}

#[cfg(target_vendor = "apple")]
impl Drop for ArchiveWriteFuture {
    fn drop(&mut self) {
        let Some(registration) = self.registration.take() else {
            return;
        };
        let was_front = {
            let mut state = self
                .access
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            let was_front = state
                .writers
                .front()
                .is_some_and(|(value, _)| *value == registration);
            state.writers.retain(|(value, _)| *value != registration);
            was_front
        };
        if was_front {
            self.access.wake_next_writer();
        }
    }
}

#[cfg(target_vendor = "apple")]
impl ArchiveWriteLease {
    fn into_read(mut self) -> ArchiveReadLease {
        let access = self
            .0
            .take()
            .expect("writer lease owns archive access until read conversion");
        let mut state = access
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        debug_assert!(state.writer);
        state.writer = false;
        state.readers += 1;
        access.changed.notify_all();
        drop(state);
        ArchiveReadLease(access)
    }
}

#[cfg(target_vendor = "apple")]
impl Drop for ArchiveWriteLease {
    fn drop(&mut self) {
        let Some(access) = self.0.take() else {
            return;
        };
        let mut state = access
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        debug_assert!(state.writer);
        state.writer = false;
        access.changed.notify_all();
        drop(state);
        access.wake_next_writer();
    }
}

#[cfg(target_vendor = "apple")]
impl Clone for ArchiveReadLease {
    fn clone(&self) -> Self {
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        debug_assert!(!state.writer && state.readers != 0);
        state.readers += 1;
        Self(self.0.clone())
    }
}

#[cfg(target_vendor = "apple")]
impl Drop for ArchiveReadLease {
    fn drop(&mut self) {
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        debug_assert!(state.readers != 0);
        state.readers -= 1;
        if state.readers == 0 {
            self.0.changed.notify_all();
            drop(state);
            self.0.wake_next_writer();
        }
    }
}

#[cfg(target_vendor = "apple")]
impl Clone for PreparedArchive {
    fn clone(&self) -> Self {
        Self {
            _cache: self._cache.clone(),
            _read: self._read.clone(),
        }
    }
}

#[cfg(target_vendor = "apple")]
struct TemporaryArchive {
    directory: std::path::PathBuf,
    file: std::path::PathBuf,
}

#[cfg(target_vendor = "apple")]
impl TemporaryArchive {
    fn new() -> Result<Self> {
        use std::os::unix::fs::DirBuilderExt;
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir();
        for _ in 0..128 {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let directory = base.join(format!("fluxel-metal-archive-{}-{id}", std::process::id()));
            match std::fs::DirBuilder::new().mode(0o700).create(&directory) {
                Ok(()) => {
                    return Ok(Self {
                        file: directory.join("archive.metallib"),
                        directory,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(Error::Backend(format!(
                        "Metal archive temporary directory: {error}"
                    )));
                }
            }
        }
        Err(Error::Backend(
            "Metal archive temporary directory collision".into(),
        ))
    }
    fn url(&self) -> Result<objc2::rc::Retained<objc2_foundation::NSURL>> {
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(self.file.as_os_str().as_bytes())
            .map_err(|_| Error::InvalidState)?;
        let pointer =
            std::ptr::NonNull::new(path.as_ptr().cast_mut()).ok_or(Error::InvalidState)?;
        // SAFETY: pointer addresses a NUL-terminated path for the duration of
        // the synchronous Foundation call; Foundation constructs an owned URL.
        Ok(unsafe {
            objc2_foundation::NSURL::fileURLWithFileSystemRepresentation_isDirectory_relativeToURL(
                pointer, false, None,
            )
        })
    }
    fn write(&self, data: &[u8]) -> Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.file)
            .map_err(|error| Error::Backend(format!("Metal archive input: {error}")))?;
        file.write_all(data)
            .map_err(|error| Error::Backend(format!("Metal archive input: {error}")))
    }
}
#[cfg(target_vendor = "apple")]
impl Drop for TemporaryArchive {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.file);
        let _ = std::fs::remove_dir(&self.directory);
    }
}

impl CacheBackend for Metal {
    fn create_cache(device: &MetalDevice, data: Option<&[u8]>) -> Result<MetalPipelineCache> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2::runtime::NSObjectProtocol as _;
            use objc2_metal::{MTLBinaryArchiveDescriptor, MTLDevice as _};
            let native = &device._state._device;
            if !native.respondsToSelector(objc2::sel!(newBinaryArchiveWithDescriptor:error:)) {
                return Err(Error::Unsupported("Metal binary archives on this OS"));
            }
            let descriptor = MTLBinaryArchiveDescriptor::new();
            let input = if let Some(data) = data {
                let temp = TemporaryArchive::new()?;
                temp.write(data)?;
                let url = temp.url()?;
                descriptor.setUrl(Some(&url));
                Some(temp)
            } else {
                None
            };
            let archive = native
                .newBinaryArchiveWithDescriptor_error(&descriptor)
                .map_err(|error| Error::Backend(format!("Metal archive creation: {error:?}")))?;
            Ok(MetalPipelineCache {
                inner: std::sync::Arc::new(CacheInner {
                    archive,
                    _input: input,
                }),
                access: std::sync::Arc::new(ArchiveAccess::new()),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, data);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn serialize_cache(cache: &MetalPipelineCache) -> Result<Vec<u8>> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2::runtime::NSObjectProtocol as _;
            use objc2_metal::MTLBinaryArchive as _;
            let _write = cache.access.write_blocking();
            if !cache
                .inner
                .archive
                .respondsToSelector(objc2::sel!(serializeToURL:error:))
            {
                return Err(Error::Unsupported(
                    "Metal binary archive serialization on this OS",
                ));
            }
            let temp = TemporaryArchive::new()?;
            let url = temp.url()?;
            cache
                .inner
                .archive
                .serializeToURL_error(&url)
                .map_err(|error| {
                    Error::Backend(format!("Metal archive serialization: {error:?}"))
                })?;
            std::fs::read(&temp.file)
                .map_err(|error| Error::Backend(format!("Metal archive output: {error}")))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = cache;
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
}

#[cfg(target_vendor = "apple")]
impl MetalPipelineCache {
    /// Adds one render descriptor to the shared archive and attaches that same
    /// archive to it. The expensive native pipeline creation must run only
    /// after this returns, while retaining the returned read lease.
    pub(super) async fn prepare_render(
        &self,
        desc: &objc2_metal::MTLRenderPipelineDescriptor,
    ) -> Result<PreparedArchive> {
        let write = self.access.write().await?;
        self.attach_render(desc, write)
    }

    fn attach_render(
        &self,
        desc: &objc2_metal::MTLRenderPipelineDescriptor,
        write: ArchiveWriteLease,
    ) -> Result<PreparedArchive> {
        use objc2::runtime::NSObjectProtocol as _;
        use objc2_metal::MTLBinaryArchive as _;
        if !self
            .inner
            .archive
            .respondsToSelector(objc2::sel!(addRenderPipelineFunctionsWithDescriptor:error:))
        {
            return Err(Error::Unsupported("Metal render binary archive on this OS"));
        }
        self.inner
            .archive
            .addRenderPipelineFunctionsWithDescriptor_error(desc)
            .map_err(|error| {
                Error::Backend(format!("Metal archive render functions: {error:?}"))
            })?;
        desc.setBinaryArchives(Some(&objc2_foundation::NSArray::from_slice(&[&*self
            .inner
            .archive])));
        Ok(PreparedArchive {
            _cache: self.clone(),
            _read: write.into_read(),
        })
    }

    /// Adds one compute descriptor to the archive and returns the ownership
    /// lease required through native compilation completion.
    pub(super) async fn prepare_compute(
        &self,
        desc: &objc2_metal::MTLComputePipelineDescriptor,
    ) -> Result<PreparedArchive> {
        let write = self.access.write().await?;
        self.attach_compute(desc, write)
    }

    fn attach_compute(
        &self,
        desc: &objc2_metal::MTLComputePipelineDescriptor,
        write: ArchiveWriteLease,
    ) -> Result<PreparedArchive> {
        use objc2::runtime::NSObjectProtocol as _;
        use objc2_metal::MTLBinaryArchive as _;
        if !self
            .inner
            .archive
            .respondsToSelector(objc2::sel!(addComputePipelineFunctionsWithDescriptor:error:))
        {
            return Err(Error::Unsupported(
                "Metal compute binary archive on this OS",
            ));
        }
        self.inner
            .archive
            .addComputePipelineFunctionsWithDescriptor_error(desc)
            .map_err(|error| {
                Error::Backend(format!("Metal archive compute functions: {error:?}"))
            })?;
        desc.setBinaryArchives(Some(&objc2_foundation::NSArray::from_slice(&[&*self
            .inner
            .archive])));
        Ok(PreparedArchive {
            _cache: self.clone(),
            _read: write.into_read(),
        })
    }

    /// Adds one mesh descriptor to the archive and returns the ownership lease
    /// required through native compilation completion.
    pub(super) async fn prepare_mesh(
        &self,
        desc: &objc2_metal::MTLMeshRenderPipelineDescriptor,
    ) -> Result<PreparedArchive> {
        let write = self.access.write().await?;
        self.attach_mesh(desc, write)
    }

    fn attach_mesh(
        &self,
        desc: &objc2_metal::MTLMeshRenderPipelineDescriptor,
        write: ArchiveWriteLease,
    ) -> Result<PreparedArchive> {
        use objc2::runtime::NSObjectProtocol as _;
        use objc2_metal::MTLBinaryArchive as _;
        if !self
            .inner
            .archive
            .respondsToSelector(objc2::sel!(addMeshRenderPipelineFunctionsWithDescriptor:error:))
        {
            return Err(Error::Unsupported("Metal mesh binary archive on this OS"));
        }
        self.inner
            .archive
            .addMeshRenderPipelineFunctionsWithDescriptor_error(desc)
            .map_err(|error| Error::Backend(format!("Metal archive mesh functions: {error:?}")))?;
        desc.setBinaryArchives(Some(&objc2_foundation::NSArray::from_slice(&[&*self
            .inner
            .archive])));
        Ok(PreparedArchive {
            _cache: self.clone(),
            _read: write.into_read(),
        })
    }
}
