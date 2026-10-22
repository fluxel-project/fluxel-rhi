//! Native Metal pipeline binary archives.
//!
//! Metal consumes and emits archives through file URLs. Each temporary
//! directory is created exclusively, is private to this process, and lives as
//! long as a loaded archive. A mutex serializes archive mutation, pipeline
//! creation against it, and serialization; no pipeline state is CPU replayed.
use super::{Metal, MetalDevice};
use fluxel_rhi_core::api::{CacheBackend, Error, Result};

/// Cloneable owner of a native Metal binary archive.
#[derive(Clone)]
pub struct MetalPipelineCache {
    #[cfg(target_vendor = "apple")]
    inner: std::sync::Arc<std::sync::Mutex<CacheInner>>,
}

#[cfg(target_vendor = "apple")]
struct CacheInner {
    archive: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBinaryArchive>>,
    // Metal may retain the URL and access archive bytes after construction.
    _input: Option<TemporaryArchive>,
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
                inner: std::sync::Arc::new(std::sync::Mutex::new(CacheInner {
                    archive,
                    _input: input,
                })),
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
            let inner = cache
                .inner
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if !inner
                .archive
                .respondsToSelector(objc2::sel!(serializeToURL:error:))
            {
                return Err(Error::Unsupported(
                    "Metal binary archive serialization on this OS",
                ));
            }
            let temp = TemporaryArchive::new()?;
            let url = temp.url()?;
            inner.archive.serializeToURL_error(&url).map_err(|error| {
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
    pub(super) fn with_render<T>(
        &self,
        desc: &objc2_metal::MTLRenderPipelineDescriptor,
        create: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        use objc2::runtime::NSObjectProtocol as _;
        use objc2_metal::MTLBinaryArchive as _;
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if !inner
            .archive
            .respondsToSelector(objc2::sel!(addRenderPipelineFunctionsWithDescriptor:error:))
        {
            return Err(Error::Unsupported("Metal render binary archive on this OS"));
        }
        inner
            .archive
            .addRenderPipelineFunctionsWithDescriptor_error(desc)
            .map_err(|error| {
                Error::Backend(format!("Metal archive render functions: {error:?}"))
            })?;
        desc.setBinaryArchives(Some(&objc2_foundation::NSArray::from_slice(&[
            &*inner.archive
        ])));
        create()
    }
    pub(super) fn with_compute<T>(
        &self,
        desc: &objc2_metal::MTLComputePipelineDescriptor,
        create: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        use objc2::runtime::NSObjectProtocol as _;
        use objc2_metal::MTLBinaryArchive as _;
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if !inner
            .archive
            .respondsToSelector(objc2::sel!(addComputePipelineFunctionsWithDescriptor:error:))
        {
            return Err(Error::Unsupported(
                "Metal compute binary archive on this OS",
            ));
        }
        inner
            .archive
            .addComputePipelineFunctionsWithDescriptor_error(desc)
            .map_err(|error| {
                Error::Backend(format!("Metal archive compute functions: {error:?}"))
            })?;
        desc.setBinaryArchives(Some(&objc2_foundation::NSArray::from_slice(&[
            &*inner.archive
        ])));
        create()
    }
    pub(super) fn with_mesh<T>(
        &self,
        desc: &objc2_metal::MTLMeshRenderPipelineDescriptor,
        create: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        use objc2::runtime::NSObjectProtocol as _;
        use objc2_metal::MTLBinaryArchive as _;
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if !inner
            .archive
            .respondsToSelector(objc2::sel!(addMeshRenderPipelineFunctionsWithDescriptor:error:))
        {
            return Err(Error::Unsupported("Metal mesh binary archive on this OS"));
        }
        inner
            .archive
            .addMeshRenderPipelineFunctionsWithDescriptor_error(desc)
            .map_err(|error| Error::Backend(format!("Metal archive mesh functions: {error:?}")))?;
        desc.setBinaryArchives(Some(&objc2_foundation::NSArray::from_slice(&[
            &*inner.archive
        ])));
        create()
    }
}
