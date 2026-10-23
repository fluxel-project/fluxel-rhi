//! GL mapping leases.
//!
//! Desktop GL buffer storage and GLES EXT_buffer_storage use coherent persistent
//! native mappings. Other native profiles and WebGL use a snapshot that may
//! require a blocking read.
//! Snapshot write leases upload flushed ranges immediately; releasing a snapshot
//! does not publish unflushed writes. Read invalidation refreshes the requested
//! snapshot range. Persistent leases instead retain their buffer and context;
//! coherent memory needs no copy on flush/invalidate, but the caller must still
//! establish completion and exclude conflicting GPU accesses while borrowing.
use super::types::{Gl, GlBuffer, GlDevice, GlMap};
use fluxel_rhi_core::api::{self, Error, Result};
use std::{
    future::{Ready, ready},
    ops::Range,
    rc::Rc,
};

impl api::MappingBackend for Gl {
    type NativeMap = GlMap;
    type MapFuture<'a> = Ready<Result<GlMap>>;
    fn map<'a>(
        device: &'a GlDevice,
        buffer: &'a GlBuffer,
        range: Range<u64>,
        mode: api::MapMode,
    ) -> Self::MapFuture<'a> {
        ready((|| {
            if !Rc::ptr_eq(&device.0, &buffer.state) {
                return Err(Error::WrongDevice);
            }
            if range.start >= range.end || range.end > buffer.desc.size {
                return Err(Error::InvalidRange);
            }
            if buffer.map_active().get() {
                return Err(Error::InvalidState);
            }
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            {
                let persistent = device
                    .0
                    .with_native("map GL persistent buffer", |provider| {
                        provider.persistent_buffer_lease(buffer.id, mode == api::MapMode::Write)
                    })
                    .map_err(map_error)?;
                if let Some(ptr) = persistent {
                    let len = usize::try_from(range.end - range.start)
                        .map_err(|_| Error::InvalidRange)?;
                    let offset = usize::try_from(range.start).map_err(|_| Error::InvalidRange)?;
                    // SAFETY: creation mapped the entire buffer and range was validated.
                    let ptr = unsafe { core::ptr::NonNull::new_unchecked(ptr.add(offset)) };
                    buffer.map_active().set(true);
                    return Ok(GlMap {
                        buffer: buffer.clone(),
                        bytes: Vec::new(),
                        offset: range.start,
                        mode,
                        persistent: Some(ptr),
                        len,
                    });
                }
                let bytes = device
                    .0
                    .with_native("snapshot GL mapping", |provider| {
                        provider.snapshot_buffer_lease(
                            buffer.id,
                            range.start,
                            range.end - range.start,
                            mode == api::MapMode::Write,
                        )
                    })
                    .map_err(map_error)?;
                buffer.map_active().set(true);
                Ok(GlMap {
                    buffer: buffer.clone(),
                    bytes,
                    offset: range.start,
                    mode,
                    persistent: None,
                    len: (range.end - range.start) as usize,
                })
            }
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            {
                use crate::backend::gl::api::GlBufferUsage;
                let required = if mode == api::MapMode::Read {
                    GlBufferUsage::MAP_READ
                } else {
                    GlBufferUsage::MAP_WRITE
                };
                if !buffer.desc.usage.contains(required) {
                    return Err(Error::Unsupported("buffer lacks requested mapping usage"));
                }
                let bytes = device.0.owner.read_buffer(
                    &buffer.browser,
                    range.start,
                    range.end - range.start,
                )?;
                buffer.map_active().set(true);
                Ok(GlMap {
                    buffer: buffer.clone(),
                    bytes,
                    offset: range.start,
                    mode,
                    len: (range.end - range.start) as usize,
                })
            }
            #[cfg(not(any(
                all(windows, feature = "native-gl-wgl"),
                feature = "native-gles-egl",
                all(target_arch = "wasm32", feature = "webgl2")
            )))]
            {
                let _ = mode;
                Err(Error::Unsupported("GL native buffer mapping"))
            }
        })())
    }
    fn bytes(map: &GlMap) -> &[u8] {
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        if let Some(ptr) = map.persistent {
            // SAFETY: the buffer retains the native mapping and the unsafe Core
            // contract requires the caller to exclude conflicting GPU writes.
            return unsafe { core::slice::from_raw_parts(ptr.as_ptr(), map.len) };
        }
        &map.bytes
    }
    fn bytes_mut(map: &mut GlMap) -> &mut [u8] {
        assert_eq!(map.mode, api::MapMode::Write);
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        if let Some(ptr) = map.persistent {
            // SAFETY: the lease is unique and caller excludes GPU accesses.
            return unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr(), map.len) };
        }
        &mut map.bytes
    }
    fn flush(map: &mut GlMap, range: Range<u64>) -> Result<()> {
        validate(map, &range)?;
        if map.mode != api::MapMode::Write {
            return Err(Error::InvalidState);
        }
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        if map.persistent.is_some() {
            return Ok(());
        }
        write_range(map, range)
    }
    fn invalidate(map: &mut GlMap, range: Range<u64>) -> Result<()> {
        validate(map, &range)?;
        if map.mode != api::MapMode::Read {
            return Err(Error::InvalidState);
        }
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        if map.persistent.is_some() {
            return Ok(());
        }
        if range.is_empty() {
            return Ok(());
        }
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            let bytes = map
                .buffer
                .state
                .with_native("invalidate GL mapping", |provider| {
                    provider.snapshot_buffer_lease(
                        map.buffer.id,
                        map.offset + range.start,
                        range.end - range.start,
                        false,
                    )
                })
                .map_err(map_error)?;
            map.bytes[range.start as usize..range.end as usize].copy_from_slice(&bytes);
            Ok(())
        }
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            let bytes = map.buffer.state.owner.read_buffer(
                &map.buffer.browser,
                map.offset + range.start,
                range.end - range.start,
            )?;
            map.bytes[range.start as usize..range.end as usize].copy_from_slice(&bytes);
            Ok(())
        }
        #[cfg(not(any(
            all(windows, feature = "native-gl-wgl"),
            feature = "native-gles-egl",
            all(target_arch = "wasm32", feature = "webgl2")
        )))]
        {
            Err(Error::Unsupported("GL native buffer mapping"))
        }
    }
    fn unmap(map: GlMap) {
        drop(map);
    }
}
impl Drop for GlMap {
    fn drop(&mut self) {
        self.buffer.map_active().set(false);
    }
}
fn validate(map: &GlMap, range: &Range<u64>) -> Result<()> {
    if range.start > range.end || range.end > map.len as u64 {
        Err(Error::InvalidRange)
    } else {
        Ok(())
    }
}
fn write_range(map: &GlMap, range: Range<u64>) -> Result<()> {
    if range.is_empty() {
        return Ok(());
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::{GlBufferRange, GlCopyDomainApi as _};
        map.buffer
            .state
            .with_native("flush GL mapping", |provider| {
                provider.upload_buffer(
                    GlBufferRange {
                        buffer: map.buffer.id,
                        offset: map.offset + range.start,
                        size: range.end - range.start,
                    },
                    &map.bytes[range.start as usize..range.end as usize],
                )
            })
            .map_err(map_error)
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        map.buffer.state.owner.write_buffer_lease(
            &map.buffer.browser,
            map.offset + range.start,
            &map.bytes[range.start as usize..range.end as usize],
        )
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        Err(Error::Unsupported("GL native buffer mapping"))
    }
}
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
fn map_error(error: crate::backend::gl::api::GlError) -> Error {
    match error {
        crate::backend::gl::api::GlError::Unsupported { reason, .. } => Error::Unsupported(reason),
        crate::backend::gl::api::GlError::Validation { .. } => Error::InvalidState,
        other => super::spi::native_error(other),
    }
}
