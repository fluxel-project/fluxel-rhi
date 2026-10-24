//! GL mapping leases.
//!
//! Desktop GL buffer storage and GLES EXT_buffer_storage use coherent persistent
//! native mappings. Other native profiles and WebGL snapshot through a fenced
//! GPU staging copy, then expose CPU-owned bytes after that fence completes.
//! Snapshot write leases upload flushed ranges immediately; releasing a snapshot
//! does not publish unflushed writes. Read invalidation refreshes the requested
//! snapshot range. Persistent leases instead retain their buffer and context;
//! coherent memory needs no copy on flush/invalidate, but the caller must still
//! establish completion and exclude conflicting GPU accesses while borrowing.
use super::types::{Gl, GlBuffer, GlDevice, GlMap};
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
use super::{
    completion::{GlCompletion, NativeWaitFuture},
    types::GlBufferInner,
};
use fluxel_rhi_core::api::{self, Error, Result};
#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    feature = "native-gles-egl",
    all(target_arch = "wasm32", feature = "webgl2")
))]
use std::rc::Rc;
use std::{
    future::Future,
    ops::Range,
    pin::Pin,
    task::{Context, Poll},
};

/// A native non-persistent map captures the requested bytes at map creation
/// into a private staging buffer, then observes the resulting GL fence from
/// the owner thread.  It never maps a driver buffer while that copy is
/// outstanding, so polling is non-blocking just like a Core completion wait.
pub struct NativeMapFuture(NativeMapFutureState);

enum NativeMapFutureState {
    Ready(Option<Result<GlMap>>),
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    Staged {
        buffer: GlBuffer,
        staging: GlBuffer,
        wait: NativeWaitFuture,
        size: u64,
        offset: u64,
        mode: api::MapMode,
        complete: bool,
    },
}

fn required_usage(mode: api::MapMode) -> crate::backend::gl::api::GlBufferUsage {
    match mode {
        api::MapMode::Read => crate::backend::gl::api::GlBufferUsage::MAP_READ,
        api::MapMode::Write => crate::backend::gl::api::GlBufferUsage::MAP_WRITE,
    }
}

impl NativeMapFuture {
    fn ready(result: Result<GlMap>) -> Self {
        Self(NativeMapFutureState::Ready(Some(result)))
    }
}

impl Future for NativeMapFuture {
    type Output = Result<GlMap>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match &mut self.as_mut().get_mut().0 {
            NativeMapFutureState::Ready(result) => {
                Poll::Ready(result.take().unwrap_or(Err(Error::InvalidState)))
            }
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            NativeMapFutureState::Staged {
                buffer,
                staging,
                wait,
                size,
                offset,
                mode,
                complete,
            } => match Pin::new(wait).poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(Err(error)) => {
                    buffer.map_active().set(false);
                    *complete = true;
                    Poll::Ready(Err(error))
                }
                Poll::Ready(Ok(())) => {
                    use crate::backend::gl::api::{GlBufferRange, GlCopyDomainApi as _};
                    let result = staging
                        .state
                        .with_native("read staged GL mapping", |provider| {
                            provider.read_buffer(GlBufferRange {
                                buffer: staging.id,
                                offset: 0,
                                size: *size,
                            })
                        })
                        .map_err(map_error)
                        .and_then(|bytes| {
                            let len = usize::try_from(*size).map_err(|_| Error::OutOfMemory)?;
                            Ok(GlMap {
                                buffer: buffer.clone(),
                                bytes,
                                offset: *offset,
                                mode: *mode,
                                persistent: None,
                                len,
                            })
                        });
                    if result.is_err() {
                        buffer.map_active().set(false);
                    }
                    *complete = true;
                    Poll::Ready(result)
                }
            },
        }
    }
}

impl Drop for NativeMapFuture {
    fn drop(&mut self) {
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        if let NativeMapFutureState::Staged {
            buffer, complete, ..
        } = &mut self.0
        {
            if !*complete {
                buffer.map_active().set(false);
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub struct WebGlMapFuture {
    buffer: GlBuffer,
    staging: Option<crate::backend::gl::browser::BrowserBuffer>,
    wait: Option<crate::backend::gl::browser::BrowserCompletionFuture>,
    size: u64,
    offset: u64,
    mode: api::MapMode,
    complete: bool,
    failed: Option<Error>,
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
impl Future for WebGlMapFuture {
    type Output = Result<GlMap>;

    fn poll(
        mut self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        if let Some(error) = self.failed.take() {
            return core::task::Poll::Ready(Err(error));
        }
        match core::pin::Pin::new(self.wait.as_mut().expect("browser map wait")).poll(cx) {
            core::task::Poll::Pending => core::task::Poll::Pending,
            core::task::Poll::Ready(Err(error)) => {
                self.buffer.map_active().set(false);
                self.complete = true;
                core::task::Poll::Ready(Err(error))
            }
            core::task::Poll::Ready(Ok(())) => {
                let staging = self.staging.as_ref().expect("browser map staging");
                let len = self.size;
                match self.buffer.state.owner.read_buffer(staging, 0, len) {
                    Ok(bytes) => {
                        self.complete = true;
                        let Ok(len) = usize::try_from(len) else {
                            self.buffer.map_active().set(false);
                            return core::task::Poll::Ready(Err(Error::OutOfMemory));
                        };
                        core::task::Poll::Ready(Ok(GlMap {
                            buffer: self.buffer.clone(),
                            bytes,
                            offset: self.offset,
                            mode: self.mode,
                            len,
                        }))
                    }
                    Err(error) => {
                        self.buffer.map_active().set(false);
                        self.complete = true;
                        core::task::Poll::Ready(Err(error))
                    }
                }
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
impl WebGlMapFuture {
    fn failed(buffer: &GlBuffer, error: Error) -> Self {
        Self {
            buffer: buffer.clone(),
            staging: None,
            wait: None,
            offset: 0,
            size: 0,
            mode: api::MapMode::Read,
            complete: true,
            failed: Some(error),
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
impl Drop for WebGlMapFuture {
    fn drop(&mut self) {
        if !self.complete {
            self.buffer.map_active().set(false);
        }
    }
}

impl api::MappingBackend for Gl {
    type NativeMap = GlMap;
    #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
    type MapFuture<'a> = NativeMapFuture;
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    type MapFuture<'a> = WebGlMapFuture;
    fn map<'a>(
        device: &'a GlDevice,
        buffer: &'a GlBuffer,
        range: Range<u64>,
        mode: api::MapMode,
    ) -> Self::MapFuture<'a> {
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        {
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            {
                if !Rc::ptr_eq(&device.0, &buffer.state) {
                    return NativeMapFuture::ready(Err(Error::WrongDevice));
                }
                if range.start >= range.end || range.end > buffer.desc.size {
                    return NativeMapFuture::ready(Err(Error::InvalidRange));
                }
                if buffer.map_active().get() {
                    return NativeMapFuture::ready(Err(Error::InvalidState));
                }
                let required = required_usage(mode);
                if !buffer.desc.usage.contains(required) {
                    return NativeMapFuture::ready(Err(Error::Unsupported(
                        "buffer lacks requested mapping usage",
                    )));
                }
                let persistent = match device
                    .0
                    .with_native("map GL persistent buffer", |provider| {
                        provider.persistent_buffer_lease(buffer.id, mode == api::MapMode::Write)
                    })
                    .map_err(map_error)
                {
                    Ok(persistent) => persistent,
                    Err(error) => return NativeMapFuture::ready(Err(error)),
                };
                if let Some(ptr) = persistent {
                    let len = match usize::try_from(range.end - range.start) {
                        Ok(len) => len,
                        Err(_) => return NativeMapFuture::ready(Err(Error::InvalidRange)),
                    };
                    let offset = match usize::try_from(range.start) {
                        Ok(offset) => offset,
                        Err(_) => return NativeMapFuture::ready(Err(Error::InvalidRange)),
                    };
                    // SAFETY: creation mapped the entire buffer and range was validated.
                    let ptr = unsafe { core::ptr::NonNull::new_unchecked(ptr.add(offset)) };
                    buffer.map_active().set(true);
                    return NativeMapFuture::ready(Ok(GlMap {
                        buffer: buffer.clone(),
                        bytes: Vec::new(),
                        offset: range.start,
                        mode,
                        persistent: Some(ptr),
                        len,
                    }));
                }

                use crate::backend::gl::api::{
                    GlBufferDesc, GlBufferRange, GlBufferUsage, GlCopyDomainApi as _,
                    GlResourceApi as _,
                };
                let size = range.end - range.start;
                let desc = GlBufferDesc {
                    size,
                    usage: GlBufferUsage::COPY_DESTINATION | GlBufferUsage::MAP_READ,
                };
                let id = match device
                    .0
                    .with_native("create GL map staging", |provider| {
                        provider.create_buffer_resource(desc)
                    })
                    .map_err(map_error)
                {
                    Ok(id) => id,
                    Err(error) => return NativeMapFuture::ready(Err(error)),
                };
                let staging = GlBuffer::from_inner(GlBufferInner {
                    state: device.0.clone(),
                    id,
                    desc,
                });
                let copy = device.0.with_native("stage GL buffer map", |provider| {
                    provider.copy_buffer_range(
                        GlBufferRange {
                            buffer: buffer.id,
                            offset: range.start,
                            size,
                        },
                        GlBufferRange {
                            buffer: staging.id,
                            offset: 0,
                            size,
                        },
                    )
                });
                if let Err(error) = copy {
                    return NativeMapFuture::ready(Err(map_error(error)));
                }
                let completion = match GlCompletion::after_issued(device.0.clone()) {
                    Ok(completion) => completion,
                    Err(error) => return NativeMapFuture::ready(Err(error)),
                };
                buffer.map_active().set(true);
                NativeMapFuture(NativeMapFutureState::Staged {
                    buffer: buffer.clone(),
                    staging,
                    wait: NativeWaitFuture::new(completion),
                    size,
                    offset: range.start,
                    mode,
                    complete: false,
                })
            }
            #[cfg(not(any(
                all(windows, feature = "native-gl-wgl"),
                feature = "native-gles-egl"
            )))]
            return NativeMapFuture::ready(Err(Error::Unsupported("GL native buffer mapping")));
        }
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            if !Rc::ptr_eq(&device.0, &buffer.state) {
                return WebGlMapFuture::failed(buffer, Error::WrongDevice);
            }
            if range.start >= range.end || range.end > buffer.desc.size {
                return WebGlMapFuture::failed(buffer, Error::InvalidRange);
            }
            if buffer.map_active().get() {
                return WebGlMapFuture::failed(buffer, Error::InvalidState);
            }
            let required = required_usage(mode);
            if !buffer.desc.usage.contains(required) {
                return WebGlMapFuture::failed(
                    buffer,
                    Error::Unsupported("buffer lacks requested mapping usage"),
                );
            }
            let size = range.end - range.start;
            match device
                .0
                .owner
                .stage_map_read(&buffer.browser, range.start, size)
            {
                Ok((staging, completion)) => {
                    buffer.map_active().set(true);
                    WebGlMapFuture {
                        buffer: buffer.clone(),
                        staging: Some(staging),
                        wait: Some(completion.wait()),
                        size,
                        offset: range.start,
                        mode,
                        complete: false,
                        failed: None,
                    }
                }
                Err(error) => WebGlMapFuture::failed(buffer, error),
            }
        }
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
