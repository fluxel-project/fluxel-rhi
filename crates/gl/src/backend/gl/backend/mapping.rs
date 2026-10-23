//! CPU-owned GL mapping leases.
//!
//! A short native mapping snapshots the requested buffer range. A write lease
//! uploads flushed ranges immediately; releasing a lease does not publish
//! unflushed writes. A read
//! invalidate refreshes the requested range. No driver pointer escapes a call.
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
        &map.bytes
    }
    fn bytes_mut(map: &mut GlMap) -> &mut [u8] {
        assert_eq!(map.mode, api::MapMode::Write);
        &mut map.bytes
    }
    fn flush(map: &mut GlMap, range: Range<u64>) -> Result<()> {
        validate(map, &range)?;
        if map.mode != api::MapMode::Write {
            return Err(Error::InvalidState);
        }
        write_range(map, range)
    }
    fn invalidate(map: &mut GlMap, range: Range<u64>) -> Result<()> {
        validate(map, &range)?;
        if map.mode != api::MapMode::Read {
            return Err(Error::InvalidState);
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
    if range.start > range.end || range.end > map.bytes.len() as u64 {
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
