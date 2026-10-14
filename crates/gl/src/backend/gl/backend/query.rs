//! Core occlusion-query adaptation for immediate GL contexts.
//!
//! A query set owns one GL query object per Core slot. Reset is a logical
//! invalidation because GL replaces a slot's value on the next begin; resolve
//! synchronously observes `QUERY_RESULT` and writes the public u64 layout into
//! the destination buffer. No query work is queued or replayed.

use std::ops::Range;
use std::rc::Rc;

use fluxel_rhi_core::api::{self, Error, Result};

use super::spi::native_error;
use super::types::{GlBuffer, GlDevice, GlEncoder, GlQuerySet, GlQuerySetInner};

fn native_profile_enabled(device: &GlDevice) -> bool {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::{GlCapability, GlFamilyApi as _};
        return device
            .0
            .with_native("query GL occlusion support", |provider| {
                Ok(provider
                    .discovery()
                    .capabilities()
                    .supports(GlCapability::OcclusionQuery))
            })
            .unwrap_or(false);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = device;
        false
    }
}

pub(crate) fn query_supported(device: &GlDevice, kind: api::QueryKind) -> bool {
    matches!(kind, api::QueryKind::Occlusion) && native_profile_enabled(device)
}

pub(crate) fn create_query_set(device: &GlDevice, desc: &api::QueryDesc) -> Result<GlQuerySet> {
    if !matches!(desc.kind, api::QueryKind::Occlusion) || desc.count == 0 {
        return Err(Error::Unsupported("GL query kind or count"));
    }
    if !native_profile_enabled(device) {
        return Err(Error::Unsupported("GL native occlusion query"));
    }

    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::{GlQueryObjectsApi as _, QueryId};

        let capacity = usize::try_from(desc.count).map_err(|_| Error::OutOfMemory)?;
        let mut queries = Vec::<QueryId>::new();
        queries
            .try_reserve_exact(capacity)
            .map_err(|_| Error::OutOfMemory)?;
        let result = device.0.with_native("create GL query set", |provider| {
            for _ in 0..desc.count {
                match provider.create_query() {
                    Ok(query) => queries.push(query),
                    Err(error) => {
                        for query in queries.drain(..) {
                            let _ = provider.destroy_query(query);
                        }
                        return Err(error);
                    }
                }
            }
            Ok(())
        });
        result.map_err(native_error)?;
        return Ok(GlQuerySet::from_inner(GlQuerySetInner {
            state: device.0.clone(),
            kind: desc.kind,
            queries,
            measured: std::cell::RefCell::new(vec![false; capacity]),
        }));
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = (device, desc);
        Err(Error::Unsupported("GL native occlusion query"))
    }
}

fn slots(set: &GlQuerySet, range: Range<u32>) -> Result<std::ops::Range<usize>> {
    if !matches!(set.kind, api::QueryKind::Occlusion) || range.start > range.end {
        return Err(Error::InvalidRange);
    }
    let start = usize::try_from(range.start).map_err(|_| Error::InvalidRange)?;
    let end = usize::try_from(range.end).map_err(|_| Error::InvalidRange)?;
    if end > set.queries.len() {
        return Err(Error::InvalidRange);
    }
    Ok(start..end)
}

pub(crate) fn reset(encoder: &mut GlEncoder, set: &GlQuerySet, range: Range<u32>) -> Result<()> {
    if !Rc::ptr_eq(&encoder.state, &set.state) || encoder.occlusion_query_active {
        return Err(Error::InvalidState);
    }
    let range = slots(set, range)?;
    let mut measured = set
        .measured
        .try_borrow_mut()
        .map_err(|_| Error::InvalidState)?;
    measured[range].fill(false);
    Ok(())
}

pub(crate) fn write_timestamp(_: &mut GlEncoder, _: &GlQuerySet, _: u32) -> Result<()> {
    Err(Error::Unsupported("GL timestamp queries through Core"))
}

pub(crate) fn begin_occlusion(encoder: &mut GlEncoder, set: &GlQuerySet, slot: u32) -> Result<()> {
    if encoder.raster_extent.is_none()
        || encoder.occlusion_query_active
        || !encoder
            .occlusion_query_set
            .as_ref()
            .is_some_and(|active| Rc::ptr_eq(&active.0, &set.0))
        || !Rc::ptr_eq(&encoder.state, &set.state)
    {
        return Err(Error::InvalidState);
    }
    let index = slots(set, slot..slot.checked_add(1).ok_or(Error::InvalidRange)?)?.start;
    if set.measured.try_borrow().map_err(|_| Error::InvalidState)?[index] {
        return Err(Error::InvalidState);
    }

    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlOcclusionQueryApi as _;
        let result = encoder
            .state
            .with_native("begin GL occlusion query", |provider| {
                provider.begin_occlusion_query(set.queries[index])
            })
            .map_err(native_error);
        if let Err(error) = result {
            encoder.indeterminate = true;
            return Err(error);
        }
        set.measured
            .try_borrow_mut()
            .map_err(|_| Error::InvalidState)?[index] = true;
        encoder.occlusion_query_active = true;
        encoder.issued = true;
        return Ok(());
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    Err(Error::Unsupported("GL native occlusion query"))
}

pub(crate) fn end_occlusion(encoder: &mut GlEncoder) -> Result<()> {
    if !encoder.occlusion_query_active || encoder.raster_extent.is_none() {
        return Err(Error::InvalidState);
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlOcclusionQueryApi as _;
        let result = encoder
            .state
            .with_native("end GL occlusion query", |provider| {
                provider.end_occlusion_query()
            })
            .map_err(native_error);
        if let Err(error) = result {
            encoder.indeterminate = true;
            return Err(error);
        }
        encoder.occlusion_query_active = false;
        encoder.issued = true;
        return Ok(());
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    Err(Error::Unsupported("GL native occlusion query"))
}

pub(crate) fn resolve(
    encoder: &mut GlEncoder,
    set: &GlQuerySet,
    range: Range<u32>,
    destination: &GlBuffer,
    offset: u64,
) -> Result<()> {
    if encoder.occlusion_query_active
        || !Rc::ptr_eq(&encoder.state, &set.state)
        || !Rc::ptr_eq(&encoder.state, &destination.state)
        || !destination
            .desc
            .usage
            .contains(crate::backend::gl::api::GlBufferUsage::COPY_DESTINATION)
        || offset % 8 != 0
    {
        return Err(Error::InvalidState);
    }
    let range = slots(set, range)?;
    let byte_len = u64::try_from(range.len())
        .map_err(|_| Error::InvalidRange)?
        .checked_mul(8)
        .ok_or(Error::InvalidRange)?;
    if offset
        .checked_add(byte_len)
        .is_none_or(|end| end > destination.desc.size)
    {
        return Err(Error::InvalidRange);
    }
    if !set.measured.try_borrow().map_err(|_| Error::InvalidState)?[range.clone()]
        .iter()
        .all(|measured| *measured)
    {
        return Err(Error::InvalidState);
    }

    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlCopyDomainApi as _;
        let buffer_range = crate::backend::gl::api::GlBufferRange {
            buffer: destination.id,
            offset,
            size: byte_len,
        };
        let result = encoder
            .state
            .with_native("resolve GL occlusion queries", |provider| {
                let mut bytes = Vec::with_capacity(usize::try_from(byte_len).unwrap_or(0));
                for query in &set.queries[range] {
                    bytes.extend_from_slice(
                        &provider
                            .resolve_query_result_blocking(*query)?
                            .to_le_bytes(),
                    );
                }
                provider.upload_buffer(buffer_range, &bytes)
            });
        if let Err(error) = result {
            encoder.indeterminate = true;
            return Err(native_error(error));
        }
        encoder.issued = true;
        return Ok(());
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    Err(Error::Unsupported("GL native occlusion query"))
}
