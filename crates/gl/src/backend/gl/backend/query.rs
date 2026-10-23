//! Core query adaptation for immediate GL contexts.
//!
//! A query set owns one GL query object per Core slot. Reset is a logical
//! invalidation because GL replaces a slot's value on the next begin. Desktop
//! query buffers resolve in GPU order; other native profiles use a synchronous
//! read and WebGL requires completion and a readiness preflight. No query work
//! is queued or replayed.

use std::ops::Range;
use std::rc::Rc;

use fluxel_rhi_core::api::{self, Error, Result};

use super::spi::native_error;
use super::types::{GlBuffer, GlDevice, GlEncoder, GlQuerySet, GlQuerySetInner};

fn native_profile_enabled(device: &GlDevice) -> bool {
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        let _ = device;
        // WebGL2 core includes boolean occlusion queries.
        return true;
    }
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
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = device;
        false
    }
}

pub(crate) fn query_supported(device: &GlDevice, kind: api::QueryKind) -> bool {
    match kind {
        api::QueryKind::Occlusion => native_profile_enabled(device),
        api::QueryKind::Timestamp => timestamp_bits(device).is_some(),
        api::QueryKind::PipelineStatistics => {
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            {
                return device
                    .0
                    .with_native("query GL pipeline statistics support", |provider| {
                        Ok(provider.supports_pipeline_statistics())
                    })
                    .unwrap_or(false);
            }
            #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
            {
                false
            }
        }
    }
}

pub(crate) fn timestamp_bits(device: &GlDevice) -> Option<u32> {
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        use crate::backend::gl::api::GlCapability;
        let snapshot = device.0.owner.discovery_snapshot();
        let bits = snapshot.limits().query_counter_bits;
        return (snapshot.capabilities().supports(GlCapability::TimerQuery)
            && (1..=64).contains(&bits))
        .then_some(bits);
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::{GlCapability, GlFamilyApi as _, GlFamilyProfile};
        return device
            .0
            .with_native("query GL timestamp support", |provider| {
                let discovery = provider.discovery();
                let bits = discovery.limits().query_counter_bits;
                if matches!(
                    discovery.context().profile(),
                    GlFamilyProfile::Embedded { .. }
                ) {
                    return Ok(provider.gles_timestamp_bits());
                }
                Ok((matches!(
                    discovery.context().profile(),
                    GlFamilyProfile::Desktop { .. }
                ) && discovery.capabilities().supports(GlCapability::TimerQuery)
                    && (1..=64).contains(&bits))
                .then_some(bits))
            })
            .ok()
            .flatten();
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = device;
        None
    }
}

pub(crate) fn create_query_set(device: &GlDevice, desc: &api::QueryDesc) -> Result<GlQuerySet> {
    if desc.count == 0 {
        return Err(Error::Unsupported("GL query kind or count"));
    }
    if !query_supported(device, desc.kind) {
        return Err(Error::Unsupported("GL query kind"));
    }

    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::{GlQueryObjectsApi as _, QueryId};

        let capacity = usize::try_from(desc.count).map_err(|_| Error::OutOfMemory)?;
        let mut queries = Vec::<QueryId>::new();
        let mut statistics_queries = Vec::<QueryId>::new();
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
            if desc.kind == api::QueryKind::PipelineStatistics {
                for _ in 0..capacity.checked_mul(7).ok_or(
                    crate::backend::gl::api::GlError::OutOfMemory {
                        operation: "create-statistics",
                    },
                )? {
                    match provider.create_query() {
                        Ok(query) => statistics_queries.push(query),
                        Err(error) => {
                            for query in queries.drain(..).chain(statistics_queries.drain(..)) {
                                let _ = provider.destroy_query(query);
                            }
                            return Err(error);
                        }
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
            statistics_queries,
            measured: std::cell::RefCell::new(vec![false; capacity]),
        }));
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        let queries = device.0.owner.create_core_queries(desc.count)?;
        let capacity = queries.len();
        return Ok(GlQuerySet::from_inner(GlQuerySetInner {
            state: device.0.clone(),
            kind: desc.kind,
            queries,
            statistics_queries: Vec::new(),
            measured: std::cell::RefCell::new(vec![false; capacity]),
        }));
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = (device, desc);
        Err(Error::Unsupported("GL native occlusion query"))
    }
}

fn slots(set: &GlQuerySet, range: Range<u32>) -> Result<std::ops::Range<usize>> {
    if range.start > range.end {
        return Err(Error::InvalidRange);
    }
    let start = usize::try_from(range.start).map_err(|_| Error::InvalidRange)?;
    let end = usize::try_from(range.end).map_err(|_| Error::InvalidRange)?;
    if end > set.queries.len() {
        return Err(Error::InvalidRange);
    }
    Ok(start..end)
}

pub(crate) type PassTimestamps = Option<(GlQuerySet, Option<u32>, Option<u32>)>;

pub(crate) fn prepare_pass_timestamps(
    encoder: &GlEncoder,
    writes: Option<&api::PassTimestampWrites<'_, super::Gl>>,
) -> Result<PassTimestamps> {
    let Some(writes) = writes else {
        return Ok(None);
    };
    // SAFETY: the borrowed Core descriptor retains the set during this call.
    let set = unsafe { writes.set.native_ref() }.clone();
    if set.kind != api::QueryKind::Timestamp
        || !Rc::ptr_eq(&encoder.state, &set.state)
        || encoder.pass_timestamp_end.is_some()
    {
        return Err(Error::InvalidState);
    }
    if writes.begin.is_some() && writes.begin == writes.end {
        return Err(Error::InvalidState);
    }
    let measured = set.measured.try_borrow().map_err(|_| Error::InvalidState)?;
    for slot in [writes.begin, writes.end].into_iter().flatten() {
        let index = slots(&set, slot..slot.checked_add(1).ok_or(Error::InvalidRange)?)?.start;
        if measured[index] {
            return Err(Error::InvalidState);
        }
    }
    drop(measured);
    Ok(Some((set, writes.begin, writes.end)))
}

pub(crate) fn start_pass_timestamps(
    encoder: &mut GlEncoder,
    timestamps: PassTimestamps,
) -> Result<()> {
    if let Some((set, begin, end)) = timestamps {
        if let Some(slot) = begin {
            write_timestamp(encoder, &set, slot)?;
        }
        encoder.pass_timestamp_end = end.map(|slot| (set, slot));
    }
    Ok(())
}

pub(crate) fn finish_pass_timestamps(encoder: &mut GlEncoder) -> Result<()> {
    if let Some((set, slot)) = encoder.pass_timestamp_end.take() {
        write_timestamp(encoder, &set, slot)?;
    }
    Ok(())
}

pub(crate) fn reset(encoder: &mut GlEncoder, set: &GlQuerySet, range: Range<u32>) -> Result<()> {
    if !Rc::ptr_eq(&encoder.state, &set.state)
        || encoder.occlusion_query_active
        || encoder.statistics_query_active
    {
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

pub(crate) fn write_timestamp(encoder: &mut GlEncoder, set: &GlQuerySet, slot: u32) -> Result<()> {
    if encoder.raster_multiview.is_some() {
        return Err(Error::Unsupported("OVR multiview timer queries"));
    }
    if set.kind != api::QueryKind::Timestamp || !Rc::ptr_eq(&encoder.state, &set.state) {
        return Err(Error::InvalidState);
    }
    let index = slots(set, slot..slot.checked_add(1).ok_or(Error::InvalidRange)?)?.start;
    if set.measured.try_borrow().map_err(|_| Error::InvalidState)?[index] {
        return Err(Error::InvalidState);
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlTimestampQueryApi as _;
        let result = encoder
            .state
            .with_native("write GL timestamp", |provider| {
                provider.query_timestamp(set.queries[index])
            })
            .map_err(native_error);
        if let Err(error) = result {
            encoder.indeterminate = true;
            return Err(error);
        }
        set.measured
            .try_borrow_mut()
            .map_err(|_| Error::InvalidState)?[index] = true;
        encoder.issued = true;
        return Ok(());
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        let result = encoder.state.owner.write_core_timestamp(set.queries[index]);
        if let Err(error) = result {
            encoder.indeterminate = true;
            return Err(error);
        }
        set.measured
            .try_borrow_mut()
            .map_err(|_| Error::InvalidState)?[index] = true;
        encoder.issued = true;
        return Ok(());
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    Err(Error::Unsupported("GL timestamp platform"))
}

pub(crate) fn begin_occlusion(encoder: &mut GlEncoder, set: &GlQuerySet, slot: u32) -> Result<()> {
    if set.kind != api::QueryKind::Occlusion
        || encoder.raster_extent.is_none()
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
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        let result = encoder.state.owner.begin_core_occlusion(set.queries[index]);
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
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
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
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        let result = encoder.state.owner.end_core_occlusion();
        if let Err(error) = result {
            encoder.indeterminate = true;
            return Err(error);
        }
        encoder.occlusion_query_active = false;
        encoder.issued = true;
        return Ok(());
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    Err(Error::Unsupported("GL native occlusion query"))
}

pub(crate) fn begin_statistics(encoder: &mut GlEncoder, set: &GlQuerySet, slot: u32) -> Result<()> {
    if set.kind != api::QueryKind::PipelineStatistics
        || !Rc::ptr_eq(&encoder.state, &set.state)
        || encoder.statistics_query_active
    {
        return Err(Error::InvalidState);
    }
    let index = slots(set, slot..slot.checked_add(1).ok_or(Error::InvalidRange)?)?.start;
    if set.measured.try_borrow().map_err(|_| Error::InvalidState)?[index] {
        return Err(Error::InvalidState);
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        // The caller can release its QuerySet after begin; keep the native
        // objects alive until end or the immediate encoder's abort cleanup.
        encoder.statistics_query_set = Some(set.clone());
        let result = encoder
            .state
            .with_native("begin GL pipeline statistics", |provider| {
                provider
                    .begin_pipeline_statistics(&set.statistics_queries[index * 7..index * 7 + 7])
            })
            .map_err(native_error);
        if let Err(error) = result {
            encoder.indeterminate = true;
            return Err(error);
        }
        set.measured
            .try_borrow_mut()
            .map_err(|_| Error::InvalidState)?[index] = true;
        encoder.statistics_query_active = true;
        encoder.issued = true;
        Ok(())
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = index;
        Err(Error::Unsupported("GL pipeline statistics"))
    }
}

pub(crate) fn end_statistics(encoder: &mut GlEncoder) -> Result<()> {
    if !encoder.statistics_query_active {
        return Err(Error::InvalidState);
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        let result = encoder
            .state
            .with_native("end GL pipeline statistics", |provider| {
                provider.end_pipeline_statistics()
            })
            .map_err(native_error);
        if let Err(error) = result {
            encoder.indeterminate = true;
            return Err(error);
        }
        encoder.statistics_query_active = false;
        encoder.statistics_query_set = None;
        encoder.issued = true;
        Ok(())
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        Err(Error::Unsupported("GL pipeline statistics"))
    }
}

pub(crate) fn resolve(
    encoder: &mut GlEncoder,
    set: &GlQuerySet,
    range: Range<u32>,
    destination: &GlBuffer,
    offset: u64,
) -> Result<()> {
    if encoder.occlusion_query_active
        || encoder.statistics_query_active
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
        .checked_mul(set.kind.result_stride())
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
                if set.kind == api::QueryKind::PipelineStatistics {
                    let zeros = vec![
                        0;
                        usize::try_from(byte_len).map_err(|_| {
                            crate::backend::gl::api::GlError::OutOfMemory {
                                operation: "resolve-statistics",
                            }
                        })?
                    ];
                    provider.upload_buffer(buffer_range, &zeros)?;
                    for (slot_index, index) in range.clone().enumerate() {
                        for (j, field) in [0_u64, 1, 2, 5, 6, 7, 10].into_iter().enumerate() {
                            let dest = crate::backend::gl::api::GlBufferRange {
                                buffer: destination.id,
                                offset: offset + slot_index as u64 * 88 + field * 8,
                                size: 8,
                            };
                            let query = set.statistics_queries[index * 7 + j];
                            if !provider.resolve_query_buffer(&[query], dest)? {
                                let value = provider.resolve_query_result_blocking(query)?;
                                provider.upload_buffer(dest, &value.to_ne_bytes())?;
                            }
                        }
                    }
                    return Ok(());
                }
                if provider.resolve_query_buffer(&set.queries[range.clone()], buffer_range)? {
                    return Ok(());
                }
                let mut values = Vec::with_capacity(range.len());
                for query in &set.queries[range.clone()] {
                    values.push(provider.resolve_query_result_blocking(*query)?);
                }
                // A later result can consume the context-wide disjoint flag and
                // invalidate earlier slots. Re-read those slot facts after the
                // whole batch so newly written epochs remain independent.
                if set.kind == api::QueryKind::Timestamp
                    && provider.gles_timestamp_bits().is_some()
                    && values.contains(&u64::MAX)
                {
                    for (value, query) in values.iter_mut().zip(&set.queries[range.clone()]) {
                        *value = provider.resolve_query_result_blocking(*query)?;
                    }
                }
                let mut bytes = Vec::with_capacity(usize::try_from(byte_len).unwrap_or(0));
                for value in values {
                    bytes.extend_from_slice(&value.to_ne_bytes());
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
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        let answers = encoder
            .state
            .owner
            .core_query_results(&set.queries[range])?;
        let mut bytes = Vec::with_capacity(answers.len() * 8);
        for answer in answers {
            bytes.extend_from_slice(&answer.to_ne_bytes());
        }
        let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
        if let Err(error) = browser.write_browser_buffer(&destination.browser, offset, &bytes) {
            encoder.indeterminate = true;
            return Err(error);
        }
        encoder.issued = true;
        return Ok(());
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    Err(Error::Unsupported("GL native occlusion query"))
}
