//! Metal timestamp sampling and submission-ordered slot state.
//!
//! Only Apple-family GPUs with blit-boundary timestamp counters can satisfy
//! the fixed clock contract; Intel/AMD GPU clocks require dynamic calibration.
//! Fixed-layout pipeline statistics remain unsupported because Metal's counters
//! do not define Core's eleven fields. The separate pass-boundary statistics
//! path exposes the available native counters and their sampling failures.
//! Neither path has been validated on an Apple GPU.
use super::{Metal, MetalBuffer, MetalCommandEncoder, MetalDevice};
use fluxel_rhi_core::api::{self, Error, QueryDesc, QueryKind, Result};
use fluxel_rhi_core::backend::QueryBackend;
use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

#[cfg(target_vendor = "apple")]
#[path = "query_occlusion.rs"]
mod occlusion;
#[cfg(target_vendor = "apple")]
pub(super) use occlusion::{VisibilityMergeCache, finish_occlusion_segments, split_occlusion};

#[cfg(target_vendor = "apple")]
const _: () = {
    assert!(
        std::mem::size_of::<objc2_metal::MTLCounterResultStatistic>()
            == std::mem::size_of::<api::StatisticsSample>()
    );
    assert!(
        std::mem::offset_of!(
            objc2_metal::MTLCounterResultStatistic,
            tessellationInputPatches
        ) == std::mem::offset_of!(api::StatisticsSample, tessellation_input_patches)
    );
    assert!(
        std::mem::offset_of!(objc2_metal::MTLCounterResultStatistic, vertexInvocations)
            == std::mem::offset_of!(api::StatisticsSample, vertex_invocations)
    );
    assert!(
        std::mem::offset_of!(
            objc2_metal::MTLCounterResultStatistic,
            postTessellationVertexInvocations
        ) == std::mem::offset_of!(api::StatisticsSample, post_tessellation_vertices)
    );
    assert!(
        std::mem::offset_of!(objc2_metal::MTLCounterResultStatistic, clipperInvocations)
            == std::mem::offset_of!(api::StatisticsSample, clipper_invocations)
    );
    assert!(
        std::mem::offset_of!(objc2_metal::MTLCounterResultStatistic, clipperPrimitivesOut)
            == std::mem::offset_of!(api::StatisticsSample, clipper_primitives_out)
    );
    assert!(
        std::mem::offset_of!(objc2_metal::MTLCounterResultStatistic, fragmentInvocations)
            == std::mem::offset_of!(api::StatisticsSample, fragment_invocations)
    );
    assert!(
        std::mem::offset_of!(objc2_metal::MTLCounterResultStatistic, fragmentsPassed)
            == std::mem::offset_of!(api::StatisticsSample, fragments_passed)
    );
    assert!(
        std::mem::offset_of!(
            objc2_metal::MTLCounterResultStatistic,
            computeKernelInvocations
        ) == std::mem::offset_of!(api::StatisticsSample, compute_invocations)
    );
};

#[derive(Clone)]
pub struct MetalQuerySet(Arc<QueryState>);
struct QueryState {
    #[cfg(target_vendor = "apple")]
    raw: Option<
        objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCounterSampleBuffer>,
        >,
    >,
    #[cfg(target_vendor = "apple")]
    visibility:
        Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    submitted: Mutex<Submitted>,
    statistics_pass: Option<api::StatisticsPass>,
}
#[derive(Clone)]
struct Submitted {
    slots: Vec<Option<(u32, bool)>>,
    generations: Vec<Option<Arc<()>>>,
}
pub(super) struct QueryCommit(Vec<(MetalQuerySet, usize, Arc<()>)>);
#[derive(Clone)]
pub(super) enum QueryOp {
    Reset(MetalQuerySet, Range<u32>),
    Write(MetalQuerySet, u32),
    Resolve(MetalQuerySet, Range<u32>),
}
impl QueryOp {
    fn set(&self) -> &MetalQuerySet {
        match self {
            Self::Reset(s, _) | Self::Write(s, _) | Self::Resolve(s, _) => s,
        }
    }
    fn apply(&self, slots: &mut [Option<(u32, bool)>], lane: u32) -> Result<()> {
        match self {
            Self::Reset(_, r) => {
                for i in r.clone() {
                    slots[i as usize] = Some((lane, false));
                }
            }
            Self::Write(_, i) => {
                if !slots[*i as usize].is_some_and(|(_, written)| !written) {
                    return Err(Error::InvalidState);
                }
                slots[*i as usize] = Some((lane, true));
            }
            Self::Resolve(_, r) => {
                if r.clone()
                    .any(|i| !slots[i as usize].is_some_and(|(_, written)| written))
                {
                    return Err(Error::InvalidState);
                }
            }
        }
        Ok(())
    }
}
pub(super) fn validate_ops<'a>(ops: impl Iterator<Item = (u32, &'a QueryOp)>) -> Result<()> {
    let mut sets = std::collections::HashMap::<usize, Vec<Option<(u32, bool)>>>::new();
    for (lane, op) in ops {
        let set = op.set();
        let slots = sets.entry(Arc::as_ptr(&set.0) as usize).or_insert_with(|| {
            set.0
                .submitted
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .slots
                .clone()
        });
        op.apply(slots, lane)?;
    }
    Ok(())
}
pub(super) fn commit_ops(ops: &[QueryOp], lane: u32) -> QueryCommit {
    let generation = Arc::new(());
    let mut changes = Vec::new();
    for op in ops {
        let mut submitted = op
            .set()
            .0
            .submitted
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        op.apply(&mut submitted.slots, lane)
            .expect("preflighted query state");
        let range = match op {
            QueryOp::Reset(_, range) => range.clone(),
            QueryOp::Write(_, index) => *index..*index + 1,
            QueryOp::Resolve(_, _) => continue,
        };
        for index in range {
            submitted.generations[index as usize] = Some(generation.clone());
            changes.push((op.set().clone(), index as usize, generation.clone()));
        }
    }
    QueryCommit(changes)
}
pub(super) fn invalidate_ops(commit: &QueryCommit, lane: u32) {
    for (set, index, generation) in &commit.0 {
        let mut submitted = set.0.submitted.lock().unwrap_or_else(|e| e.into_inner());
        if submitted.generations[*index]
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, generation))
        {
            submitted.slots[*index] = Some((lane, false));
            submitted.generations[*index] = None;
        }
    }
}

#[cfg(target_vendor = "apple")]
pub(super) fn configure_occlusion(
    e: &mut MetalCommandEncoder,
    desc: &objc2_metal::MTLRenderPassDescriptor,
    set: Option<&api::QuerySet<Metal>>,
) -> Result<()> {
    e.occlusion_set = if let Some(set) = set {
        // SAFETY: Core validated query set provenance and kind.
        let native = unsafe { set.native_ref() };
        let buffer = native.0.visibility.as_deref().ok_or(Error::InvalidState)?;
        desc.setVisibilityResultBuffer(Some(buffer));
        Some(native.clone())
    } else {
        None
    };
    e.occlusion_active = false;
    e.occlusion_slot = None;
    e.occlusion_segments.clear();
    Ok(())
}

#[cfg(target_vendor = "apple")]
fn supported(device: &MetalDevice) -> bool {
    use objc2_foundation::NSObjectProtocol as _;
    use objc2_metal::{
        MTLCommonCounterSetTimestamp, MTLCommonCounterTimestamp, MTLCounter as _,
        MTLCounterSamplingPoint, MTLCounterSet as _, MTLDevice as _, MTLGPUFamily,
    };
    if !device
        ._state
        ._device
        .respondsToSelector(objc2::sel!(supportsFamily:))
        || !device
            ._state
            ._device
            .respondsToSelector(objc2::sel!(supportsCounterSampling:))
        || !device
            ._state
            ._device
            .respondsToSelector(objc2::sel!(counterSets))
        || !device
            ._state
            ._device
            .respondsToSelector(objc2::sel!(newCounterSampleBufferWithDescriptor:error:))
        || !device._state._device.supportsFamily(MTLGPUFamily::Apple1)
        || !device
            ._state
            ._device
            .supportsCounterSampling(MTLCounterSamplingPoint::AtBlitBoundary)
    {
        return false;
    }
    device._state._device.counterSets().is_some_and(|sets| {
        sets.iter().any(|s| {
            // SAFETY: the Metal framework exports this constant for the current process.
            &*s.name() == unsafe { MTLCommonCounterSetTimestamp }
                && s.counters().iter().any(|counter| {
                    // SAFETY: Metal exports this counter name for the process.
                    &*counter.name() == unsafe { MTLCommonCounterTimestamp }
                })
        })
    })
}
#[cfg(target_vendor = "apple")]
fn timebase() -> Option<f64> {
    #[repr(C)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }
    unsafe extern "C" {
        fn mach_timebase_info(info: *mut Timebase) -> i32;
    }
    let mut info = Timebase { numer: 0, denom: 0 };
    // SAFETY: info is a valid writable system timebase structure.
    if unsafe { mach_timebase_info(&mut info) } != 0 || info.numer == 0 || info.denom == 0 {
        return None;
    }
    Some(info.numer as f64 / info.denom as f64)
}
impl QueryBackend for Metal {
    fn query_supported(device: &MetalDevice, kind: QueryKind) -> bool {
        #[cfg(target_vendor = "apple")]
        {
            kind == QueryKind::Occlusion
                || (kind == QueryKind::Timestamp && supported(device) && timebase().is_some())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, kind);
            false
        }
    }
    fn timestamp_result_format(_: &MetalDevice) -> api::TimestampResultFormat {
        #[cfg(target_vendor = "apple")]
        {
            api::TimestampResultFormat {
                invalid_value: Some(objc2_metal::MTLCounterErrorValue),
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            api::TimestampResultFormat::default()
        }
    }
    fn timestamp_clock(device: &MetalDevice, lane: api::LaneId) -> Option<api::TimestampClock> {
        #[cfg(target_vendor = "apple")]
        {
            if lane.ordinal > 1 || !supported(device) {
                return None;
            }
            Some(api::TimestampClock {
                period_ns: timebase()?,
                valid_bits: 64,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, lane);
            None
        }
    }
    fn create_query_set(device: &MetalDevice, desc: &QueryDesc) -> Result<MetalQuerySet> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{
                MTLCommonCounterSetTimestamp, MTLCounterSampleBufferDescriptor, MTLCounterSet as _,
                MTLDevice as _, MTLStorageMode,
            };
            if desc.kind == QueryKind::Occlusion {
                use objc2_metal::{MTLGPUFamily, MTLResourceOptions};
                let max_offset = if device._state._device.supportsFamily(MTLGPUFamily::Apple7)
                    || device._state._device.supportsFamily(MTLGPUFamily::Mac2)
                {
                    262144
                } else {
                    65528
                };
                if u64::from(desc.count.saturating_sub(1)) * 8 > max_offset {
                    return Err(Error::InvalidRange);
                }
                let length = (desc.count as usize)
                    .checked_mul(8)
                    .ok_or(Error::InvalidRange)?;
                let visibility = device
                    ._state
                    ._device
                    .newBufferWithLength_options(length, MTLResourceOptions::StorageModePrivate)
                    .ok_or(Error::OutOfMemory)?;
                return Ok(MetalQuerySet(Arc::new(QueryState {
                    raw: None,
                    visibility: Some(visibility),
                    statistics_pass: None,
                    submitted: Mutex::new(Submitted {
                        slots: vec![None; desc.count as usize],
                        generations: vec![None; desc.count as usize],
                    }),
                })));
            }
            if desc.kind != QueryKind::Timestamp || !supported(device) {
                return Err(Error::Unsupported("Metal timestamp queries"));
            }
            let sets = device
                ._state
                ._device
                .counterSets()
                .ok_or(Error::Unsupported("Metal timestamp counters"))?;
            // SAFETY: the framework's exported constant is valid for the process.
            let set = sets
                .iter()
                .find(|s| &*s.name() == unsafe { MTLCommonCounterSetTimestamp })
                .ok_or(Error::Unsupported("Metal timestamp counters"))?;
            let native_desc = MTLCounterSampleBufferDescriptor::new();
            native_desc.setCounterSet(Some(&set));
            native_desc.setStorageMode(MTLStorageMode::Private);
            // SAFETY: Core checks that count is nonzero; Metal validates device limits.
            unsafe { native_desc.setSampleCount(desc.count as usize) };
            let raw = device
                ._state
                ._device
                .newCounterSampleBufferWithDescriptor_error(&native_desc)
                .map_err(|_| Error::Unsupported("Metal counter sample buffer"))?;
            Ok(MetalQuerySet(Arc::new(QueryState {
                raw: Some(raw),
                visibility: None,
                statistics_pass: None,
                submitted: Mutex::new(Submitted {
                    slots: vec![None; desc.count as usize],
                    generations: vec![None; desc.count as usize],
                }),
            })))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, desc);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn reset_queries(
        e: &mut MetalCommandEncoder,
        set: &MetalQuerySet,
        slots: Range<u32>,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        if let Some(buffer) = &set.0.visibility {
            use objc2_metal::{
                MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _,
            };
            if e.render.is_some() || e.compute.is_some() {
                return Err(Error::InvalidState);
            }
            let blit = e.command.blitCommandEncoder().ok_or(Error::OutOfMemory)?;
            blit.fillBuffer_range_value(
                buffer,
                objc2_foundation::NSRange::new(
                    slots.start as usize * 8,
                    (slots.end - slots.start) as usize * 8,
                ),
                0,
            );
            blit.endEncoding();
        }
        e.query_ops.push(QueryOp::Reset(set.clone(), slots));
        Ok(())
    }
    fn write_timestamp(e: &mut MetalCommandEncoder, set: &MetalQuerySet, slot: u32) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{
                MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _,
            };
            if e.render.is_some() || e.compute.is_some() {
                return Err(Error::InvalidState);
            }
            let blit = e.command.blitCommandEncoder().ok_or(Error::OutOfMemory)?;
            // SAFETY: Core checked slot bounds and the command buffer retains encoded resources.
            unsafe {
                blit.sampleCountersInBuffer_atSampleIndex_withBarrier(
                    set.0.raw.as_deref().ok_or(Error::InvalidState)?,
                    slot as usize,
                    true,
                );
            }
            blit.endEncoding();
            e.query_ops.push(QueryOp::Write(set.clone(), slot));
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (e, set, slot);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn resolve_queries(
        e: &mut MetalCommandEncoder,
        set: &MetalQuerySet,
        slots: Range<u32>,
        dst: &MetalBuffer,
        offset: u64,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{
                MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _,
            };
            if e.render.is_some() || e.compute.is_some() {
                return Err(Error::InvalidState);
            }
            let bytes = u64::from(slots.end - slots.start) * 8;
            if !dst.usage.contains(api::BufferUsage::QUERY_RESOLVE)
                || offset % 256 != 0
                || offset.checked_add(bytes).is_none_or(|v| v > dst.size())
            {
                return Err(Error::InvalidRange);
            }
            let blit = e.command.blitCommandEncoder().ok_or(Error::OutOfMemory)?;
            if let Some(buffer) = &set.0.visibility {
                // SAFETY: Core validates the query range, and destination bounds were checked above.
                unsafe {
                    blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        buffer,
                        slots.start as usize * 8,
                        &dst.raw,
                        offset as usize,
                        bytes as usize,
                    );
                }
                blit.endEncoding();
                e.query_ops.push(QueryOp::Resolve(set.clone(), slots));
                return Ok(());
            }
            // SAFETY: Core checked slot range; destination usage, alignment and size were checked above.
            unsafe {
                blit.resolveCounters_inRange_destinationBuffer_destinationOffset(
                    set.0.raw.as_deref().ok_or(Error::InvalidState)?,
                    objc2_foundation::NSRange::new(
                        slots.start as usize,
                        (slots.end - slots.start) as usize,
                    ),
                    &dst.raw,
                    offset as usize,
                );
            }
            blit.endEncoding();
            e.query_ops.push(QueryOp::Resolve(set.clone(), slots));
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (e, set, slots, dst, offset);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn begin_occlusion(e: &mut MetalCommandEncoder, set: &MetalQuerySet, slot: u32) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLRenderCommandEncoder as _;
            if !e
                .occlusion_set
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(&current.0, &set.0))
                || e.occlusion_active
            {
                return Err(Error::InvalidState);
            }
            e.render
                .as_ref()
                .ok_or(Error::InvalidState)?
                .setVisibilityResultMode_offset(
                    objc2_metal::MTLVisibilityResultMode::Boolean,
                    slot as usize * 8,
                );
            e.occlusion_active = true;
            e.occlusion_slot = Some(slot);
            e.query_ops.push(QueryOp::Write(set.clone(), slot));
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (e, set, slot);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn end_occlusion(e: &mut MetalCommandEncoder) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLRenderCommandEncoder as _;
            if !e.occlusion_active {
                return Err(Error::InvalidState);
            }
            e.render
                .as_ref()
                .ok_or(Error::InvalidState)?
                .setVisibilityResultMode_offset(objc2_metal::MTLVisibilityResultMode::Disabled, 0);
            e.occlusion_active = false;
            e.occlusion_slot = None;
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = e;
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn begin_statistics(_: &mut MetalCommandEncoder, _: &MetalQuerySet, _: u32) -> Result<()> {
        Err(Error::Unsupported("Metal pipeline statistics"))
    }
    fn end_statistics(_: &mut MetalCommandEncoder) -> Result<()> {
        Err(Error::Unsupported("Metal pipeline statistics"))
    }
}

#[cfg(target_vendor = "apple")]
fn statistics_fields(device: &MetalDevice) -> api::StatisticsFields {
    use objc2_foundation::NSObjectProtocol as _;
    use objc2_metal::{MTLCounter as _, MTLCounterSet as _, MTLDevice as _};
    let native = &device._state._device;
    if !native.respondsToSelector(objc2::sel!(supportsCounterSampling:))
        || !native.respondsToSelector(objc2::sel!(counterSets))
        || !native.respondsToSelector(objc2::sel!(newCounterSampleBufferWithDescriptor:error:))
        || !native.supportsCounterSampling(objc2_metal::MTLCounterSamplingPoint::AtStageBoundary)
    {
        return api::StatisticsFields(0);
    }
    let Some(sets) = native.counterSets() else {
        return api::StatisticsFields(0);
    };
    // SAFETY: these constants are exported by the loaded Metal framework.
    let Some(set) = sets
        .iter()
        .find(|s| &*s.name() == unsafe { objc2_metal::MTLCommonCounterSetStatistic })
    else {
        return api::StatisticsFields(0);
    };
    let mut mask = 0;
    for counter in set.counters().iter() {
        let name = counter.name();
        // SAFETY: the named constants are process-lifetime Metal framework objects.
        let names = unsafe {
            [
                objc2_metal::MTLCommonCounterTessellationInputPatches,
                objc2_metal::MTLCommonCounterVertexInvocations,
                objc2_metal::MTLCommonCounterPostTessellationVertexInvocations,
                objc2_metal::MTLCommonCounterClipperInvocations,
                objc2_metal::MTLCommonCounterClipperPrimitivesOut,
                objc2_metal::MTLCommonCounterFragmentInvocations,
                objc2_metal::MTLCommonCounterFragmentsPassed,
                objc2_metal::MTLCommonCounterComputeKernelInvocations,
            ]
        };
        for (index, candidate) in names.iter().enumerate() {
            if &*name == *candidate {
                mask |= 1 << index;
            }
        }
    }
    api::StatisticsFields(mask)
}

impl fluxel_rhi_core::backend::PassStatisticsBackend for Metal {
    fn pass_statistics_fields(device: &MetalDevice) -> api::StatisticsFields {
        #[cfg(target_vendor = "apple")]
        {
            statistics_fields(device)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = device;
            api::StatisticsFields(0)
        }
    }
    fn pass_statistics_invalid_value(_: &MetalDevice) -> Option<u64> {
        #[cfg(target_vendor = "apple")]
        {
            Some(objc2_metal::MTLCounterErrorValue)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            None
        }
    }
    fn create_pass_statistics(
        device: &MetalDevice,
        pass: api::StatisticsPass,
        count: u32,
    ) -> Result<MetalQuerySet> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{
                MTLCounterSampleBufferDescriptor, MTLCounterSet as _, MTLDevice as _,
                MTLStorageMode,
            };
            if statistics_fields(device).0 == 0 {
                return Err(Error::Unsupported("Metal pass statistics"));
            }
            let samples_per_slot = match pass {
                api::StatisticsPass::Raster => 4,
                api::StatisticsPass::Compute => 2,
            };
            let samples = count
                .checked_mul(samples_per_slot)
                .filter(|n| *n != 0)
                .ok_or(Error::InvalidRange)?;
            let sets = device
                ._state
                ._device
                .counterSets()
                .ok_or(Error::Unsupported("Metal statistic counters"))?;
            // SAFETY: Metal exports this name for the process lifetime.
            let set = sets
                .iter()
                .find(|s| &*s.name() == unsafe { objc2_metal::MTLCommonCounterSetStatistic })
                .ok_or(Error::Unsupported("Metal statistic counters"))?;
            let desc = MTLCounterSampleBufferDescriptor::new();
            desc.setCounterSet(Some(&set));
            desc.setStorageMode(MTLStorageMode::Private);
            // SAFETY: checked arithmetic gives the requested nonzero native sample count.
            unsafe { desc.setSampleCount(samples as usize) };
            let raw = device
                ._state
                ._device
                .newCounterSampleBufferWithDescriptor_error(&desc)
                .map_err(|_| Error::Unsupported("Metal statistic sample buffer"))?;
            Ok(MetalQuerySet(Arc::new(QueryState {
                raw: Some(raw),
                visibility: None,
                statistics_pass: Some(pass),
                submitted: Mutex::new(Submitted {
                    slots: vec![None; count as usize],
                    generations: vec![None; count as usize],
                }),
            })))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, pass, count);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn prepare_pass_statistics(
        encoder: &mut MetalCommandEncoder,
        set: &MetalQuerySet,
        slot: u32,
        pass: api::StatisticsPass,
    ) -> Result<()> {
        if set.0.statistics_pass != Some(pass) || encoder.pending_statistics.is_some() {
            return Err(Error::InvalidState);
        }
        encoder.pending_statistics = Some((set.clone(), slot, pass));
        Ok(())
    }
    fn resolve_pass_statistics(
        e: &mut MetalCommandEncoder,
        set: &MetalQuerySet,
        slots: Range<u32>,
        dst: &MetalBuffer,
        offset: u64,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{
                MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _,
            };
            let pass = set.0.statistics_pass.ok_or(Error::InvalidState)?;
            let factor = match pass {
                api::StatisticsPass::Raster => 4u32,
                api::StatisticsPass::Compute => 2u32,
            };
            let bytes = u64::from(slots.end - slots.start)
                .checked_mul(pass.result_stride())
                .ok_or(Error::InvalidRange)?;
            if e.render.is_some()
                || e.compute.is_some()
                || !dst.usage.contains(api::BufferUsage::QUERY_RESOLVE)
                || offset % 256 != 0
                || offset.checked_add(bytes).is_none_or(|end| end > dst.size())
            {
                return Err(Error::InvalidRange);
            }
            let first = slots.start.checked_mul(factor).ok_or(Error::InvalidRange)?;
            let count = (slots.end - slots.start)
                .checked_mul(factor)
                .ok_or(Error::InvalidRange)?;
            let blit = e.command.blitCommandEncoder().ok_or(Error::OutOfMemory)?;
            // SAFETY: Core validated the logical range; allocation uses factor native
            // samples per slot, and destination bounds and alignment were checked.
            unsafe {
                blit.resolveCounters_inRange_destinationBuffer_destinationOffset(
                    set.0.raw.as_deref().ok_or(Error::InvalidState)?,
                    objc2_foundation::NSRange::new(first as usize, count as usize),
                    &dst.raw,
                    offset as usize,
                );
            }
            blit.endEncoding();
            e.query_ops.push(QueryOp::Resolve(set.clone(), slots));
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (e, set, slots, dst, offset);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
}

#[cfg(target_vendor = "apple")]
pub(super) fn configure_raster_statistics(
    e: &mut MetalCommandEncoder,
    desc: &objc2_metal::MTLRenderPassDescriptor,
) -> Result<()> {
    let Some((set, slot, pass)) = e.pending_statistics.take() else {
        return Ok(());
    };
    if pass != api::StatisticsPass::Raster {
        return Err(Error::InvalidState);
    }
    let first = slot.checked_mul(4).ok_or(Error::InvalidRange)? as usize;
    // SAFETY: attachment 0 exists; checked creation allocated four samples per slot.
    let attachment = unsafe { desc.sampleBufferAttachments().objectAtIndexedSubscript(0) };
    attachment.setSampleBuffer(set.0.raw.as_deref());
    // SAFETY: slot is validated by Core and four samples are allocated for it.
    unsafe {
        attachment.setStartOfVertexSampleIndex(first);
        attachment.setEndOfVertexSampleIndex(first + 1);
        attachment.setStartOfFragmentSampleIndex(first + 2);
        attachment.setEndOfFragmentSampleIndex(first + 3);
    }
    e.query_ops.push(QueryOp::Write(set, slot));
    Ok(())
}

#[cfg(target_vendor = "apple")]
pub(super) fn configure_compute_statistics(
    e: &mut MetalCommandEncoder,
    desc: &objc2_metal::MTLComputePassDescriptor,
) -> Result<()> {
    let Some((set, slot, pass)) = e.pending_statistics.take() else {
        return Ok(());
    };
    if pass != api::StatisticsPass::Compute {
        return Err(Error::InvalidState);
    }
    let first = slot.checked_mul(2).ok_or(Error::InvalidRange)? as usize;
    // SAFETY: attachment 0 exists; checked creation allocated two samples per slot.
    let attachment = unsafe { desc.sampleBufferAttachments().objectAtIndexedSubscript(0) };
    attachment.setSampleBuffer(set.0.raw.as_deref());
    // SAFETY: slot is validated by Core and two samples are allocated for it.
    unsafe {
        attachment.setStartOfEncoderSampleIndex(first);
        attachment.setEndOfEncoderSampleIndex(first + 1);
    }
    e.query_ops.push(QueryOp::Write(set, slot));
    Ok(())
}

#[cfg(all(test, not(target_vendor = "apple")))]
#[path = "../tests/backend/query.rs"]
mod tests;
