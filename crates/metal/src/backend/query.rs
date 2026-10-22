//! Metal timestamp sampling and submission-ordered slot state.
//!
//! Only Apple-family GPUs with blit-boundary timestamp counters can satisfy
//! the fixed clock contract; Intel/AMD GPU clocks require dynamic calibration.
//! Pipeline statistics remain unsupported because Metal's counters do not
//! define Core's eleven fields. The supported subset is determined by runtime
//! feature and counter-set checks; it has not been validated on an Apple GPU.
use super::{Metal, MetalBuffer, MetalCommandEncoder, MetalDevice};
use fluxel_rhi_core::api::{self, Error, QueryBackend, QueryDesc, QueryKind, Result};
use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

#[derive(Clone)]
pub struct MetalQuerySet(Arc<QueryState>);
struct QueryState {
    #[cfg(target_vendor = "apple")]
    raw: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCounterSampleBuffer>,
    >,
    submitted: Mutex<Submitted>,
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
        if slots.iter().any(|s| s.is_some_and(|(l, _)| l != lane)) {
            return Err(Error::Unsupported("Metal query set across lanes"));
        }
        match self {
            Self::Reset(_, r) => {
                for i in r.clone() {
                    slots[i as usize] = Some((lane, false));
                }
            }
            Self::Write(_, i) => {
                if slots[*i as usize] != Some((lane, false)) {
                    return Err(Error::InvalidState);
                }
                slots[*i as usize] = Some((lane, true));
            }
            Self::Resolve(_, r) => {
                if r.clone().any(|i| slots[i as usize] != Some((lane, true))) {
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
            kind == QueryKind::Timestamp && supported(device) && timebase().is_some()
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
                raw,
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
                    &set.0.raw,
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
            // SAFETY: Core checked slot range; destination usage, alignment and size were checked above.
            unsafe {
                blit.resolveCounters_inRange_destinationBuffer_destinationOffset(
                    &set.0.raw,
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
    fn begin_occlusion(_: &mut MetalCommandEncoder, _: &MetalQuerySet, _: u32) -> Result<()> {
        Err(Error::Unsupported("Metal occlusion queries"))
    }
    fn end_occlusion(_: &mut MetalCommandEncoder) -> Result<()> {
        Err(Error::Unsupported("Metal occlusion queries"))
    }
    fn begin_statistics(_: &mut MetalCommandEncoder, _: &MetalQuerySet, _: u32) -> Result<()> {
        Err(Error::Unsupported("Metal pipeline statistics"))
    }
    fn end_statistics(_: &mut MetalCommandEncoder) -> Result<()> {
        Err(Error::Unsupported("Metal pipeline statistics"))
    }
}

#[cfg(all(test, not(target_vendor = "apple")))]
mod tests {
    use super::*;
    #[test]
    fn slot_state_follows_submission_order_and_lane() {
        let set = MetalQuerySet(Arc::new(QueryState {
            submitted: Mutex::new(Submitted {
                slots: vec![None; 2],
                generations: vec![None; 2],
            }),
        }));
        let reset = QueryOp::Reset(set.clone(), 0..2);
        let write = QueryOp::Write(set.clone(), 0);
        let resolve = QueryOp::Resolve(set.clone(), 0..1);
        assert!(matches!(
            validate_ops([(0, &resolve)].into_iter()),
            Err(Error::InvalidState)
        ));
        assert!(matches!(
            validate_ops([(0, &write)].into_iter()),
            Err(Error::InvalidState)
        ));
        validate_ops([(0, &reset), (0, &write), (0, &resolve)].into_iter()).unwrap();
        let first = commit_ops(&[reset.clone(), write.clone()], 0);
        assert!(validate_ops([(0, &resolve)].into_iter()).is_ok());
        assert!(matches!(
            validate_ops([(1, &resolve)].into_iter()),
            Err(Error::Unsupported(_))
        ));
        invalidate_ops(&first, 0);
        assert!(matches!(
            validate_ops([(0, &resolve)].into_iter()),
            Err(Error::InvalidState)
        ));
        let older = commit_ops(&[reset.clone(), write.clone()], 0);
        let newer = commit_ops(&[reset, write], 0);
        invalidate_ops(&older, 0);
        assert!(validate_ops([(0, &resolve)].into_iter()).is_ok());
        invalidate_ops(&newer, 0);
        assert!(matches!(
            validate_ops([(0, &resolve)].into_iter()),
            Err(Error::InvalidState)
        ));
    }
}
