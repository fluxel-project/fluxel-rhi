//! Query allocation, command recording, and result resolution contract.
//!
//! Query availability is established by explicit completion evidence; Core does
//! not poll a driver or choose when a renderer consumes results.
#![allow(missing_docs, dead_code, unused_variables, unused_imports)]
use super::*;
use std::{
    future::Future,
    marker::PhantomData,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

/// Identifies the measurement recorded by a query set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryKind {
    /// Tests whether rasterized samples passed the fragment tests.
    ///
    /// Resolved values are unsigned integers. Zero means no samples passed;
    /// any nonzero value means at least one sample passed. The exact nonzero
    /// value is backend-defined: a backend may expose its sample count or the
    /// native visibility predicate when only a boolean query is available.
    Occlusion,
    Timestamp,
    PipelineStatistics,
}
/// Producer ordering required before resolving query results.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QueryResolveMode {
    /// Resolve may follow the producer in GPU submission order without a host wait.
    #[default]
    Ordered,
    /// Establish producer completion first. The backend checks native result
    /// availability before any destination write. Pending results cannot be
    /// resolved by the GPU or synchronously forced ready on this route.
    AfterCompletion,
}

impl QueryKind {
    /// Size in bytes of each resolved slot. Results use native-endian u64 values.
    pub const fn result_stride(self) -> u64 {
        match self {
            Self::Occlusion | Self::Timestamp => 8,
            Self::PipelineStatistics => 88,
        }
    }
}
/// Pipeline counters in the order written to a resolved slot.
/// Counters for stages absent from a pipeline are zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineStatistics {
    pub input_assembler_vertices: u64,
    pub input_assembler_primitives: u64,
    pub vertex_shader_invocations: u64,
    pub geometry_shader_invocations: u64,
    pub geometry_shader_primitives: u64,
    pub clipping_invocations: u64,
    pub clipping_primitives: u64,
    pub fragment_shader_invocations: u64,
    pub hull_shader_invocations: u64,
    pub domain_shader_invocations: u64,
    pub compute_shader_invocations: u64,
}
/// A nonempty set of slots that all record the same measurement.
pub struct QueryDesc {
    pub kind: QueryKind,
    pub count: u32,
}
/// Queue timestamp scale and width. `period_ns` is an f64 approximation of
/// nanoseconds per tick; use differences from the same lane. Ticks wrap modulo
/// 2^valid_bits, so elapsed intervals must be shorter than one wrap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimestampClock {
    pub period_ns: f64,
    pub valid_bits: u32,
}
/// Backend-defined interpretation of a resolved timestamp slot.
///
/// Consumers must check the reserved value before performing timestamp
/// arithmetic, including when the resolved buffer is consumed on the GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimestampResultFormat {
    /// A reserved value reporting a failed sample, if the backend uses one.
    pub invalid_value: Option<u64>,
}
/// A resolved timestamp sample could not be collected by the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidTimestampSample;

/// A pass type whose stage boundaries can be sampled for workload statistics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatisticsPass {
    Raster,
    Compute,
}

/// Availability bits for the fields of [`StatisticsSample`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StatisticsFields(pub u32);
impl StatisticsFields {
    pub const TESSELLATION_INPUT_PATCHES: Self = Self(1 << 0);
    pub const VERTEX_INVOCATIONS: Self = Self(1 << 1);
    pub const POST_TESSELLATION_VERTICES: Self = Self(1 << 2);
    pub const CLIPPER_INVOCATIONS: Self = Self(1 << 3);
    pub const CLIPPER_PRIMITIVES_OUT: Self = Self(1 << 4);
    pub const FRAGMENT_INVOCATIONS: Self = Self(1 << 5);
    pub const FRAGMENTS_PASSED: Self = Self(1 << 6);
    pub const COMPUTE_INVOCATIONS: Self = Self(1 << 7);
    pub const fn contains(self, field: Self) -> bool {
        self.0 & field.0 == field.0
    }
}

/// One native-endian workload counter sample. A field is meaningful only if
/// its bit is set in the device's availability mask and its value is not the
/// reported invalid sentinel. Values have the meaning of the backend's
/// workload counters at the sampled boundary; there is no guarantee that
/// samples may be subtracted, including samples of the same pass. Consult the
/// backend's documented counter semantics before interpreting them as totals.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StatisticsSample {
    pub tessellation_input_patches: u64,
    pub vertex_invocations: u64,
    pub post_tessellation_vertices: u64,
    pub clipper_invocations: u64,
    pub clipper_primitives_out: u64,
    pub fragment_invocations: u64,
    pub fragments_passed: u64,
    pub compute_invocations: u64,
}

impl StatisticsPass {
    /// Raster slots contain vertex start/end then fragment start/end; compute
    /// slots contain compute start/end. Each sample is 64 bytes.
    pub const fn result_stride(self) -> u64 {
        match self {
            Self::Raster => 256,
            Self::Compute => 128,
        }
    }
}

/// A set of whole-pass counter samples belonging to one device.
pub struct PassStatisticsSet<B: PassStatisticsBackend> {
    device: DeviceId,
    native: B::NativeQuerySet,
    pass: StatisticsPass,
    count: u32,
}
impl<B: PassStatisticsBackend> PassStatisticsSet<B> {
    /// The pass type selected at creation.
    pub fn pass(&self) -> StatisticsPass {
        self.pass
    }
    /// Number of logical pass slots in this set.
    pub fn count(&self) -> u32 {
        self.count
    }
}

/// Records and resolves whole-pass stage counter samples. A slot must be reset
/// before use, written by a finished pass, and resolved after that pass in
/// submission order. Completion evidence is required before reading results.
pub struct PassStatistics<'a, B: PassStatisticsBackend> {
    device: &'a Device<B>,
}
impl<B: PassStatisticsBackend> Device<B> {
    pub fn pass_statistics(&self) -> Option<PassStatistics<'_, B>> {
        (B::pass_statistics_fields(&self.native).0 != 0).then_some(PassStatistics { device: self })
    }
}
impl<B: PassStatisticsBackend> PassStatistics<'_, B> {
    /// Fields that this device can sample. Other fields have no defined value.
    pub fn fields(&self) -> StatisticsFields {
        B::pass_statistics_fields(&self.device.native)
    }
    /// Reserved value that indicates a failed field sample, if any.
    pub fn invalid_value(&self) -> Option<u64> {
        B::pass_statistics_invalid_value(&self.device.native)
    }
    /// Allocates a nonempty set for one pass type.
    pub fn create(&self, pass: StatisticsPass, count: u32) -> Result<PassStatisticsSet<B>> {
        if count == 0 {
            return Err(Error::InvalidRange);
        }
        let native = B::create_pass_statistics(&self.device.native, pass, count)?;
        Ok(PassStatisticsSet {
            device: self.device.id,
            native,
            pass,
            count,
        })
    }
    /// Resets a nonempty slot range. Submission ordering applies to resets.
    pub fn reset(
        &self,
        encoder: &mut CommandEncoder<B>,
        set: &PassStatisticsSet<B>,
        slots: Range<u32>,
    ) -> Result<()> {
        self.validate(encoder, set, &slots)?;
        encoder.call(|e| B::reset_queries(e, &set.native, slots))
    }
    fn validate(
        &self,
        encoder: &CommandEncoder<B>,
        set: &PassStatisticsSet<B>,
        slots: &Range<u32>,
    ) -> Result<()> {
        if encoder.device != self.device.id || set.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        if slots.start >= slots.end || slots.end > set.count {
            return Err(Error::InvalidRange);
        }
        Ok(())
    }
    /// Begins a raster pass that samples its vertex and fragment boundaries.
    pub fn begin_raster<'a>(
        &self,
        encoder: &'a mut CommandEncoder<B>,
        set: &PassStatisticsSet<B>,
        slot: u32,
        desc: &RasterPassDesc<'_, B>,
    ) -> Result<RasterPass<'a, B>>
    where
        B: RasterBackend,
    {
        self.validate(encoder, set, &(slot..slot.saturating_add(1)))?;
        if set.pass != StatisticsPass::Raster {
            return Err(Error::InvalidState);
        }
        encoder.begin_raster_prepared(desc, |e| {
            B::prepare_pass_statistics(e, &set.native, slot, set.pass)
        })
    }
    /// Begins a compute pass that samples its start and end boundaries.
    pub fn begin_compute<'a>(
        &self,
        encoder: &'a mut CommandEncoder<B>,
        set: &PassStatisticsSet<B>,
        slot: u32,
    ) -> Result<ComputePass<'a, B>> {
        self.validate(encoder, set, &(slot..slot.saturating_add(1)))?;
        if set.pass != StatisticsPass::Compute {
            return Err(Error::InvalidState);
        }
        let domain = self.device.compute().ok_or(Error::Unsupported("compute"))?;
        domain.begin_prepared(encoder, |e| {
            B::prepare_pass_statistics(e, &set.native, slot, set.pass)
        })
    }
    /// Resolves slots into a QUERY_RESOLVE buffer using the corresponding
    /// [`StatisticsPass::result_stride`]. Backends validate offset alignment,
    /// buffer capacity, and pass state.
    pub fn resolve(
        &self,
        encoder: &mut CommandEncoder<B>,
        set: &PassStatisticsSet<B>,
        slots: Range<u32>,
        destination: &Buffer<B>,
        offset: u64,
    ) -> Result<()> {
        self.validate(encoder, set, &slots)?;
        if destination.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        encoder.call(|e| {
            B::resolve_pass_statistics(e, &set.native, slots, &destination.native, offset)
        })
    }
}
impl TimestampResultFormat {
    /// Checks a native-endian resolved slot and returns its tick value.
    pub fn decode(self, value: u64) -> std::result::Result<u64, InvalidTimestampSample> {
        if self.invalid_value == Some(value) {
            Err(InvalidTimestampSample)
        } else {
            Ok(value)
        }
    }
}
pub struct QuerySet<B: Backend> {
    pub(crate) device: DeviceId,
    pub(crate) native: B::NativeQuerySet,
    kind: QueryKind,
    count: u32,
}
impl<B: Backend> QuerySet<B> {
    /// The measurement recorded by this set.
    pub fn kind(&self) -> QueryKind {
        self.kind
    }
    /// The number of query slots in this set.
    pub fn count(&self) -> u32 {
        self.count
    }

    fn validate_slot(&self, slot: u32) -> Result<()> {
        if slot >= self.count {
            return Err(Error::InvalidRange);
        }
        Ok(())
    }
    fn validate_range(&self, slots: &Range<u32>) -> Result<()> {
        if slots.start >= slots.end || slots.end > self.count {
            return Err(Error::InvalidRange);
        }
        Ok(())
    }
    /// Backend bridge for native pass descriptors that must declare a query
    /// set before any individual query slot is begun.
    ///
    /// # Safety
    /// Call only from `B`'s implementation and retain the owning `QuerySet`
    /// for every native use.
    #[doc(hidden)]
    pub unsafe fn native_ref(&self) -> &B::NativeQuerySet {
        &self.native
    }
}
/// Operations for the query kind selected by [`Device::queries`].
///
/// Creation and operations reject a different kind before invoking the backend.
/// Query ranges must be nonempty and within the set; individual slots must be
/// less than its count. Backends validate native alignment and pass state.
pub struct Queries<'a, B: QueryBackend> {
    device: &'a Device<B>,
    kind: QueryKind,
}
impl<B: QueryBackend> Device<B> {
    /// Opens a domain for a supported query kind on this device.
    pub fn queries(&self, kind: QueryKind) -> Option<Queries<'_, B>> {
        B::query_supported(&self.native, kind).then_some(Queries { device: self, kind })
    }
}
impl<B: QueryBackend> Queries<'_, B> {
    /// Reports whether resolve can follow a producer in submission order or
    /// requires its completion first. Core never waits, polls, or defers query
    /// commands on the caller's behalf.
    pub fn resolve_mode(&self) -> QueryResolveMode {
        B::query_resolve_mode(&self.device.native, self.kind)
    }
    /// Result format for this timestamp domain; other query kinds return `None`.
    pub fn timestamp_result_format(&self) -> Option<TimestampResultFormat> {
        (self.kind == QueryKind::Timestamp).then(|| B::timestamp_result_format(&self.device.native))
    }
    /// Timestamp ticks per second on the specified queue. Only tick
    /// differences on the same queue are meaningful; the epoch is unspecified.
    pub fn timestamp_frequency(&self, lane: LaneId) -> Option<u64> {
        (self.kind == QueryKind::Timestamp
            && lane.device == self.device.id
            && self.device.lanes().iter().any(|info| info.id == lane))
        .then(|| B::timestamp_frequency(&self.device.native, lane))
        .flatten()
        .filter(|frequency| *frequency != 0)
    }
    /// Timestamp scale and counter width for the specified lane.
    pub fn timestamp_clock(&self, lane: LaneId) -> Option<TimestampClock> {
        (self.kind == QueryKind::Timestamp
            && lane.device == self.device.id
            && self.device.lanes().iter().any(|info| info.id == lane))
        .then(|| B::timestamp_clock(&self.device.native, lane))
        .flatten()
        .filter(|clock| {
            clock.period_ns.is_finite()
                && clock.period_ns > 0.0
                && (1..=64).contains(&clock.valid_bits)
        })
    }
    fn require_kind(&self, kind: QueryKind) -> Result<()> {
        if self.kind != kind {
            return Err(Error::InvalidState);
        }
        Ok(())
    }
    fn validate_set(&self, set: &QuerySet<B>) -> Result<()> {
        if set.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        self.require_kind(set.kind)
    }
    /// Resets a nonempty slot range of a set created for this domain kind.
    /// The encoder and set must belong to this device.
    pub fn reset(
        &self,
        encoder: &mut CommandEncoder<B>,
        set: &QuerySet<B>,
        slots: Range<u32>,
    ) -> Result<()> {
        if encoder.device != self.device.id || set.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        self.validate_set(set)?;
        set.validate_range(&slots)?;
        encoder.call(|e| B::reset_queries(e, &set.native, slots))
    }
    /// Creates a nonempty set whose kind must match this domain.
    pub fn create(&self, desc: &QueryDesc) -> Result<QuerySet<B>> {
        self.require_kind(desc.kind)?;
        if desc.count == 0 {
            return Err(Error::InvalidRange);
        }
        Ok(QuerySet {
            device: self.device.id,
            native: B::create_query_set(&self.device.native, desc)?,
            kind: desc.kind,
            count: desc.count,
        })
    }
    pub fn timestamp(
        &self,
        encoder: &mut CommandEncoder<B>,
        set: &QuerySet<B>,
        slot: u32,
    ) -> Result<()> {
        if encoder.device != self.device.id || set.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        self.require_kind(QueryKind::Timestamp)?;
        self.validate_set(set)?;
        set.validate_slot(slot)?;
        encoder.call(|e| B::write_timestamp(e, &set.native, slot))
    }
    pub fn begin_occlusion(
        &self,
        pass: &mut RasterPass<'_, B>,
        set: &QuerySet<B>,
        slot: u32,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if pass.encoder.device != self.device.id || set.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        self.require_kind(QueryKind::Occlusion)?;
        self.validate_set(set)?;
        set.validate_slot(slot)?;
        pass.call(|e| B::begin_occlusion(e, &set.native, slot))
    }
    pub fn end_occlusion(&self, pass: &mut RasterPass<'_, B>) -> Result<()>
    where
        B: RasterBackend,
    {
        if pass.encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        self.require_kind(QueryKind::Occlusion)?;
        pass.call(B::end_occlusion)
    }
    pub fn begin_statistics(
        &self,
        encoder: &mut CommandEncoder<B>,
        set: &QuerySet<B>,
        slot: u32,
    ) -> Result<()> {
        if encoder.device != self.device.id || set.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        self.require_kind(QueryKind::PipelineStatistics)?;
        self.validate_set(set)?;
        set.validate_slot(slot)?;
        encoder.call(|e| B::begin_statistics(e, &set.native, slot))
    }
    pub fn end_statistics(&self, encoder: &mut CommandEncoder<B>) -> Result<()> {
        if encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        self.require_kind(QueryKind::PipelineStatistics)?;
        encoder.call(B::end_statistics)
    }
    /// Resolves a nonempty slot range of this domain kind into a device-local buffer.
    /// Each resolved slot must have been written in submission order after its
    /// most recent reset. Backends may validate this when the work is submitted.
    /// Buffer usage, alignment and pass state are checked by the backend.
    /// When [`Self::resolve_mode`] is [`QueryResolveMode::AfterCompletion`],
    /// establish producer completion before this call. A result that is still
    /// unavailable returns [`Error::InvalidState`] before destination writes;
    /// no pending resolution is retained or replayed later.
    pub fn resolve(
        &self,
        encoder: &mut CommandEncoder<B>,
        set: &QuerySet<B>,
        slots: Range<u32>,
        dst: &Buffer<B>,
        offset: u64,
    ) -> Result<()> {
        if encoder.device != self.device.id
            || set.device != self.device.id
            || dst.device != self.device.id
        {
            return Err(Error::WrongDevice);
        }
        self.validate_set(set)?;
        set.validate_range(&slots)?;
        encoder.call(|e| B::resolve_queries(e, &set.native, slots, &dst.native, offset))
    }
}
