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
/// A nonempty set of slots that all record the same measurement.
pub struct QueryDesc {
    pub kind: QueryKind,
    pub count: u32,
}
pub trait QueryBackend: CommandBackend {
    fn query_supported(device: &Self::NativeDevice, kind: QueryKind) -> bool;
    fn create_query_set(
        device: &Self::NativeDevice,
        desc: &QueryDesc,
    ) -> Result<Self::NativeQuerySet>;
    fn write_timestamp(
        encoder: &mut Self::NativeEncoder,
        set: &Self::NativeQuerySet,
        slot: u32,
    ) -> Result<()>;
    fn begin_occlusion(
        encoder: &mut Self::NativeEncoder,
        set: &Self::NativeQuerySet,
        slot: u32,
    ) -> Result<()>;
    fn reset_queries(
        encoder: &mut Self::NativeEncoder,
        set: &Self::NativeQuerySet,
        slots: Range<u32>,
    ) -> Result<()>;
    fn end_occlusion(encoder: &mut Self::NativeEncoder) -> Result<()>;
    fn begin_statistics(
        encoder: &mut Self::NativeEncoder,
        set: &Self::NativeQuerySet,
        slot: u32,
    ) -> Result<()>;
    fn end_statistics(encoder: &mut Self::NativeEncoder) -> Result<()>;
    fn resolve_queries(
        encoder: &mut Self::NativeEncoder,
        set: &Self::NativeQuerySet,
        slots: Range<u32>,
        destination: &Self::NativeBuffer,
        offset: u64,
    ) -> Result<()>;
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
    /// Buffer usage, alignment, pass state, and native result layout are checked by the backend.
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
