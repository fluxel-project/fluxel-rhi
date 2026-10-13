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

#[derive(Clone, Copy, Debug)]
pub enum QueryKind {
    Occlusion,
    Timestamp,
    PipelineStatistics,
}
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
}
impl<B: Backend> QuerySet<B> {
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
pub struct Queries<'a, B: QueryBackend> {
    device: &'a Device<B>,
}
impl<B: QueryBackend> Device<B> {
    pub fn queries(&self, kind: QueryKind) -> Option<Queries<'_, B>> {
        B::query_supported(&self.native, kind).then_some(Queries { device: self })
    }
}
impl<B: QueryBackend> Queries<'_, B> {
    pub fn reset(
        &self,
        encoder: &mut CommandEncoder<B>,
        set: &QuerySet<B>,
        slots: Range<u32>,
    ) -> Result<()> {
        if encoder.device != self.device.id || set.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        encoder.call(|e| B::reset_queries(e, &set.native, slots))
    }
    pub fn create(&self, desc: &QueryDesc) -> Result<QuerySet<B>> {
        Ok(QuerySet {
            device: self.device.id,
            native: B::create_query_set(&self.device.native, desc)?,
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
        pass.call(|e| B::begin_occlusion(e, &set.native, slot))
    }
    pub fn end_occlusion(&self, pass: &mut RasterPass<'_, B>) -> Result<()>
    where
        B: RasterBackend,
    {
        if pass.encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
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
        encoder.call(|e| B::begin_statistics(e, &set.native, slot))
    }
    pub fn end_statistics(&self, encoder: &mut CommandEncoder<B>) -> Result<()> {
        if encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        encoder.call(B::end_statistics)
    }
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
        encoder.call(|e| B::resolve_queries(e, &set.native, slots, &dst.native, offset))
    }
}
