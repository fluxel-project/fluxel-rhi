//! Device-local acceleration compaction and completion evidence.

use super::*;
use crate::backend::AccelerationCompactionBackend;

/// A compaction size observation tied to one recorded build.
pub struct AccelerationCompactionTicket<B: AccelerationCompactionBackend> {
    device: DeviceId,
    native: B::NativeCompactionTicket,
}

impl<B: AccelerationCompactionBackend> RayDomain<'_, B> {
    /// Records a build and a private readback of its compacted size. The
    /// caller allocates source and scratch using the compaction-enabled bounds.
    pub fn build_for_compaction(
        &self,
        encoder: &mut CommandEncoder<B>,
        source: &AccelerationStructure<B>,
        scratch: &Buffer<B>,
        desc: &AccelerationDesc<B>,
    ) -> Result<AccelerationCompactionTicket<B>> {
        self.record_for_compaction(encoder, source, scratch, desc, false)
    }

    /// Refits an update-enabled, compaction-enabled build and measures the
    /// new revision. The ordering and geometry requirements of `update` apply.
    pub fn update_for_compaction(
        &self,
        encoder: &mut CommandEncoder<B>,
        source: &AccelerationStructure<B>,
        scratch: &Buffer<B>,
        desc: &AccelerationDesc<B>,
    ) -> Result<AccelerationCompactionTicket<B>> {
        if !self.device.caps.acceleration_structure_update {
            return Err(Error::Unsupported("acceleration structure update"));
        }
        if !desc.allow_update {
            return Err(Error::InvalidState);
        }
        self.record_for_compaction(encoder, source, scratch, desc, true)
    }

    fn record_for_compaction(
        &self,
        encoder: &mut CommandEncoder<B>,
        source: &AccelerationStructure<B>,
        scratch: &Buffer<B>,
        desc: &AccelerationDesc<B>,
        refit: bool,
    ) -> Result<AccelerationCompactionTicket<B>> {
        self.require_compaction()?;
        if encoder.device != self.device.id
            || source.device != self.device.id
            || scratch.device != self.device.id
        {
            return Err(Error::WrongDevice);
        }
        self.validate_acceleration_input(&desc.input)?;
        let native = B::record_compactable_build(
            encoder.ready()?,
            &source.native,
            &scratch.native,
            desc,
            refit,
        );
        if native.is_err() {
            encoder.poisoned = true;
        }
        Ok(AccelerationCompactionTicket {
            device: self.device.id,
            native: native?,
        })
    }

    /// Observes the tracked size once its native submission fence completes.
    /// An unsubmitted or pending ticket returns `None`.
    pub fn compacted_size(&self, ticket: &AccelerationCompactionTicket<B>) -> Result<Option<u64>> {
        self.require_compaction()?;
        if ticket.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        B::compacted_size(&ticket.native)
    }

    /// Allocates the exact measured size and records a compact copy. The
    /// returned structure is usable after the copy is ordered and visible.
    /// Pending tickets and tickets invalidated by a later build are rejected.
    pub fn compact(
        &self,
        encoder: &mut CommandEncoder<B>,
        ticket: &AccelerationCompactionTicket<B>,
    ) -> Result<AccelerationStructure<B>> {
        self.require_compaction()?;
        if ticket.device != self.device.id || encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let native = B::compact_acceleration(encoder.ready()?, &ticket.native)?;
        // SAFETY: the backend created this structure for the validated encoder device.
        Ok(unsafe { AccelerationStructure::from_native(self.device.id, native) })
    }

    fn require_compaction(&self) -> Result<()> {
        if !self.device.caps.acceleration_compaction {
            return Err(Error::Unsupported("acceleration compaction"));
        }
        self.require_compaction_size()
    }
}
