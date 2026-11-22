//! Immutable device-local index storage for ordered BLAS build and refit.

use super::*;

/// Device-local immutable index storage initialized by one recorded GPU copy.
pub struct AccelerationIndexSnapshot<B: Backend> {
    pub(super) device: DeviceId,
    native: <B::Ray as RayOps<B>>::NativeIndexSnapshot,
    count: u32,
    format: IndexFormat,
}

impl<B: Backend> Clone for AccelerationIndexSnapshot<B> {
    fn clone(&self) -> Self {
        Self {
            device: self.device,
            native: self.native.clone(),
            count: self.count,
            format: self.format,
        }
    }
}

impl<B: Backend> AccelerationIndexSnapshot<B> {
    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn format(&self) -> IndexFormat {
        self.format
    }

    /// Bytes copied and stored by the snapshot, excluding allocation padding.
    pub fn byte_len(&self) -> u64 {
        u64::from(self.count)
            * match self.format {
                IndexFormat::Uint16 => 2,
                IndexFormat::Uint32 => 4,
            }
    }

    /// # Safety
    /// Only the backend may read this storage during acceleration builds.
    /// It must not expose it as a mutable buffer or write it again.
    #[doc(hidden)]
    pub unsafe fn native_ref(&self) -> &<B::Ray as RayOps<B>>::NativeIndexSnapshot {
        &self.native
    }
}
impl<B: Backend> RayDomain<'_, B> {
    /// Reports whether indexed BLAS refit can use immutable index snapshots.
    pub fn supports_index_snapshots(&self) -> bool {
        self.device.caps.acceleration_structure_update && B::Ray::INDEX_SNAPSHOTS_AVAILABLE
    }

    /// Records one copy of the selected index bytes into a private GPU buffer.
    /// The caller transitions the source to `COPY_READ` and orders this copy
    /// before the first build. The backend handles the new buffer's transitions
    /// from copy destination to acceleration input. Build and refit use the
    /// same snapshot. The source buffer can be changed after the copy completes.
    pub fn snapshot_indices(
        &self,
        encoder: &mut CommandEncoder<B>,
        source: &AccelerationIndices<B>,
    ) -> Result<AccelerationIndexSnapshot<B>>
    where
        B: CommandBackend,
    {
        if !self.supports_index_snapshots() {
            return Err(Error::Unsupported("acceleration index snapshots"));
        }
        if encoder.device != self.device.id || source.buffer.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let range = source.byte_range().ok_or(Error::InvalidRange)?;
        let native = B::Ray::snapshot_indices(encoder.ready()?, &source.buffer.native, range)
            .inspect_err(|_| encoder.poisoned = true)?;
        Ok(AccelerationIndexSnapshot {
            device: self.device.id,
            native,
            count: source.count,
            format: source.format,
        })
    }
}
