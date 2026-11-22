//! Explicit GPU instance packing for top-level acceleration structures.

use super::*;

/// Size in bytes of one logical GPU instance record. The sixteen little-endian
/// words are three row-major `float4` transform rows, custom index and mask
/// (low 24 and high 8 bits), shader table offset and instance flags (low 24
/// and high 8 bits), a zero-based BLAS table index, and a reserved zero word.
/// Instance flag bits are those of [`AccelerationInstanceFlags`]. Records
/// with an invalid table index, flags, reserved word, or nonfinite transform
/// are excluded from traversal.
pub const GPU_TLAS_INSTANCE_BYTES: u64 = 64;

/// Source records and the typed table they may index. Table order defines the
/// GPU indices. Every entry is a previously built or explicitly ordered BLAS.
pub struct GpuTlasPackDesc<'a, B: Backend> {
    pub records: &'a Buffer<B>,
    pub offset: u64,
    pub count: u32,
    pub blas: &'a [AccelerationStructure<B>],
}

/// Native recording and temporary storage cost of one explicit pack.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuTlasPackCost {
    /// Bytes read from the logical records and bytes written to private packed storage.
    pub packed_bytes: u64,
    /// Bytes uploaded for native BLAS addresses, excluding upload alignment padding.
    pub address_upload_bytes: u64,
    /// Dispatch dimensions for one compute command.
    pub workgroups: [u32; 3],
}

/// Layout for querying or allocating a GPU-produced TLAS before recording.
#[derive(Clone, Copy, Debug)]
pub struct GpuTlasLayout {
    pub count: u32,
    pub fast_trace: bool,
    pub allow_update: bool,
}

/// Private packed records produced by an explicit GPU command.
pub struct PackedTlasInstances<B: Backend> {
    pub(super) device: DeviceId,
    count: u32,
    native: <B::Ray as RayOps<B>>::NativePackedTlas,
}

impl<B: Backend> Clone for PackedTlasInstances<B> {
    fn clone(&self) -> Self {
        Self {
            device: self.device,
            count: self.count,
            native: self.native.clone(),
        }
    }
}

impl<B: Backend> PackedTlasInstances<B> {
    pub fn count(&self) -> u32 {
        self.count
    }

    /// # Safety
    /// The backend may borrow the private instances only for same-device TLAS builds.
    #[doc(hidden)]
    pub unsafe fn native_ref(&self) -> &<B::Ray as RayOps<B>>::NativePackedTlas {
        &self.native
    }
}

pub(super) fn validate_tlas_count(count: u32, limit: u32) -> Result<()> {
    if count == 0 || (limit != 0 && count > limit) {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

impl<B: Backend> RayDomain<'_, B> {
    /// Reports whether this device can pack GPU-produced TLAS records.
    pub fn supports_gpu_tlas(&self) -> bool {
        self.device.caps.acceleration_structure && B::Ray::GPU_TLAS_AVAILABLE
    }

    fn require_gpu_tlas(&self, count: u32) -> Result<()> {
        if !self.supports_gpu_tlas() {
            return Err(Error::Unsupported("GPU-produced TLAS"));
        }
        validate_tlas_count(count, self.device.caps.limits.max_tlas_instance_count)
    }

    fn validate_gpu_tlas_layout(&self, layout: GpuTlasLayout) -> Result<()> {
        self.require_gpu_tlas(layout.count)?;
        if layout.allow_update && !self.device.caps.acceleration_structure_update {
            return Err(Error::Unsupported("acceleration structure update"));
        }
        Ok(())
    }

    /// Reports the cost before recording. The offsets of the packed buffer and
    /// native addresses remain private to the backend.
    pub fn gpu_tlas_pack_cost(&self, count: u32, blas_count: u32) -> Result<GpuTlasPackCost> {
        self.require_gpu_tlas(count)?;
        B::Ray::gpu_tlas_pack_cost(&self.device.native, count, blas_count)
    }

    /// Queries native storage and scratch for the known instance count.
    pub fn gpu_tlas_sizes(&self, layout: GpuTlasLayout) -> Result<AccelerationSizes> {
        self.validate_gpu_tlas_layout(layout)?;
        B::Ray::gpu_tlas_sizes(&self.device.native, layout)
    }

    /// Allocates the TLAS before recording the producer or the pack.
    pub fn create_gpu_tlas(&self, layout: GpuTlasLayout) -> Result<AccelerationStructure<B>> {
        self.validate_gpu_tlas_layout(layout)?;
        let native = B::Ray::create_gpu_tlas(&self.device.native, layout)?;
        // SAFETY: the backend allocated top-level storage on this device.
        Ok(unsafe { AccelerationStructure::from_native(self.device.id, native) })
    }

    /// Records GPU packing without reading the logical records on the CPU.
    /// The caller places the source range in compute shader read state, orders
    /// its producer before this command, and orders this command before build.
    /// Invalid logical records cannot be hit and keep their instance ordinal.
    pub fn pack_tlas_instances(
        &self,
        encoder: &mut CommandEncoder<B>,
        desc: GpuTlasPackDesc<'_, B>,
    ) -> Result<PackedTlasInstances<B>>
    where
        B: CommandBackend,
    {
        self.require_gpu_tlas(desc.count)?;
        if encoder.device != self.device.id
            || desc.records.device != self.device.id
            || desc.blas.iter().any(|blas| blas.device != self.device.id)
        {
            return Err(Error::WrongDevice);
        }
        if desc.offset % 16 != 0
            || desc
                .offset
                .checked_add(u64::from(desc.count) * GPU_TLAS_INSTANCE_BYTES)
                .is_none()
        {
            return Err(Error::InvalidRange);
        }
        let native = B::Ray::pack_tlas_instances(encoder.ready()?, &desc)
            .inspect_err(|_| encoder.poisoned = true)?;
        Ok(PackedTlasInstances {
            device: self.device.id,
            count: desc.count,
            native,
        })
    }
}
