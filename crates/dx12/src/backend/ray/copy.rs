//! Direct DXR acceleration structure copies into caller-owned storage.

use fluxel_rhi_core::api::{Error, Result};
use windows::{Win32::Graphics::Direct3D12::*, core::Interface};

use super::{CopySource, Dx12Acceleration, RecordedAccelerationBuild, ray_device};
use crate::backend::{encoder::Dx12Encoder, native_error};

pub(super) fn record_clone(
    encoder: &mut Dx12Encoder,
    dst: &Dx12Acceleration,
    src: &Dx12Acceleration,
) -> Result<()> {
    encoder.device.require_active()?;
    ray_device(&encoder.device)?;
    if src.0.is_top_level != dst.0.is_top_level {
        return Err(Error::InvalidState);
    }
    if dst.0.size < src.0.size {
        return Err(Error::InvalidRange);
    }
    // SAFETY: both acceleration handles retain their device-owned allocation.
    let (src_address, dst_address) = unsafe {
        (
            src.0.resource.GetGPUVirtualAddress(),
            dst.0.resource.GetGPUVirtualAddress(),
        )
    };
    let src_end = src_address
        .checked_add(src.0.size)
        .ok_or(Error::InvalidRange)?;
    let dst_end = dst_address
        .checked_add(dst.0.size)
        .ok_or(Error::InvalidRange)?;
    if src_address < dst_end && dst_address < src_end {
        return Err(Error::InvalidRange);
    }
    let list = encoder
        .list
        .cast::<ID3D12GraphicsCommandList4>()
        .map_err(native_error)?;
    // SAFETY: distinct same-kind allocations are large enough and in the
    // acceleration structure state. Admission checks that the source has a
    // preceding build or copy, and the caller supplies execution ordering.
    unsafe {
        list.CopyRaytracingAccelerationStructure(
            dst_address,
            src_address,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE,
        );
    }
    encoder.accelerations.extend([src.clone(), dst.clone()]);
    encoder.ray_builds.push(RecordedAccelerationBuild {
        target: dst.clone(),
        blas: Vec::new(),
        segment: encoder.closed_segments.len(),
        signature: None,
        refit: false,
        receipt: None,
        source: Some(CopySource::Clone(src.clone())),
        validate_blas: false,
    });
    Ok(())
}
