//! Device-local Direct3D 12 ray tracing support.
//!
//! DXR is optional even on D3D12 devices. Query the actual adapter tier and
//! obtain the required interface before attempting any ray tracing operation.
use windows::{
    Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS5, D3D12_FEATURE_DATA_D3D12_OPTIONS5, D3D12_RAYTRACING_TIER_1_0,
        ID3D12Device, ID3D12Device5,
    },
    core::Interface,
};

mod geometry;
mod ops;
pub use ops::Ray;
pub(crate) mod pipeline;
mod trace;

use super::{Dx12, Dx12Acceleration, Dx12Device};
use fluxel_rhi_core::api::{self, Error, Result};
use std::sync::{Arc, Mutex};
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

pub(crate) struct AccelerationState {
    pub(crate) resource: ID3D12Resource,
    pub(crate) size: u64,
    pub(crate) is_top_level: bool,
    // A TLAS stores BLAS GPU virtual addresses. Every recorded build version
    // stays valid until the TLAS itself is dropped, including concurrent or
    // abandoned builds: completion order cannot safely select a single owner.
    referenced_blas: Mutex<Vec<Arc<AccelerationState>>>,
}

fn retain_unique<T>(retained: &mut Vec<Arc<T>>, incoming: impl IntoIterator<Item = Arc<T>>) {
    for value in incoming {
        if !retained
            .iter()
            .any(|existing| Arc::ptr_eq(existing, &value))
        {
            retained.push(value);
        }
    }
}

fn ray_device(device: &Dx12Device) -> Result<&ID3D12Device5> {
    device
        .0
        .ray_device
        .as_ref()
        .ok_or(Error::Unsupported("DX12 ray tracing tier 1.0"))
}

fn sizes(
    device: &Dx12Device,
    desc: &api::AccelerationDesc<Dx12>,
) -> Result<api::AccelerationSizes> {
    let geometry = geometry::GeometryInputs::new(desc)?;
    let input = geometry.native(0);
    let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    // SAFETY: the descriptor and its geometry array remain alive throughout
    // this synchronous native prebuild query; the output is initialized storage.
    unsafe {
        ray_device(device)?.GetRaytracingAccelerationStructurePrebuildInfo(&input, &mut info)
    };
    if info.ResultDataMaxSizeInBytes == 0 || info.ScratchDataSizeInBytes == 0 {
        return Err(Error::Unsupported("DX12 acceleration structure input"));
    }
    Ok(api::AccelerationSizes {
        storage_bytes: info.ResultDataMaxSizeInBytes,
        scratch_bytes: info
            .ScratchDataSizeInBytes
            .max(info.UpdateScratchDataSizeInBytes),
        scratch_alignment: u64::from(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT),
    })
}

fn create_acceleration(
    device: &Dx12Device,
    desc: &api::AccelerationDesc<Dx12>,
) -> Result<Dx12Acceleration> {
    let size = sizes(device, desc)?.storage_bytes;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    };
    let resource_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };
    let mut resource = None;
    // SAFETY: descriptors and output slot are valid for this synchronous
    // creation call, which returns an owned COM resource.
    unsafe {
        device.0.native.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &resource_desc,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            None,
            &mut resource,
        )
    }
    .map_err(super::native_error)?;
    Ok(Dx12Acceleration(Arc::new(AccelerationState {
        resource: resource.ok_or(Error::OutOfMemory)?,
        size,
        is_top_level: matches!(desc.input, api::AccelerationInput::Tlas(_)),
        referenced_blas: Mutex::new(Vec::new()),
    })))
}

fn build(
    encoder: &mut super::encoder::Dx12Encoder,
    dst: &Dx12Acceleration,
    scratch: &super::Dx12Buffer,
    desc: &api::AccelerationDesc<Dx12>,
) -> Result<()> {
    let sizes = sizes(&encoder.device, desc)?;
    if !scratch.usage.contains(api::BufferUsage::ACCEL_SCRATCH)
        || scratch.size < sizes.scratch_bytes
        || dst.0.size < sizes.storage_bytes
        || dst.0.is_top_level != matches!(desc.input, api::AccelerationInput::Tlas(_))
    {
        return Err(Error::InvalidRange);
    }
    let geometry = geometry::GeometryInputs::new(desc)?;
    let mut instance_address = 0;
    let mut referenced_blas = Vec::new();
    if let api::AccelerationInput::Tlas(instances) = &desc.input {
        let mut native_instances = Vec::with_capacity(instances.len());
        for instance in instances {
            if instance.custom_index > 0x00ff_ffff || instance.shader_table_offset > 0x00ff_ffff {
                return Err(Error::InvalidRange);
            }
            // SAFETY: Core validates the device of each BLAS; cloning below
            // retains its native allocation until the submission retires.
            let blas = unsafe { instance.blas.native_ref() };
            if blas.0.is_top_level {
                return Err(Error::InvalidState);
            }
            let transform = instance
                .transform
                .concat()
                .try_into()
                .map_err(|_| Error::InvalidRange)?;
            // SAFETY: the retained acceleration structure has a live resource.
            let address = unsafe { blas.0.resource.GetGPUVirtualAddress() };
            let mut flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
            for (public, native) in [
                (
                    api::AccelerationInstanceFlags::TRIANGLE_CULL_DISABLE,
                    D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE,
                ),
                (
                    api::AccelerationInstanceFlags::TRIANGLE_FRONT_COUNTER_CLOCKWISE,
                    D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE,
                ),
                (
                    api::AccelerationInstanceFlags::FORCE_OPAQUE,
                    D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE,
                ),
                (
                    api::AccelerationInstanceFlags::FORCE_NON_OPAQUE,
                    D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE,
                ),
            ] {
                if instance.flags.contains(public) {
                    flags |= native;
                }
            }
            native_instances.push(D3D12_RAYTRACING_INSTANCE_DESC {
                Transform: transform,
                _bitfield1: instance.custom_index | (u32::from(instance.mask) << 24),
                _bitfield2: instance.shader_table_offset | ((flags.0 as u32) << 24),
                AccelerationStructure: address,
            });
            encoder.accelerations.push(blas.clone());
            referenced_blas.push(blas.0.clone());
        }
        let length = native_instances
            .len()
            .checked_mul(std::mem::size_of::<D3D12_RAYTRACING_INSTANCE_DESC>())
            .ok_or(Error::InvalidRange)?;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let resource_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: u64::try_from(length).map_err(|_| Error::InvalidRange)?,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut upload: Option<ID3D12Resource> = None;
        // SAFETY: initialized descriptors and a valid output slot are supplied;
        // the upload resource is retained by the encoder after this call.
        unsafe {
            encoder.device.0.native.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &resource_desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut upload,
            )
        }
        .map_err(super::native_error)?;
        let upload = upload.ok_or(Error::OutOfMemory)?;
        let mut mapped = std::ptr::null_mut();
        // SAFETY: this fresh upload buffer has `length` bytes and is mapped
        // exclusively here; the input slice covers exactly that byte count.
        unsafe {
            upload
                .Map(0, None, Some(&mut mapped))
                .map_err(super::native_error)?;
            std::ptr::copy_nonoverlapping(
                native_instances.as_ptr().cast::<u8>(),
                mapped.cast::<u8>(),
                length,
            );
            upload.Unmap(0, None);
            instance_address = upload.GetGPUVirtualAddress();
        }
        encoder.uploads.push(upload);
    }
    // SAFETY: resources are live and kept by the encoder through completion.
    let scratch_address = unsafe { scratch.resource.GetGPUVirtualAddress() };
    if scratch_address % sizes.scratch_alignment != 0 {
        return Err(Error::InvalidRange);
    }
    let native = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
        // SAFETY: the destination resource remains alive in the encoder.
        DestAccelerationStructureData: unsafe { dst.0.resource.GetGPUVirtualAddress() },
        Inputs: geometry.native(instance_address),
        SourceAccelerationStructureData: 0,
        ScratchAccelerationStructureData: scratch_address,
    };
    let list = encoder
        .list
        .cast::<ID3D12GraphicsCommandList4>()
        .map_err(super::native_error)?;
    if !referenced_blas.is_empty() {
        let mut retained = dst
            .0
            .referenced_blas
            .lock()
            .map_err(|_| Error::Backend("DX12 acceleration ownership lock poisoned".into()))?;
        retain_unique(&mut retained, referenced_blas);
    }
    // SAFETY: native geometry pointers are valid for the synchronous command
    // recording call; caller-provided resource states follow Core barriers.
    unsafe { list.BuildRaytracingAccelerationStructure(&native, None) };
    encoder.buffers.extend(geometry.buffers);
    encoder.buffers.push(scratch.clone());
    encoder.accelerations.push(dst.clone());
    Ok(())
}

pub(super) fn probe(device: &ID3D12Device) -> Option<ID3D12Device5> {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
    // SAFETY: the output structure matches the requested feature and lives
    // throughout the synchronous feature query.
    let supported = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS5,
            (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS5).cast(),
            std::mem::size_of_val(&options) as u32,
        )
    }
    .is_ok()
        && options.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_0.0;
    supported.then(|| device.cast().ok()).flatten()
}

/// Inline ray queries require both DXR 1.1 and shader model 6.5.
pub(super) fn probe_inline(device: &ID3D12Device) -> bool {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
    let mut shader_model = D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: D3D_SHADER_MODEL_6_5,
    };
    // SAFETY: each feature query receives the exact initialized structure and
    // performs synchronous writes while the borrowed storage remains valid.
    let has_tier = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS5,
            (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS5).cast(),
            std::mem::size_of_val(&options) as u32,
        )
    }
    .is_ok()
        && options.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0;
    let has_shader_model = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            (&mut shader_model as *mut D3D12_FEATURE_DATA_SHADER_MODEL).cast(),
            std::mem::size_of_val(&shader_model) as u32,
        )
    }
    .is_ok()
        && shader_model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_5.0;
    has_tier && has_shader_model && device.cast::<ID3D12Device5>().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tlas_retains_each_distinct_blas_until_tlas_release() {
        let first = Arc::new(());
        let second = Arc::new(());
        let first_weak = Arc::downgrade(&first);
        let second_weak = Arc::downgrade(&second);
        let mut retained = Vec::new();
        retain_unique(&mut retained, [first.clone(), first.clone()]);
        retain_unique(&mut retained, [second.clone()]);
        assert_eq!(retained.len(), 2);
        drop(first);
        drop(second);
        assert!(first_weak.upgrade().is_some());
        assert!(second_weak.upgrade().is_some());
        drop(retained);
        assert!(first_weak.upgrade().is_none());
        assert!(second_weak.upgrade().is_none());
    }
}
