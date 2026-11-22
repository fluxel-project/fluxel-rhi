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

mod compaction;
mod copy;
mod geometry;
mod ops;
pub use ops::Ray;
pub(crate) mod pipeline;
mod refit;
pub(crate) use refit::validate_sequence as validate_acceleration_builds;
mod trace;

#[cfg(all(test, feature = "hardware-tests"))]
#[path = "../../tests/backend/ray/retention_gpu.rs"]
mod retention_gpu;

use super::{Dx12, Dx12Acceleration, Dx12Device};
use fluxel_rhi_core::api::{self, Error, Result};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

pub(crate) struct AccelerationState {
    pub(crate) resource: ID3D12Resource,
    pub(crate) size: u64,
    pub(crate) is_top_level: bool,
    // The BLAS addresses in native TLAS storage are mutable in place.  This
    // names only the version that crossed ExecuteCommandLists; recording and
    // cancellation never alter it.
    committed_blas: Mutex<Vec<Dx12Acceleration>>,
    committed_signature: Mutex<Option<Arc<refit::RefitSignature>>>,
    committed_revision: Mutex<Option<Arc<()>>>,
    committed_built: AtomicBool,
}

/// An acceleration build attached to the native list segment that records it.
/// It is metadata for ownership publication, never a replayable GPU command.
pub(crate) struct RecordedAccelerationBuild {
    pub(crate) target: Dx12Acceleration,
    pub(crate) blas: Vec<Dx12Acceleration>,
    pub(crate) segment: usize,
    signature: Option<Arc<refit::RefitSignature>>,
    refit: bool,
    receipt: Option<Arc<compaction::CompactionReceipt>>,
    source: Option<CopySource>,
}

enum CopySource {
    Clone(Dx12Acceleration),
    Compaction(Dx12Acceleration, Arc<()>),
}

fn extend_unique(
    retained: &mut Vec<Dx12Acceleration>,
    incoming: impl IntoIterator<Item = Dx12Acceleration>,
) {
    for value in incoming {
        if !retained
            .iter()
            .any(|existing| Arc::ptr_eq(&existing.0, &value.0))
        {
            retained.push(value);
        }
    }
}

fn committed(acceleration: &Dx12Acceleration) -> Vec<Dx12Acceleration> {
    acceleration
        .0
        .committed_blas
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// Pins the conservative dependency union for one native Execute batch.
///
/// `bound_groups` covers raster and compute inline ray queries as well as
/// dispatch rays: the backend cannot infer which shader instruction consumes
/// an acceleration structure, so every retained binding is a possible use.
pub(crate) fn pin_batch(work: &mut [super::encoder::Dx12RecordedWork]) {
    if work.is_empty() {
        return;
    }
    let mut dependencies = Vec::new();
    for record in work.iter() {
        for group in &record.encoder.bound_groups {
            for acceleration in group.0.accelerations() {
                if acceleration.0.is_top_level {
                    extend_unique(&mut dependencies, committed(acceleration));
                }
            }
        }
        for secondary in &record.encoder.secondary {
            for group in &secondary.groups {
                for acceleration in group.0.accelerations() {
                    if acceleration.0.is_top_level {
                        extend_unique(&mut dependencies, committed(acceleration));
                    }
                }
            }
        }
        for build in &record.encoder.ray_builds {
            if build.refit && build.target.0.is_top_level {
                extend_unique(&mut dependencies, committed(&build.target));
            }
            extend_unique(&mut dependencies, build.blas.clone());
            if let Some(CopySource::Clone(source)) = &build.source {
                if source.0.is_top_level {
                    extend_unique(&mut dependencies, committed(source));
                }
            }
        }
    }
    // One record in this accepted batch owns the union through the enclosing
    // fence-retired `issued` list. This is bounded by that batch, never TLAS
    // history, and also protects an indeterminate handoff quarantine.
    extend_unique(&mut work[0].encoder.accelerations, dependencies);
}

/// Publishes just the builds whose native command-list segment was accepted.
pub(crate) fn publish_segment(work: &super::encoder::Dx12RecordedWork, segment: usize) {
    for build in work
        .encoder
        .ray_builds
        .iter()
        .filter(|build| build.segment == segment)
    {
        let (blas, signature) = match &build.source {
            Some(CopySource::Clone(source)) => (committed(source), committed_signature(source)),
            _ => (build.blas.clone(), build.signature.clone()),
        };
        if build.target.0.is_top_level {
            *build
                .target
                .0
                .committed_blas
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = blas;
        }
        *build
            .target
            .0
            .committed_signature
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = signature;
        *build
            .target
            .0
            .committed_revision
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = build
            .receipt
            .as_ref()
            .map(|receipt| receipt.revision.clone());
        // Both publication and admission read this bit under the submission gate.
        build
            .target
            .0
            .committed_built
            .store(true, Ordering::Relaxed);
    }
}

fn committed_signature(src: &Dx12Acceleration) -> Option<Arc<refit::RefitSignature>> {
    src.0
        .committed_signature
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
}

pub(crate) fn seal_compaction_receipts(
    work: &super::encoder::Dx12RecordedWork,
    fence: &ID3D12Fence,
    value: u64,
) {
    for receipt in work
        .encoder
        .ray_builds
        .iter()
        .filter_map(|build| build.receipt.as_ref())
    {
        receipt.seal(fence, value);
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
    compactable: bool,
) -> Result<api::AccelerationSizes> {
    let geometry = geometry::GeometryInputs::new(desc, device.0.dxr_1_1)?;
    let info = prebuild(device, &geometry, compactable)?;
    Ok(api::AccelerationSizes {
        storage_bytes: info.ResultDataMaxSizeInBytes,
        scratch_bytes: info
            .ScratchDataSizeInBytes
            .max(info.UpdateScratchDataSizeInBytes),
        scratch_alignment: u64::from(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT),
    })
}

fn prebuild(
    device: &Dx12Device,
    geometry: &geometry::GeometryInputs,
    compactable: bool,
) -> Result<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO> {
    let mut input = geometry.native(0);
    if compactable {
        input.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    }
    let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    // SAFETY: the descriptor and its geometry array remain alive throughout
    // this synchronous native prebuild query; the output is initialized storage.
    unsafe {
        ray_device(device)?.GetRaytracingAccelerationStructurePrebuildInfo(&input, &mut info)
    };
    if info.ResultDataMaxSizeInBytes == 0 || info.ScratchDataSizeInBytes == 0 {
        return Err(Error::Unsupported("DX12 acceleration structure input"));
    }
    Ok(info)
}

fn create_acceleration(
    device: &Dx12Device,
    desc: &api::AccelerationDesc<Dx12>,
    compactable: bool,
) -> Result<Dx12Acceleration> {
    let size = sizes(device, desc, compactable)?.storage_bytes;
    allocate_acceleration(
        device,
        size,
        matches!(desc.input, api::AccelerationInput::Tlas(_)),
    )
}

fn allocate_acceleration(
    device: &Dx12Device,
    size: u64,
    is_top_level: bool,
) -> Result<Dx12Acceleration> {
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
        is_top_level,
        committed_blas: Mutex::new(Vec::new()),
        committed_signature: Mutex::new(None),
        committed_revision: Mutex::new(None),
        committed_built: AtomicBool::new(false),
    })))
}

fn build(
    encoder: &mut super::encoder::Dx12Encoder,
    dst: &Dx12Acceleration,
    scratch: &super::Dx12Buffer,
    desc: &api::AccelerationDesc<Dx12>,
    refit: bool,
    output: Option<(&super::Dx12Buffer, u64, api::AccelerationSizeKind)>,
) -> Result<()> {
    let compactable =
        output.is_some_and(|(_, _, kind)| kind == api::AccelerationSizeKind::Compacted);
    let signature = refit::RefitSignature::from_desc(desc, compactable);
    if refit && signature.is_none() {
        return Err(Error::Unsupported("DX12 indexed BLAS refit"));
    }
    let geometry = geometry::GeometryInputs::new(desc, encoder.device.0.dxr_1_1)?;
    let postbuild = output
        .map(|(buffer, offset, kind)| {
            if buffer.memory != api::MemoryClass::Device
                || !buffer.usage.contains(api::BufferUsage::STORAGE)
            {
                return Err(Error::Unsupported(
                    "DX12 acceleration size output requires a device storage buffer",
                ));
            }
            if offset.checked_add(8).is_none_or(|end| end > buffer.size) {
                return Err(Error::InvalidRange);
            }
            if buffer.resource.as_raw() == scratch.resource.as_raw()
                || geometry
                    .buffers
                    .iter()
                    .any(|input| input.resource.as_raw() == buffer.resource.as_raw())
            {
                return Err(Error::InvalidState);
            }
            // SAFETY: the retained buffer owns its native address.
            let base = unsafe { buffer.resource.GetGPUVirtualAddress() };
            let address = base
                .checked_add(offset)
                .filter(|address| address % 8 == 0)
                .ok_or(Error::InvalidRange)?;
            Ok(
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                    DestBuffer: address,
                    InfoType: match kind {
                        api::AccelerationSizeKind::Current => {
                            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE
                        }
                        api::AccelerationSizeKind::Compacted => {
                            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE
                        }
                    },
                },
            )
        })
        .transpose()?;
    let sizes = prebuild(&encoder.device, &geometry, compactable)?;
    let scratch_bytes = if refit {
        if sizes.UpdateScratchDataSizeInBytes == 0 {
            return Err(Error::Unsupported("DX12 acceleration input refit"));
        }
        sizes.UpdateScratchDataSizeInBytes
    } else {
        sizes
            .ScratchDataSizeInBytes
            .max(sizes.UpdateScratchDataSizeInBytes)
    };
    if !scratch.usage.contains(api::BufferUsage::ACCEL_SCRATCH)
        || scratch.size < scratch_bytes
        || (!refit && dst.0.size < sizes.ResultDataMaxSizeInBytes)
        || dst.0.is_top_level != matches!(desc.input, api::AccelerationInput::Tlas(_))
    {
        return Err(Error::InvalidRange);
    }
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
            referenced_blas.push(blas.clone());
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
    if scratch_address % u64::from(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT) != 0 {
        return Err(Error::InvalidRange);
    }
    // SAFETY: the destination resource is retained through this recording.
    let dst_address = unsafe { dst.0.resource.GetGPUVirtualAddress() };
    let mut inputs = geometry.native(instance_address);
    if compactable {
        inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    }
    if refit {
        inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    }
    let native = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
        DestAccelerationStructureData: dst_address,
        Inputs: inputs,
        SourceAccelerationStructureData: if refit { dst_address } else { 0 },
        ScratchAccelerationStructureData: scratch_address,
    };
    let list = encoder
        .list
        .cast::<ID3D12GraphicsCommandList4>()
        .map_err(super::native_error)?;
    // SAFETY: native geometry pointers are valid for the synchronous command
    // recording call; caller-provided resource states follow Core barriers.
    unsafe {
        list.BuildRaytracingAccelerationStructure(
            &native,
            postbuild.as_ref().map(std::slice::from_ref),
        )
    };
    encoder.buffers.extend(geometry.buffers);
    encoder.buffers.push(scratch.clone());
    if let Some((buffer, _, _)) = output {
        encoder.buffers.push(buffer.clone());
    }
    encoder.accelerations.push(dst.clone());
    let mut unique = Vec::new();
    extend_unique(&mut unique, referenced_blas);
    encoder.ray_builds.push(RecordedAccelerationBuild {
        target: dst.clone(),
        blas: unique,
        segment: encoder.closed_segments.len(),
        signature,
        refit,
        receipt: None,
        source: None,
    });
    Ok(())
}

pub(super) fn probe(device: &ID3D12Device) -> (Option<ID3D12Device5>, bool) {
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
    let ray = supported.then(|| device.cast().ok()).flatten();
    let dxr_1_1 = ray.is_some() && options.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0;
    (ray, dxr_1_1)
}
