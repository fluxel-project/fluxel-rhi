//! GPU conversion from logical TLAS instances to private DXR instance records.

use std::{
    collections::HashSet,
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use fluxel_rhi_core::{
    api::{self, Error, Result},
    backend::ResourceBackend,
};
use windows::Win32::Graphics::Direct3D12::*;

use super::{Dx12, Dx12Acceleration, Dx12Device, PreparedInputId, PreparedInputOp};
use crate::backend::{
    Dx12Buffer,
    encoder::{Dx12Encoder, barrier},
    native_error, shader_naga,
};

pub(super) const MAX_TLAS_INSTANCES: u32 =
    D3D12_RAYTRACING_MAX_INSTANCES_PER_TOP_LEVEL_ACCELERATION_STRUCTURE;
const GROUP_SIZE: u32 = 64;
const _: [(); api::GPU_TLAS_INSTANCE_BYTES as usize] =
    [(); std::mem::size_of::<D3D12_RAYTRACING_INSTANCE_DESC>()];
const HLSL: &str = r#"
ByteAddressBuffer logical_records : register(t0);
ByteAddressBuffer blas_addresses : register(t1);
RWByteAddressBuffer native_records : register(u0);
cbuffer Parameters : register(b0) { uint count; uint table_count; uint groups_x; };
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint index = (group.y * groups_x + group.x) * 64 + lane;
    if (index >= count) return;
    uint offset = 64 * index;
    uint4 row0 = logical_records.Load4(offset);
    uint4 row1 = logical_records.Load4(offset + 16);
    uint4 row2 = logical_records.Load4(offset + 32);
    uint4 tail = logical_records.Load4(offset + 48);
    uint flags = tail.y >> 24;
    bool valid = tail.z < table_count && tail.w == 0 && (flags & ~15u) == 0
              && (flags & 12u) != 12u && all(isfinite(asfloat(row0)))
              && all(isfinite(asfloat(row1))) && all(isfinite(asfloat(row2)));
    [branch]
    if (valid) {
        uint2 address = blas_addresses.Load2(8 * tail.z);
        tail.zw = address;
    } else {
        uint one = asuint(1.0f);
        row0 = uint4(one, 0, 0, 0);
        row1 = uint4(0, one, 0, 0);
        row2 = uint4(0, 0, one, 0);
        // Keep nonempty-table instances active for native refit, with a zero
        // mask that excludes them from all rays. An empty table stays null.
        uint2 fallback = 0;
        [branch] if (table_count != 0) fallback = blas_addresses.Load2(0);
        tail = uint4(0, 0, fallback.x, fallback.y);
    }
    native_records.Store4(offset, row0);
    native_records.Store4(offset + 16, row1);
    native_records.Store4(offset + 32, row2);
    native_records.Store4(offset + 48, tail);
}
"#;

struct PackPipeline {
    root: ID3D12RootSignature,
    state: ID3D12PipelineState,
}

#[derive(Default)]
pub(crate) struct TlasPackCache(Mutex<Option<Arc<PackPipeline>>>);

impl TlasPackCache {
    fn get_or_create(&self, device: &Dx12Device) -> Result<Arc<PackPipeline>> {
        let mut cache = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(pipeline) = &*cache {
            return Ok(pipeline.clone());
        }
        let pipeline = Arc::new(create_pipeline(device)?);
        *cache = Some(pipeline.clone());
        Ok(pipeline)
    }
}

fn create_pipeline(device: &Dx12Device) -> Result<PackPipeline> {
    let descriptor = |kind, register| D3D12_ROOT_PARAMETER {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: register,
                RegisterSpace: 0,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let constants = D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: 0,
                RegisterSpace: 0,
                Num32BitValues: 3,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let entries = [
        descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV, 0),
        descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV, 1),
        descriptor(D3D12_ROOT_PARAMETER_TYPE_UAV, 0),
        constants,
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: entries.len() as u32,
        pParameters: entries.as_ptr(),
        ..Default::default()
    };
    let mut blob = None;
    // SAFETY: the signature and backing entries are live through serialization.
    unsafe { D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None) }
        .map_err(native_error)?;
    let blob = blob.ok_or(Error::OutOfMemory)?;
    // SAFETY: the blob owns the serialized signature during root creation.
    let root: ID3D12RootSignature = unsafe {
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize());
        device.0.native.CreateRootSignature(0, bytes)
    }
    .map_err(native_error)?;
    let code = shader_naga::compile_direct_hlsl(HLSL, "main", api::ShaderStage::Compute, 60, &[])?;
    let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: code.as_ptr().cast(),
            BytecodeLength: code.len(),
        },
        ..Default::default()
    };
    // SAFETY: the bytecode and root are live during synchronous creation. The
    // temporary native descriptor reference is released before returning.
    let state = unsafe {
        let state = device.0.native.CreateComputePipelineState(&desc);
        ManuallyDrop::drop(&mut desc.pRootSignature);
        state
    }
    .map_err(native_error)?;
    Ok(PackPipeline { root, state })
}

#[derive(Clone)]
pub struct Dx12PackedTlas {
    pub(super) records: Dx12Buffer,
    pub(super) blas: Arc<[Dx12Acceleration]>,
    pub(super) id: PreparedInputId,
}

pub(super) fn pack_cost(
    device: &Dx12Device,
    count: u32,
    blas_count: u32,
) -> Result<api::GpuTlasPackCost> {
    device.require_active()?;
    super::ray_device(device)?;
    if !(1..=MAX_TLAS_INSTANCES).contains(&count) || blas_count > MAX_TLAS_INSTANCES {
        return Err(Error::InvalidRange);
    }
    let groups = count.div_ceil(GROUP_SIZE);
    let x = groups.min(D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
    Ok(api::GpuTlasPackCost {
        packed_bytes: u64::from(count) * api::GPU_TLAS_INSTANCE_BYTES,
        address_upload_bytes: u64::from(blas_count) * 8,
        workgroups: [x, groups.div_ceil(x), 1],
    })
}

pub(super) fn sizes(
    device: &Dx12Device,
    layout: api::GpuTlasLayout,
) -> Result<api::AccelerationSizes> {
    device.require_active()?;
    let geometry = super::geometry::GeometryInputs::tlas(layout)?;
    super::allocation_sizes(device, &geometry, false)
}

pub(super) fn pack(
    encoder: &mut Dx12Encoder,
    desc: &api::GpuTlasPackDesc<'_, Dx12>,
) -> Result<Dx12PackedTlas> {
    let table_count = u32::try_from(desc.blas.len()).map_err(|_| Error::InvalidRange)?;
    let cost = pack_cost(&encoder.device, desc.count, table_count)?;
    // SAFETY: Core checked the input's device and it remains borrowed here.
    let source = unsafe { desc.records.native_ref() };
    if desc.offset % 16 != 0
        || desc
            .offset
            .checked_add(cost.packed_bytes)
            .is_none_or(|end| end > source.size)
    {
        return Err(Error::InvalidRange);
    }
    if !source.usage.contains(api::BufferUsage::STORAGE) {
        return Err(Error::Unsupported(
            "DX12 GPU TLAS records require STORAGE usage",
        ));
    }
    // SAFETY: the buffer is live and retained with the recorded pack.
    let source_address = unsafe { source.resource.GetGPUVirtualAddress() }
        .checked_add(desc.offset)
        .filter(|address| address % 16 == 0)
        .ok_or(Error::InvalidRange)?;
    let mut blas = Vec::with_capacity(desc.blas.len());
    let mut owners = HashSet::new();
    for accel in desc.blas {
        // SAFETY: Core checked DeviceId; the typed handles remain live.
        let accel = unsafe { accel.native_ref() };
        if accel.0.is_top_level {
            return Err(Error::InvalidState);
        }
        if owners.insert(Arc::as_ptr(&accel.0)) {
            blas.push(accel.clone());
        }
    }
    let table_address = if desc.blas.is_empty() {
        // An empty table is never read. Bind the valid source instead of a null root SRV.
        source_address
    } else {
        let (table, offset) = encoder
            .upload_arena
            .write_with(
                &encoder.device.0.native,
                cost.address_upload_bytes as usize,
                16,
                |bytes| {
                    for (word, accel) in bytes.chunks_exact_mut(8).zip(desc.blas) {
                        // SAFETY: validated BLAS handles own the native resources,
                        // and their clones retain the addresses through submission.
                        let address =
                            unsafe { accel.native_ref().0.resource.GetGPUVirtualAddress() };
                        word.copy_from_slice(&address.to_le_bytes());
                    }
                    true
                },
            )?
            .ok_or(Error::InvalidState)?;
        // SAFETY: the upload arena retains the mapped table page through completion.
        unsafe { table.GetGPUVirtualAddress() }
            .checked_add(offset)
            .ok_or(Error::InvalidRange)?
    };
    let records = Dx12::create_buffer(
        &encoder.device,
        &api::BufferDesc {
            label: None,
            size: cost.packed_bytes,
            usage: api::BufferUsage::STORAGE | api::BufferUsage::ACCEL_BUILD_INPUT,
            memory: api::MemoryClass::Device,
        },
    )?;
    let pipeline = encoder
        .device
        .0
        .tlas_pack_cache
        .get_or_create(&encoder.device)?;
    let [x, y, z] = cost.workgroups;
    let constants = [desc.count, table_count, x];
    barrier(
        &encoder.list,
        &records.resource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    // SAFETY: typed native owners are retained, shader accesses are bounded by
    // the counts above, and only this private output is written by the pack.
    unsafe {
        encoder.list.SetComputeRootSignature(&pipeline.root);
        encoder.list.SetPipelineState(&pipeline.state);
        encoder
            .list
            .SetComputeRootShaderResourceView(0, source_address);
        encoder
            .list
            .SetComputeRootShaderResourceView(1, table_address);
        encoder
            .list
            .SetComputeRootUnorderedAccessView(2, records.resource.GetGPUVirtualAddress());
        encoder
            .list
            .SetComputeRoot32BitConstants(3, 3, constants.as_ptr().cast(), 0);
        encoder.list.Dispatch(x, y, z);
    }
    barrier(
        &encoder.list,
        &records.resource,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    );
    encoder.buffers.extend([source.clone(), records.clone()]);
    encoder.accelerations.extend(blas.iter().cloned());
    crate::backend::compute::restore_compute_state(encoder)?;
    let id = PreparedInputId::new();
    encoder.ray_inputs.push(PreparedInputOp::Initialize {
        id: id.clone(),
        segment: encoder.closed_segments.len(),
    });
    Ok(Dx12PackedTlas {
        records,
        blas: blas.into(),
        id,
    })
}
