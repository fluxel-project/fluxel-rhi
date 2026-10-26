//! GPU packing for WGSL vertex/instance-index indirect ABI records.
//!
//! The public draw buffer stays GPU-resident: this helper expands its native
//! draw record and the two WGSL root constants with one compute dispatch, then
//! feeds the packed records directly to `ExecuteIndirect`. It never reads
//! indirect arguments on the CPU or replays draws.

use super::*;
use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

pub(crate) struct IndirectPackPipeline {
    pub(crate) root: ID3D12RootSignature,
    pub(crate) state: ID3D12PipelineState,
}

pub(crate) struct IndirectPackCache(Mutex<Option<Arc<IndirectPackPipeline>>>);
impl Default for IndirectPackCache {
    fn default() -> Self {
        Self(Mutex::new(None))
    }
}
impl IndirectPackCache {
    pub(crate) fn get_or_create(
        &self,
        device: &Dx12Device,
    ) -> Result<Arc<IndirectPackPipeline>, Error> {
        let mut cache = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(value) = &*cache {
            return Ok(value.clone());
        }
        let value = Arc::new(create(device)?);
        *cache = Some(value.clone());
        Ok(value)
    }
    pub(crate) fn get(&self) -> Result<Arc<IndirectPackPipeline>, Error> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .ok_or(Error::InvalidState)
    }
}

const HLSL: &str = r#"
ByteAddressBuffer source : register(t0);
ByteAddressBuffer source_count : register(t1);
RWByteAddressBuffer target : register(u0);
cbuffer Parameters : register(b0) { uint stride; uint count; uint indexed; uint record_size; uint first; uint use_count; uint count_byte_offset; uint source_byte_offset; };
[numthreads(64, 1, 1)] void main(uint id : SV_DispatchThreadID) {
  uint effective = count;
  if (use_count != 0) {
    uint available = source_count.Load(count_byte_offset);
    effective = available > first ? min(available - first, count) : 0;
    if (id == 0) target.Store(count * (record_size + 8), effective);
  }
  if (id >= effective) return;
  uint source_offset = source_byte_offset + id * stride;
  uint target_offset = id * (record_size + 8);
  uint base_index = source.Load(source_offset + (indexed != 0 ? 12 : 8));
  uint instance_index = source.Load(source_offset + (indexed != 0 ? 16 : 12));
  target.Store(target_offset, base_index); target.Store(target_offset + 4, instance_index);
  for (uint word = 0; word < record_size; word += 4) target.Store(target_offset + 8 + word, source.Load(source_offset + word));
}"#;

fn create(device: &Dx12Device) -> Result<IndirectPackPipeline, Error> {
    let ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            ..Default::default()
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: 1,
            ..Default::default()
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 1,
            ..Default::default()
        },
    ];
    let table = |range: &D3D12_DESCRIPTOR_RANGE| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: 1,
                pDescriptorRanges: range,
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
                Num32BitValues: 8,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let params = [
        table(&ranges[0]),
        table(&ranges[1]),
        table(&ranges[2]),
        constants,
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        ..Default::default()
    };
    let mut blob = None;
    // SAFETY: every root-descriptor pointer remains valid through serialization.
    unsafe { D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None) }
        .map_err(native_error)?;
    let blob = blob.ok_or(Error::OutOfMemory)?;
    // SAFETY: serialized bytes are owned by the blob until root creation returns.
    let bytes =
        unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) };
    // SAFETY: the device owns root creation and the serialized byte slice
    // remains live through this synchronous call.
    let root = unsafe {
        device
            .0
            .native
            .CreateRootSignature::<ID3D12RootSignature>(0, bytes)
    }
    .map_err(native_error)?;
    let dxil = super::super::shader_naga::compile_direct_hlsl(
        HLSL,
        "main",
        api::ShaderStage::Compute,
        60,
        &[],
    )?;
    let mut pso = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: dxil.as_ptr().cast(),
            BytecodeLength: dxil.len(),
        },
        ..Default::default()
    };
    // SAFETY: root and DXIL bytes are live through synchronous PSO creation.
    let state = unsafe {
        device
            .0
            .native
            .CreateComputePipelineState::<ID3D12PipelineState>(&pso)
    }
    .map_err(native_error);
    // SAFETY: release descriptor's temporary root clone exactly once.
    unsafe { ManuallyDrop::drop(&mut pso.pRootSignature) };
    Ok(IndirectPackPipeline {
        root,
        state: state?,
    })
}

pub(crate) struct Packed {
    pub(crate) buffer: Dx12Buffer,
    pub(crate) count_offset: Option<u64>,
}

/// A measured one/three-record guard avoids fixed compute/descriptor cost.
/// Larger batches always use GPU packing, bounding this copy path to six
/// native copies rather than recording copies for a large draw count.
pub(crate) fn copy_small(
    encoder: &mut Dx12Encoder,
    args: &Dx12Buffer,
    offset: u64,
    count: u32,
    stride: u32,
    indexed: bool,
) -> Result<Dx12Buffer, Error> {
    debug_assert!((1..=3).contains(&count));
    let bytes = if indexed { 20u64 } else { 16 };
    let scratch = <Dx12 as fluxel_rhi_core::backend::ResourceBackend>::create_buffer(
        &encoder.device,
        &api::BufferDesc {
            label: None,
            size: (bytes + 8) * u64::from(count),
            usage: api::BufferUsage::COPY_DST | api::BufferUsage::INDIRECT,
            memory: api::MemoryClass::Device,
        },
    )?;
    barrier(
        &encoder.list,
        &args.resource,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    barrier(
        &encoder.list,
        &scratch.resource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    for index in 0..u64::from(count) {
        let source = offset + index * u64::from(stride);
        let target = index * (bytes + 8);
        // SAFETY: admission checked all source records; the private allocation
        // contains exactly these expanded records and is retained below.
        unsafe {
            encoder.list.CopyBufferRegion(
                &scratch.resource,
                target,
                &args.resource,
                source + if indexed { 12 } else { 8 },
                8,
            );
            encoder.list.CopyBufferRegion(
                &scratch.resource,
                target + 8,
                &args.resource,
                source,
                bytes,
            );
        }
    }
    barrier(
        &encoder.list,
        &args.resource,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    );
    barrier(
        &encoder.list,
        &scratch.resource,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    );
    encoder.buffers.push(args.clone());
    Ok(scratch)
}

// Use the guaranteed native view span even on devices with wider allocations.
// Splitting private views does not restrict the public resource/offset range.
const MAX_VIEW_BYTES: u64 = 4u64 << D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP;

/// Bounds a dispatch by the native view, shader byte ABI and group count.
/// The source may start anywhere in a u64 resource; its SRV begins at the
/// preceding 16-byte boundary, leaving at most 12 bytes of low-offset padding.
pub(crate) fn chunk_records(stride: u32, indexed: bool) -> Result<u32, Error> {
    let record_size = if indexed { 20u64 } else { 16 };
    if u64::from(stride) < record_size || stride % 4 != 0 {
        return Err(Error::InvalidRange);
    }
    let byte_window = MAX_VIEW_BYTES.min(u64::from(u32::MAX) + 1);
    let input = (byte_window - 12 - record_size) / u64::from(stride) + 1;
    // Include the optional four-byte GPU count after the packed records.
    let output = (byte_window - 4) / (record_size + 8);
    Ok(input.min(output).min(65_535 * 64) as u32)
}

fn source_window(
    offset: u64,
    count: u32,
    stride: u32,
    indexed: bool,
) -> Result<(u64, u32, u32), Error> {
    if count == 0 {
        return Err(Error::InvalidRange);
    }
    let base = offset & !15;
    let low = offset - base;
    let bytes = low
        .checked_add(u64::from(count - 1) * u64::from(stride))
        .and_then(|value| value.checked_add(if indexed { 20 } else { 16 }))
        .ok_or(Error::InvalidRange)?;
    if bytes > MAX_VIEW_BYTES.min(u64::from(u32::MAX) + 1) {
        return Err(Error::InvalidRange);
    }
    Ok((
        base,
        low as u32,
        u32::try_from(bytes.div_ceil(4)).map_err(|_| Error::InvalidRange)?,
    ))
}
pub(crate) fn record(
    encoder: &mut Dx12Encoder,
    args: &Dx12Buffer,
    offset: u64,
    count: u32,
    stride: u32,
    indexed: bool,
    count_buffer: Option<(&Dx12Buffer, u64)>,
    first: u32,
) -> Result<Packed, Error> {
    let record_size = if indexed { 20 } else { 16 };
    let packed_stride = record_size + 8;
    let scratch = <Dx12 as fluxel_rhi_core::backend::ResourceBackend>::create_buffer(
        &encoder.device,
        &api::BufferDesc {
            label: None,
            size: u64::from(packed_stride) * u64::from(count)
                + u64::from(count_buffer.is_some()) * 4,
            usage: api::BufferUsage::INDIRECT
                | api::BufferUsage::COPY_DST
                | api::BufferUsage::STORAGE,
            memory: api::MemoryClass::Device,
        },
    )?;
    let source_desc = encoder.device.0.descriptors.temporary_pair()?;
    let target_desc = encoder.device.0.descriptors.temporary_pair()?;
    let count_desc = encoder.device.0.descriptors.temporary_pair()?;
    let pipeline = encoder.device.0.indirect_pack_cache.get()?;
    let (source_base, source_low, source_elements) = source_window(offset, count, stride, indexed)?;
    let raw_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                FirstElement: source_base / 4,
                NumElements: source_elements,
                StructureByteStride: 0,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
            },
        },
    };
    let raw_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                FirstElement: 0,
                NumElements: u32::try_from(scratch.size / 4).map_err(|_| Error::InvalidRange)?,
                StructureByteStride: 0,
                CounterOffsetInBytes: 0,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
            },
        },
    };
    let (count_resource, count_offset) = count_buffer.unwrap_or((args, 0));
    let count_base = count_offset & !15;
    let count_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                FirstElement: count_base / 4,
                NumElements: u32::try_from((count_offset - count_base + 4).div_ceil(4))
                    .map_err(|_| Error::InvalidRange)?,
                StructureByteStride: 0,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
            },
        },
    };
    // SAFETY: both descriptor slots and resources are retained by this encoder through the submission fence.
    unsafe {
        encoder.device.0.native.CreateShaderResourceView(
            &args.resource,
            Some(&raw_srv),
            source_desc.view_cpu(),
        );
        encoder.device.0.native.CreateUnorderedAccessView(
            &scratch.resource,
            None::<&ID3D12Resource>,
            Some(&raw_uav),
            target_desc.view_cpu(),
        );
        encoder.device.0.native.CreateShaderResourceView(
            &count_resource.resource,
            Some(&count_srv),
            count_desc.view_cpu(),
        );
    }
    barrier(
        &encoder.list,
        &args.resource,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    );
    if count_resource.resource != args.resource {
        barrier(
            &encoder.list,
            &count_resource.resource,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
    }
    barrier(
        &encoder.list,
        &scratch.resource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    let constants = [
        stride,
        count,
        u32::from(indexed),
        record_size,
        first,
        u32::from(count_buffer.is_some()),
        u32::try_from(count_offset - count_base).expect("raw-count alignment"),
        source_low,
    ];
    // SAFETY: private immutable PSO/root and retained descriptor tables belong to this device; dispatch bounds are checked above.
    unsafe {
        encoder.list.SetComputeRootSignature(&pipeline.root);
        encoder.list.SetPipelineState(&pipeline.state);
        encoder
            .list
            .SetComputeRootDescriptorTable(0, source_desc.view_gpu());
        encoder
            .list
            .SetComputeRootDescriptorTable(1, target_desc.view_gpu());
        encoder.list.SetComputeRoot32BitConstants(
            3,
            constants.len() as u32,
            constants.as_ptr().cast(),
            0,
        );
        encoder
            .list
            .SetComputeRootDescriptorTable(2, count_desc.view_gpu());
        encoder.list.Dispatch(count.div_ceil(64), 1, 1);
    }
    barrier(
        &encoder.list,
        &args.resource,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    );
    if count_resource.resource != args.resource {
        barrier(
            &encoder.list,
            &count_resource.resource,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        );
    }
    barrier(
        &encoder.list,
        &scratch.resource,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    );
    encoder
        .mip_blit_descriptors
        .extend([source_desc, target_desc, count_desc]);
    encoder.buffers.push(args.clone());
    if let Some((buffer, _)) = count_buffer {
        encoder.buffers.push(buffer.clone());
    }
    Ok(Packed {
        buffer: scratch,
        count_offset: count_buffer.map(|_| u64::from(packed_stride) * u64::from(count)),
    })
}

#[cfg(test)]
#[path = "../../tests/backend/encoder/indirect_pack.rs"]
mod tests;
