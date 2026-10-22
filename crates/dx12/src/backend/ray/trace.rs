//! Shader table validation and Direct3D 12 ray dispatch recording.
use super::super::{Dx12, encoder::Dx12Encoder};
use fluxel_rhi_core::api::{self, Error, Result};
use windows::{Win32::Graphics::Direct3D12::*, core::Interface};

fn region(
    encoder: &mut Dx12Encoder,
    region: &api::ShaderTableRegion<Dx12>,
    raygen: bool,
) -> Result<D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE> {
    // SAFETY: Core validates buffer device identity and the encoder retains
    // the native buffer for the duration of GPU execution below.
    let buffer = unsafe { region.buffer.native_ref() };
    if !buffer.usage.contains(api::BufferUsage::SHADER_TABLE)
        || region.offset % u64::from(D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT) != 0
        || region.stride < u64::from(D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES)
        || region.stride > u64::from(D3D12_RAYTRACING_MAX_SHADER_RECORD_STRIDE)
        || region.stride % u64::from(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT) != 0
        || region.size == 0
        || region.size % region.stride != 0
        || (raygen && region.size != region.stride)
        || region
            .offset
            .checked_add(region.size)
            .is_none_or(|end| end > buffer.size)
    {
        return Err(Error::InvalidRange);
    }
    // SAFETY: a live buffer's address is stable; the offset is range checked.
    let address = unsafe { buffer.resource.GetGPUVirtualAddress() }
        .checked_add(region.offset)
        .ok_or(Error::InvalidRange)?;
    encoder.buffers.push(buffer.clone());
    Ok(D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE {
        StartAddress: address,
        SizeInBytes: region.size,
        StrideInBytes: region.stride,
    })
}

pub(super) fn dispatch(
    encoder: &mut Dx12Encoder,
    table: &api::ShaderTable<Dx12>,
    dims: [u32; 3],
) -> Result<()> {
    if dims.contains(&0)
        || dims
            .iter()
            .try_fold(1u64, |n, &x| n.checked_mul(u64::from(x)))
            .is_none_or(|n| n > u64::from(D3D12_RAYTRACING_MAX_RAY_GENERATION_SHADER_THREADS))
    {
        return Err(Error::InvalidRange);
    }
    let raygen = region(encoder, &table.raygen, true)?;
    let mut optional = |value: &Option<api::ShaderTableRegion<Dx12>>| -> Result<_> {
        value
            .as_ref()
            .map(|r| region(encoder, r, false))
            .transpose()
            .map(Option::unwrap_or_default)
    };
    let native = D3D12_DISPATCH_RAYS_DESC {
        RayGenerationShaderRecord: D3D12_GPU_VIRTUAL_ADDRESS_RANGE {
            StartAddress: raygen.StartAddress,
            SizeInBytes: raygen.SizeInBytes,
        },
        MissShaderTable: optional(&table.miss)?,
        HitGroupTable: optional(&table.hit)?,
        CallableShaderTable: optional(&table.callable)?,
        Width: dims[0],
        Height: dims[1],
        Depth: dims[2],
    };
    let list = encoder
        .list
        .cast::<ID3D12GraphicsCommandList4>()
        .map_err(super::super::native_error)?;
    // SAFETY: all table sections are validated and retained by the encoder;
    // the native command list copies the initialized dispatch descriptor.
    unsafe { list.DispatchRays(&native) };
    Ok(())
}
