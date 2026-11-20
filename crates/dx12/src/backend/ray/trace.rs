//! Shader table validation and Direct3D 12 ray dispatch recording.
use super::super::{Dx12, encoder::Dx12Encoder};
use fluxel_rhi_core::api::{self, Error, Result};
use windows::{Win32::Graphics::Direct3D12::*, core::Interface};

fn region(
    region: &api::ShaderTableRegion<Dx12>,
    raygen: bool,
) -> Result<D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE> {
    // SAFETY: this query borrows the live buffer through the table region.
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
    Ok(D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE {
        StartAddress: address,
        SizeInBytes: region.size,
        StrideInBytes: region.stride,
    })
}

pub(super) fn prepare(
    table: &api::ShaderTable<Dx12>,
    dims: [u32; 3],
) -> Result<D3D12_DISPATCH_RAYS_DESC> {
    if dims.contains(&0)
        || dims
            .iter()
            .try_fold(1u64, |n, &x| n.checked_mul(u64::from(x)))
            .is_none_or(|n| n > u64::from(D3D12_RAYTRACING_MAX_RAY_GENERATION_SHADER_THREADS))
    {
        return Err(Error::InvalidRange);
    }
    let raygen = region(&table.raygen, true)?;
    let optional = |value: &Option<api::ShaderTableRegion<Dx12>>| -> Result<_> {
        value
            .as_ref()
            .map(|r| region(r, false))
            .transpose()
            .map(Option::unwrap_or_default)
    };
    Ok(D3D12_DISPATCH_RAYS_DESC {
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
    })
}

pub(super) fn dispatch(
    encoder: &mut Dx12Encoder,
    table: &api::ShaderTable<Dx12>,
    dims: [u32; 3],
) -> Result<()> {
    let native = prepare(table, dims)?;
    let list = encoder
        .list
        .cast::<ID3D12GraphicsCommandList4>()
        .map_err(super::super::native_error)?;
    for region in std::iter::once(&table.raygen)
        .chain(table.miss.iter())
        .chain(table.hit.iter())
        .chain(table.callable.iter())
    {
        // SAFETY: Core checked the device and the clone retains native storage.
        encoder
            .buffers
            .push(unsafe { region.buffer.native_ref() }.clone());
    }
    // SAFETY: all table sections are validated and retained by the encoder;
    // the native command list copies the initialized dispatch descriptor.
    unsafe { list.DispatchRays(&native) };
    Ok(())
}
