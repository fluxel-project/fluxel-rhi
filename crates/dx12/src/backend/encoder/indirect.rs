//! DX12 lowering for the public indexed/direct indirect-draw operations.
use super::*;

fn execute(
    encoder: &mut Dx12Encoder,
    arguments: &Dx12Buffer,
    offset: u64,
    count: u32,
    stride: u32,
    indexed: bool,
) -> Result<(), Error> {
    if !arguments.usage.contains(api::BufferUsage::INDIRECT)
        || offset % 4 != 0
        || (count != 0 && (stride < if indexed { 20 } else { 16 } || stride % 4 != 0))
    {
        return Err(Error::InvalidRange);
    }
    let bytes = u64::from(count)
        .checked_mul(u64::from(stride))
        .ok_or(Error::InvalidRange)?;
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end > arguments.size)
    {
        return Err(Error::InvalidRange);
    }
    let signature = signature(arguments, indexed, stride)?;
    barrier(
        &encoder.list,
        &arguments.resource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    );
    unsafe {
        encoder.list.ExecuteIndirect(
            &signature,
            count,
            &arguments.resource,
            offset,
            None::<&ID3D12Resource>,
            0,
        );
    }
    barrier(
        &encoder.list,
        &arguments.resource,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_COMMON,
    );
    encoder.buffers.push(arguments.clone());
    Ok(())
}

/// Retrieves a device-lifetime native signature.  It is keyed solely by the
/// public argument layout and stored with the buffer's owning device, so it
/// cannot cross devices and does not retain recorded command lists.
fn signature(
    arguments: &Dx12Buffer,
    indexed: bool,
    stride: u32,
) -> Result<ID3D12CommandSignature, Error> {
    let mut cache = arguments
        .indirect_signatures
        .lock()
        .map_err(|_| Error::Backend("DX12 indirect signature cache poisoned".into()))?;
    if let Some(signature) = cache.get(&(indexed, stride)) {
        return Ok(signature.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: if indexed {
            D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED
        } else {
            D3D12_INDIRECT_ARGUMENT_TYPE_DRAW
        },
        Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0::default(),
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut device = None;
    unsafe { arguments.resource.GetDevice::<ID3D12Device>(&mut device) }.map_err(native_error)?;
    let device =
        device.ok_or_else(|| Error::Backend("indirect buffer has no DX12 device".into()))?;
    let mut output: Option<ID3D12CommandSignature> = None;
    unsafe { device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut output) }
        .map_err(native_error)?;
    let signature = output
        .ok_or_else(|| Error::Backend("CreateCommandSignature returned no signature".into()))?;
    cache.insert((indexed, stride), signature.clone());
    Ok(signature)
}
pub(crate) fn draw(
    encoder: &mut Dx12Encoder,
    args: &Dx12Buffer,
    offset: u64,
    indexed: bool,
) -> Result<(), Error> {
    execute(
        encoder,
        args,
        offset,
        1,
        if indexed { 20 } else { 16 },
        indexed,
    )
}
pub(crate) fn multi(
    encoder: &mut Dx12Encoder,
    args: &Dx12Buffer,
    offset: u64,
    count: u32,
    stride: u32,
    indexed: bool,
) -> Result<(), Error> {
    execute(encoder, args, offset, count, stride, indexed)
}
