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
    raster::validate_raw_bindings(
        encoder.pipeline.as_ref().ok_or(Error::InvalidState)?,
        &encoder.raster_bound_groups,
    )?;
    if count == 0 {
        return Ok(());
    }
    if encoder
        .pipeline
        .as_ref()
        .is_some_and(|p| p.0.vertex_index_root.is_some())
    {
        return execute_vertex_indices(encoder, arguments, offset, count, stride, indexed, None);
    }
    let signature = signature(arguments, indexed, stride)?;
    // SAFETY: the caller transitions arguments to INDIRECT_READ; the resource
    // and signature are retained by the encoder until completion.
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
    // SAFETY: arguments retains a live resource and `device` is a valid COM
    // output slot populated synchronously by its owning D3D12 device.
    unsafe { arguments.resource.GetDevice::<ID3D12Device>(&mut device) }.map_err(native_error)?;
    let device =
        device.ok_or_else(|| Error::Backend("indirect buffer has no DX12 device".into()))?;
    let mut output: Option<ID3D12CommandSignature> = None;
    // SAFETY: `argument`/`desc` remain live through the call and `output` is a
    // valid slot for an owned immutable signature cached by this device only.
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

/// Execute indirect draws limited by a GPU-written 32-bit count.
pub(crate) fn multi_count(
    encoder: &mut Dx12Encoder,
    args: &Dx12Buffer,
    offset: u64,
    count: &Dx12Buffer,
    count_offset: u64,
    max: u32,
    stride: u32,
    indexed: bool,
) -> Result<(), Error> {
    if encoder.pipeline.is_none() {
        return Err(Error::InvalidState);
    }
    if !args.usage.contains(api::BufferUsage::INDIRECT)
        || !count.usage.contains(api::BufferUsage::INDIRECT)
        || offset % 4 != 0
        || count_offset % 4 != 0
        || count_offset
            .checked_add(4)
            .is_none_or(|end| end > count.size)
        || offset
            .checked_add(u64::from(max) * u64::from(stride))
            .is_none_or(|end| end > args.size)
        || (max != 0 && (stride < if indexed { 20 } else { 16 } || stride % 4 != 0))
    {
        return Err(Error::InvalidRange);
    }
    if max == 0 {
        return Ok(());
    }
    raster::validate_raw_bindings(
        encoder.pipeline.as_ref().ok_or(Error::InvalidState)?,
        &encoder.raster_bound_groups,
    )?;
    if encoder
        .pipeline
        .as_ref()
        .is_some_and(|p| p.0.vertex_index_root.is_some())
    {
        return execute_vertex_indices(
            encoder,
            args,
            offset,
            max,
            stride,
            indexed,
            Some((count, count_offset)),
        );
    }
    let signature = signature(args, indexed, stride)?;
    // SAFETY: caller transitions both argument and count ranges to INDIRECT_READ;
    // checked resources and command signature remain retained until completion.
    unsafe {
        encoder.list.ExecuteIndirect(
            &signature,
            max,
            &args.resource,
            offset,
            Some(&count.resource),
            count_offset,
        );
    }
    encoder.buffers.push(args.clone());
    encoder.buffers.push(count.clone());
    Ok(())
}

fn execute_vertex_indices(
    encoder: &mut Dx12Encoder,
    args: &Dx12Buffer,
    offset: u64,
    count: u32,
    stride: u32,
    indexed: bool,
    count_buffer: Option<(&Dx12Buffer, u64)>,
) -> Result<(), Error> {
    let pipeline = encoder.pipeline.clone().ok_or(Error::InvalidState)?;
    let root_index = pipeline.0.vertex_index_root.ok_or(Error::InvalidState)?;
    let root = pipeline.0.root.clone();
    let record_size = if indexed { 20u32 } else { 16u32 };
    let expanded_stride = record_size + 8;
    let size = u64::from(expanded_stride)
        .checked_mul(u64::from(count))
        .ok_or(Error::InvalidRange)?;
    let scratch = <Dx12 as api::ResourceBackend>::create_buffer(
        &encoder.device,
        &api::BufferDesc {
            label: None,
            size,
            usage: api::BufferUsage::COPY_DST | api::BufferUsage::INDIRECT,
            memory: api::MemoryClass::Device,
        },
    )?;
    let signature = vertex_index_signature(&pipeline, &encoder.device, root_index, &root, indexed)?;
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
    for i in 0..count {
        let src = offset + u64::from(i) * u64::from(stride);
        let dst = u64::from(i) * u64::from(expanded_stride);
        // SAFETY: all source and scratch ranges were checked above; both buffers are retained.
        unsafe {
            // The two base-index fields are adjacent in both native argument
            // layouts, matching the two root constants at the scratch head.
            encoder.list.CopyBufferRegion(
                &scratch.resource,
                dst,
                &args.resource,
                src + if indexed { 12 } else { 8 },
                8,
            );
            encoder.list.CopyBufferRegion(
                &scratch.resource,
                dst + 8,
                &args.resource,
                src,
                u64::from(record_size),
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
    // SAFETY: scratch records and optional count are in INDIRECT_ARGUMENT and retained until completion.
    unsafe {
        encoder.list.ExecuteIndirect(
            &signature,
            count,
            &scratch.resource,
            0,
            count_buffer.map(|(buffer, _)| &buffer.resource),
            count_buffer.map_or(0, |(_, offset)| offset),
        );
    }
    encoder.buffers.extend([args.clone(), scratch]);
    if let Some((buffer, _)) = count_buffer {
        encoder.buffers.push(buffer.clone());
    }
    encoder.indirect_signatures.push(signature);
    Ok(())
}

/// Lazily creates the two command signatures whose ABI adds the WGSL
/// `vertex_index`/`instance_index` root constants.  The cache belongs to the
/// raster pipeline because D3D12 validates the exact root signature supplied
/// at creation; device-wide pointer-key caches would retain unrelated roots
/// and could pair a recycled identity with the wrong ABI.
fn vertex_index_signature(
    pipeline: &Dx12RasterPipeline,
    device: &Dx12Device,
    root_index: u32,
    root: &ID3D12RootSignature,
    indexed: bool,
) -> Result<ID3D12CommandSignature, Error> {
    let slot = usize::from(indexed);
    let mut signatures = pipeline
        .0
        .vertex_indirect_signatures
        .lock()
        .map_err(|_| Error::Backend("DX12 vertex indirect signature cache poisoned".into()))?;
    if let Some(signature) = &signatures[slot] {
        return Ok(signature.clone());
    }
    let arguments = [
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                    RootParameterIndex: root_index,
                    DestOffsetIn32BitValues: 0,
                    Num32BitValuesToSet: 2,
                },
            },
        },
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: if indexed {
                D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED
            } else {
                D3D12_INDIRECT_ARGUMENT_TYPE_DRAW
            },
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0::default(),
        },
    ];
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: if indexed { 28 } else { 24 },
        NumArgumentDescs: arguments.len() as u32,
        pArgumentDescs: arguments.as_ptr(),
        NodeMask: 0,
    };
    let mut output: Option<ID3D12CommandSignature> = None;
    // SAFETY: the descriptors and pipeline-owned matching root signature stay
    // live for the synchronous call; the immutable COM result is retained by
    // this pipeline and cloned into each encoder that executes it.
    unsafe {
        device
            .0
            .native
            .CreateCommandSignature(&desc, Some(root), &mut output)
    }
    .map_err(native_error)?;
    let signature = output.ok_or(Error::InvalidState)?;
    signatures[slot] = Some(signature.clone());
    Ok(signature)
}
