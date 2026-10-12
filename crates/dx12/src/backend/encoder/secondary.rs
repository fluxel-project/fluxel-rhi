//! D3D12 bundle lowering for inherited secondary raster command streams.
//!
//! A bundle records draw-only graphics commands on a worker thread.  It never
//! binds render targets: the direct command list owns attachments and executes
//! only bundles whose Core inheritance declaration matched that primary pass.

use super::*;

/// Closed D3D12 bundle plus every object referenced by its commands.
///
/// The direct encoder retains this after `ExecuteBundle`, so its allocator and
/// descriptor-backed objects survive until the parent work is retired.
#[derive(Clone)]
pub struct Dx12SecondaryWork {
    pub(crate) allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) buffers: Vec<Dx12Buffer>,
    pub(crate) groups: Vec<Dx12BindGroup>,
    pub(crate) pipeline: Option<Dx12RasterPipeline>,
    pub(crate) signature: api::SecondaryRasterDesc,
    pub(crate) viewport: Option<api::Viewport>,
    pub(crate) scissor: Option<api::Rect>,
}

/// Worker-local bundle recorder.  Bundles deliberately do not contain any
/// attachment state; D3D12 inherits that state from the executing direct list.
pub struct Dx12SecondaryEncoder {
    pub(crate) allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) buffers: Vec<Dx12Buffer>,
    pub(crate) groups: Vec<Dx12BindGroup>,
    pub(crate) pipeline: Option<Dx12RasterPipeline>,
    pub(crate) signature: api::SecondaryRasterDesc,
    pub(crate) viewport: Option<api::Viewport>,
    pub(crate) scissor: Option<api::Rect>,
    pub(crate) closed: bool,
}

impl Dx12SecondaryEncoder {
    pub(crate) fn new(
        device: &Dx12Device,
        signature: &api::SecondaryRasterDesc,
    ) -> Result<Self, Error> {
        validate_signature(signature)?;
        let (allocator, list) = unsafe {
            let allocator = device
                .0
                .native
                .CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_BUNDLE)
                .map_err(native_error)?;
            let list = device
                .0
                .native
                .CreateCommandList::<_, _, ID3D12GraphicsCommandList>(
                    0,
                    D3D12_COMMAND_LIST_TYPE_BUNDLE,
                    &allocator,
                    None,
                )
                .map_err(native_error)?;
            (allocator, list)
        };
        Ok(Self {
            allocator,
            list,
            buffers: Vec::new(),
            groups: Vec::new(),
            pipeline: None,
            signature: signature.clone(),
            viewport: None,
            scissor: None,
            closed: false,
        })
    }

    fn pipeline(&self) -> Result<&Dx12RasterPipeline, Error> {
        self.pipeline.as_ref().ok_or(Error::InvalidState)
    }
}

fn validate_signature(desc: &api::SecondaryRasterDesc) -> Result<(), Error> {
    if desc.color_formats.len() != 1
        || !matches!(
            desc.color_formats.first(),
            Some(Some(
                api::TextureFormat::Rgba8Unorm | api::TextureFormat::Bgra8Unorm
            ))
        )
        || desc.depth_stencil_format.is_some()
        || desc.samples != 1
        || desc.multiview_mask.is_some()
    {
        return Err(Error::Unsupported("DX12 secondary raster inheritance"));
    }
    Ok(())
}

pub(crate) fn set_pipeline(
    encoder: &mut Dx12SecondaryEncoder,
    pipeline: &Dx12RasterPipeline,
) -> Result<(), Error> {
    let color = encoder
        .signature
        .color_formats
        .first()
        .and_then(|format| *format)
        .ok_or(Error::InvalidState)?;
    if pipeline.0.color_format != color || pipeline.0.samples != encoder.signature.samples {
        return Err(Error::InvalidState);
    }
    unsafe {
        encoder.list.SetGraphicsRootSignature(&pipeline.0.root);
        encoder.list.SetPipelineState(&pipeline.0.state);
        encoder
            .list
            .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    }
    encoder.pipeline = Some(pipeline.clone());
    Ok(())
}

pub(crate) fn set_bind_group(
    encoder: &mut Dx12SecondaryEncoder,
    slot: u32,
    group: &Dx12BindGroup,
    offsets: &[u32],
) -> Result<(), Error> {
    let pipeline = encoder.pipeline()?;
    let table = pipeline
        .0
        .layout
        .tables
        .get(slot as usize)
        .ok_or(Error::InvalidRange)?;
    let native = &group.0;
    if native.has_views() != table.views.is_some()
        || native.has_samplers() != table.samplers.is_some()
        || offsets.len() != table.dynamic_cbvs.len()
    {
        return Err(Error::InvalidState);
    }
    // Descriptor heaps are state owned by the direct command list.  A D3D12
    // bundle may select root tables but cannot switch the active heap; the
    // primary installs this bundle's device-wide heaps immediately before
    // ExecuteBundle.
    unsafe {
        if let Some(root) = table.views {
            encoder
                .list
                .SetGraphicsRootDescriptorTable(root, native.view_gpu());
        }
        if let Some(root) = table.samplers {
            encoder
                .list
                .SetGraphicsRootDescriptorTable(root, native.sampler_gpu());
        }
        for (root, &offset) in table.dynamic_cbvs.iter().zip(offsets) {
            let uniform = native
                .uniform(root.binding)
                .filter(|uniform| uniform.dynamic_offset)
                .ok_or(Error::InvalidState)?;
            let offset = u64::from(offset);
            if offset % 256 != 0 {
                return Err(Error::InvalidRange);
            }
            let start = uniform
                .range
                .start
                .checked_add(offset)
                .ok_or(Error::InvalidRange)?;
            let end = start
                .checked_add(root.min_size)
                .ok_or(Error::InvalidRange)?;
            if end > uniform.buffer.size {
                return Err(Error::InvalidRange);
            }
            let address = uniform
                .buffer
                .resource
                .GetGPUVirtualAddress()
                .checked_add(start)
                .ok_or(Error::InvalidRange)?;
            if address % 256 != 0 {
                return Err(Error::InvalidRange);
            }
            encoder
                .list
                .SetGraphicsRootConstantBufferView(root.root, address);
        }
    }
    encoder.groups.push(group.clone());
    Ok(())
}

pub(crate) fn set_vertex_buffer(
    encoder: &mut Dx12SecondaryEncoder,
    slot: u32,
    buffer: &Dx12Buffer,
    offset: u64,
    size: u64,
) -> Result<(), Error> {
    let stride = *encoder
        .pipeline()?
        .0
        .vertex_strides
        .get(slot as usize)
        .ok_or(Error::InvalidRange)?;
    if offset.checked_add(size).ok_or(Error::InvalidRange)? > buffer.size
        || size > u64::from(u32::MAX)
    {
        return Err(Error::InvalidRange);
    }
    let view = D3D12_VERTEX_BUFFER_VIEW {
        BufferLocation: unsafe { buffer.resource.GetGPUVirtualAddress() } + offset,
        SizeInBytes: size as u32,
        StrideInBytes: stride,
    };
    unsafe {
        encoder
            .list
            .IASetVertexBuffers(slot, Some(std::slice::from_ref(&view)));
    }
    encoder.buffers.push(buffer.clone());
    Ok(())
}

pub(crate) fn set_index_buffer(
    encoder: &mut Dx12SecondaryEncoder,
    buffer: &Dx12Buffer,
    offset: u64,
    size: u64,
    format: api::IndexFormat,
) -> Result<(), Error> {
    if offset.checked_add(size).ok_or(Error::InvalidRange)? > buffer.size
        || size > u64::from(u32::MAX)
    {
        return Err(Error::InvalidRange);
    }
    let format = match format {
        api::IndexFormat::Uint16 => windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16_UINT,
        api::IndexFormat::Uint32 => windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
    };
    let view = D3D12_INDEX_BUFFER_VIEW {
        BufferLocation: unsafe { buffer.resource.GetGPUVirtualAddress() } + offset,
        SizeInBytes: size as u32,
        Format: format,
    };
    unsafe { encoder.list.IASetIndexBuffer(Some(&view)) };
    encoder.buffers.push(buffer.clone());
    Ok(())
}

pub(crate) fn set_viewport(
    encoder: &mut Dx12SecondaryEncoder,
    viewport: api::Viewport,
) -> Result<(), Error> {
    if viewport.width <= 0.0
        || viewport.height <= 0.0
        || viewport.min_depth > viewport.max_depth
        || !viewport.x.is_finite()
        || !viewport.y.is_finite()
        || !viewport.width.is_finite()
        || !viewport.height.is_finite()
        || !viewport.min_depth.is_finite()
        || !viewport.max_depth.is_finite()
    {
        return Err(Error::InvalidRange);
    }
    // Bundles cannot issue RSSetViewports.  Preserve the command here and
    // lower it on the executing direct list immediately before this bundle.
    encoder.viewport = Some(viewport);
    Ok(())
}

pub(crate) fn set_scissor(
    encoder: &mut Dx12SecondaryEncoder,
    rect: api::Rect,
) -> Result<(), Error> {
    if rect.width == 0 || rect.height == 0 {
        return Err(Error::InvalidRange);
    }
    let right = rect.x.checked_add(rect.width).ok_or(Error::InvalidRange)?;
    let bottom = rect.y.checked_add(rect.height).ok_or(Error::InvalidRange)?;
    if right > i32::MAX as u32 || bottom > i32::MAX as u32 {
        return Err(Error::InvalidRange);
    }
    // Bundles cannot issue RSSetScissorRects.  See `set_viewport`.
    encoder.scissor = Some(rect);
    Ok(())
}

pub(crate) fn set_push_constants(
    encoder: &mut Dx12SecondaryEncoder,
    stages: api::StageMask,
    offset: u32,
    bytes: &[u8],
) -> Result<(), Error> {
    let range = encoder
        .pipeline()?
        .0
        .layout
        .push_constants
        .as_ref()
        .ok_or(Error::Unsupported("DX12 pipeline has no push constants"))?;
    let end = offset
        .checked_add(u32::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?)
        .ok_or(Error::InvalidRange)?;
    if stages != range.stages
        || bytes.is_empty()
        || offset < range.range.start
        || end > range.range.end
        || offset % 4 != 0
        || bytes.len() % 4 != 0
    {
        return Err(Error::InvalidRange);
    }
    let values: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("exact DWORD")))
        .collect();
    unsafe {
        encoder.list.SetGraphicsRoot32BitConstants(
            range.root,
            values.len() as u32,
            values.as_ptr().cast(),
            (offset - range.range.start) / 4,
        );
    }
    Ok(())
}

pub(crate) fn draw(
    encoder: &mut Dx12SecondaryEncoder,
    vertices: std::ops::Range<u32>,
    instances: std::ops::Range<u32>,
) -> Result<(), Error> {
    if vertices.start > vertices.end || instances.start > instances.end {
        return Err(Error::InvalidRange);
    }
    encoder.pipeline()?;
    unsafe {
        encoder.list.DrawInstanced(
            vertices.end - vertices.start,
            instances.end - instances.start,
            vertices.start,
            instances.start,
        );
    }
    Ok(())
}

pub(crate) fn draw_indexed(
    encoder: &mut Dx12SecondaryEncoder,
    indices: std::ops::Range<u32>,
    base_vertex: i32,
    instances: std::ops::Range<u32>,
) -> Result<(), Error> {
    if indices.start > indices.end || instances.start > instances.end {
        return Err(Error::InvalidRange);
    }
    encoder.pipeline()?;
    unsafe {
        encoder.list.DrawIndexedInstanced(
            indices.end - indices.start,
            instances.end - indices.start,
            indices.start,
            base_vertex,
            instances.start,
        );
    }
    Ok(())
}
