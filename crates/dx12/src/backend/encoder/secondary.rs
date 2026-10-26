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
    pub(crate) segments: Vec<SecondarySegment>,
    pub(crate) allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) buffers: Vec<Dx12Buffer>,
    pub(crate) groups: Vec<Dx12BindGroup>,
    pub(crate) pipeline: Option<Dx12RasterPipeline>,
    pub(crate) pipelines: Vec<Dx12RasterPipeline>,
    pub(crate) dynamic_view_tables: Vec<std::sync::Arc<super::super::descriptor::DynamicViewTable>>,
    pub(crate) signature: api::SecondaryRasterDesc,
    pub(crate) viewport: Option<api::Viewport>,
    pub(crate) scissor: Option<api::Rect>,
}

#[derive(Clone)]
pub(crate) struct SecondarySegment {
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) viewport: Option<api::Viewport>,
    pub(crate) scissor: Option<api::Rect>,
}

/// Worker-local bundle recorder.  Bundles deliberately do not contain any
/// attachment state; D3D12 inherits that state from the executing direct list.
pub struct Dx12SecondaryEncoder {
    pub(crate) device: Dx12Device,
    pub(crate) segments: Vec<SecondarySegment>,
    pub(crate) has_draw: bool,
    pub(crate) allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) buffers: Vec<Dx12Buffer>,
    pub(crate) groups: Vec<Dx12BindGroup>,
    pub(crate) pipeline: Option<Dx12RasterPipeline>,
    pub(crate) pipelines: Vec<Dx12RasterPipeline>,
    pub(crate) dynamic_view_tables: Vec<std::sync::Arc<super::super::descriptor::DynamicViewTable>>,
    pub(crate) signature: api::SecondaryRasterDesc,
    pub(crate) viewport: Option<api::Viewport>,
    pub(crate) scissor: Option<api::Rect>,
    pub(crate) closed: bool,
    pub(crate) index_format: Option<api::IndexFormat>,
    pub(crate) bound_slots: Vec<Option<Dx12BindGroup>>,
}

impl Dx12SecondaryEncoder {
    pub(crate) fn new(
        device: &Dx12Device,
        signature: &api::SecondaryRasterDesc,
    ) -> Result<Self, Error> {
        validate_signature(signature)?;
        // SAFETY: the device owns both newly created bundle objects; allocator
        // stays inside this secondary encoder until its closed bundle is retained.
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
        super::heaps::install(device, &list);
        Ok(Self {
            device: device.clone(),
            segments: Vec::new(),
            has_draw: false,
            allocator,
            list,
            buffers: Vec::new(),
            groups: Vec::new(),
            pipeline: None,
            pipelines: Vec::new(),
            dynamic_view_tables: Vec::new(),
            signature: signature.clone(),
            viewport: None,
            scissor: None,
            closed: false,
            index_format: None,
            bound_slots: Vec::new(),
        })
    }

    fn split_after_draw(&mut self) -> Result<(), Error> {
        if !self.has_draw {
            return Ok(());
        }
        // Only one list may record against this allocator at a time.
        unsafe { self.list.Close() }.map_err(native_error)?;
        self.closed = true;
        let next = unsafe {
            self.device
                .0
                .native
                .CreateCommandList::<_, _, ID3D12GraphicsCommandList>(
                    0,
                    D3D12_COMMAND_LIST_TYPE_BUNDLE,
                    &self.allocator,
                    None,
                )
        }
        .map_err(native_error)?;
        super::heaps::install(&self.device, &next);
        self.segments.push(SecondarySegment {
            list: std::mem::replace(&mut self.list, next),
            viewport: self.viewport,
            scissor: self.scissor,
        });
        self.closed = false;
        self.has_draw = false;
        // Bundles do not inherit PSO or primitive topology. Root arguments and
        // buffer bindings propagate from the preceding bundle through the parent.
        if let Some(pipeline) = self.pipeline.clone() {
            set_pipeline(self, &pipeline)?;
        }
        Ok(())
    }

    fn pipeline(&self) -> Result<&Dx12RasterPipeline, Error> {
        self.pipeline.as_ref().ok_or(Error::InvalidState)
    }
}

fn validate_signature(desc: &api::SecondaryRasterDesc) -> Result<(), Error> {
    if desc.color_formats.len() > 8
        || desc
            .color_formats
            .iter()
            .flatten()
            .any(|&format| super::super::texture::color_texel_bytes(format).is_none())
        || desc
            .depth_stencil_format
            .is_some_and(|format| !super::super::texture::is_depth(format))
        || !matches!(desc.samples, 1 | 2 | 4 | 8 | 16 | 32)
        || desc
            .multiview_mask
            .is_some_and(|mask| mask == 0 || mask > 15)
    {
        return Err(Error::Unsupported("DX12 secondary raster inheritance"));
    }
    Ok(())
}

pub(crate) fn set_pipeline(
    encoder: &mut Dx12SecondaryEncoder,
    pipeline: &Dx12RasterPipeline,
) -> Result<(), Error> {
    if encoder.pipeline.as_ref().map_or(true, |old| {
        !std::sync::Arc::ptr_eq(&old.0.layout, &pipeline.0.layout)
    }) {
        encoder.bound_slots.clear();
    }
    if pipeline.0.color_formats != encoder.signature.color_formats
        || pipeline.0.depth_format != encoder.signature.depth_stencil_format
        || pipeline.0.samples != encoder.signature.samples
        || pipeline.0.multiview
            != encoder
                .signature
                .multiview_mask
                .map(|mask| 32 - mask.leading_zeros())
    {
        return Err(Error::InvalidState);
    }
    // Bundles start with view mask zero and do not inherit the caller's mask.
    // Set it while recording so view-instanced draws are not discarded.
    if let Some(mask) = encoder.signature.multiview_mask {
        let list: ID3D12GraphicsCommandList1 = encoder.list.cast().map_err(native_error)?;
        // SAFETY: the list is this encoder's open bundle and the mask was validated.
        unsafe { list.SetViewInstanceMask(mask) };
    }
    // SAFETY: the pipeline retains root/PSO COM objects and this bundle is
    // exclusively open on its recording thread.
    unsafe {
        encoder.list.SetGraphicsRootSignature(&pipeline.0.root);
        encoder.list.SetPipelineState(&pipeline.0.state);
        encoder.list.IASetPrimitiveTopology(pipeline.0.topology);
    }
    encoder.pipeline = Some(pipeline.clone());
    encoder.pipelines.push(pipeline.clone());
    Ok(())
}

pub(crate) fn validate_inheritance(
    inherited: Option<&api::SecondaryRasterDesc>,
    color_formats: &[Option<api::TextureFormat>],
    depth_format: Option<api::TextureFormat>,
    samples: u32,
    multiview_mask: Option<u32>,
) -> Result<(), Error> {
    let Some(inherited) = inherited else {
        return Ok(());
    };
    if inherited.color_formats != color_formats
        || inherited.depth_stencil_format != depth_format
        || inherited.samples != samples
        || inherited.multiview_mask != multiview_mask
    {
        return Err(Error::InvalidState);
    }
    validate_signature(inherited)
}

#[cfg(test)]
#[path = "../../tests/backend/encoder/secondary.rs"]
mod tests;

pub(crate) fn set_bind_group(
    encoder: &mut Dx12SecondaryEncoder,
    slot: u32,
    group: &Dx12BindGroup,
    offsets: &[u32],
) -> Result<(), Error> {
    let pipeline = encoder.pipeline()?;
    let expected = pipeline
        .0
        .layout
        .groups
        .get(slot as usize)
        .ok_or(Error::InvalidRange)?;
    if !std::sync::Arc::ptr_eq(expected, &group.0.layout) {
        return Err(Error::InvalidState);
    }
    let table = pipeline
        .0
        .layout
        .tables
        .get(slot as usize)
        .ok_or(Error::InvalidRange)?;
    let native = &group.0;
    if native.has_views() != table.views.is_some()
        || native.has_samplers() != table.samplers.is_some()
        || offsets.len() != table.dynamic_bindings.len()
    {
        return Err(Error::InvalidState);
    }
    let mut addresses = Vec::with_capacity(offsets.len());
    let mut storage_offsets = Vec::new();
    for (&(binding, class), &offset) in table.dynamic_bindings.iter().zip(offsets) {
        if class != super::super::descriptor::DescriptorClass::Cbv {
            storage_offsets.push((binding, offset));
            continue;
        }
        let root = table
            .dynamic_cbvs
            .iter()
            .find(|root| root.binding == binding)
            .ok_or(Error::InvalidState)?;
        let uniform = native
            .uniform(root.binding)
            .filter(|u| u.dynamic_offset)
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
        let end = uniform
            .range
            .end
            .checked_add(offset)
            .ok_or(Error::InvalidRange)?;
        if end > uniform.buffer.size {
            return Err(Error::InvalidRange);
        }
        // SAFETY: the group retains this same-device buffer.
        let address = unsafe { uniform.buffer.resource.GetGPUVirtualAddress() }
            .checked_add(start)
            .ok_or(Error::InvalidRange)?;
        if address % 256 != 0 {
            return Err(Error::InvalidRange);
        }
        addresses.push((root.root, address));
    }
    let dynamic_views = native.prepare_dynamic_views(&storage_offsets)?;
    // The bundle installed the device heaps when its list was created.
    unsafe {
        if let Some(root) = table.views {
            encoder.list.SetGraphicsRootDescriptorTable(
                root,
                dynamic_views
                    .as_ref()
                    .map_or_else(|| native.view_gpu(), |v| v.gpu()),
            );
        }
        if let Some(root) = table.samplers {
            encoder
                .list
                .SetGraphicsRootDescriptorTable(root, native.sampler_gpu());
        }
        for (root, address) in addresses {
            encoder
                .list
                .SetGraphicsRootConstantBufferView(root, address);
        }
    }
    encoder.groups.push(group.clone());
    if let Some(table) = dynamic_views {
        encoder.dynamic_view_tables.push(table);
    }
    let slot = slot as usize;
    if encoder.bound_slots.len() <= slot {
        encoder.bound_slots.resize_with(slot + 1, || None);
    }
    encoder.bound_slots[slot] = Some(group.clone());
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
        // SAFETY: retained same-device buffer has a stable GPU address and
        // offset/size have been checked against its allocation above.
        BufferLocation: unsafe { buffer.resource.GetGPUVirtualAddress() } + offset,
        SizeInBytes: size as u32,
        StrideInBytes: stride,
    };
    // SAFETY: fully initialized view refers to a retained buffer and is copied
    // synchronously into this exclusively recorded bundle.
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
    encoder.index_format = Some(format);
    let format = match format {
        api::IndexFormat::Uint16 => windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16_UINT,
        api::IndexFormat::Uint32 => windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
    };
    let view = D3D12_INDEX_BUFFER_VIEW {
        // SAFETY: same retained-resource/range invariant as vertex binding.
        BufferLocation: unsafe { buffer.resource.GetGPUVirtualAddress() } + offset,
        SizeInBytes: size as u32,
        Format: format,
    };
    // SAFETY: `view` is complete and names a checked range in the retained buffer.
    unsafe { encoder.list.IASetIndexBuffer(Some(&view)) };
    encoder.buffers.push(buffer.clone());
    Ok(())
}

pub(crate) fn set_viewport(
    encoder: &mut Dx12SecondaryEncoder,
    viewport: api::Viewport,
) -> Result<(), Error> {
    validate_viewport(viewport)?;
    encoder.split_after_draw()?;
    // Bundles cannot issue RSSetViewports.  Preserve the command here and
    // lower it on the executing direct list immediately before this bundle.
    encoder.viewport = Some(viewport);
    Ok(())
}

pub(crate) fn set_scissor(
    encoder: &mut Dx12SecondaryEncoder,
    rect: api::Rect,
) -> Result<(), Error> {
    validate_scissor(rect)?;
    encoder.split_after_draw()?;
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
    if bytes.is_empty() || offset % 4 != 0 || bytes.len() % 4 != 0 {
        return Err(Error::InvalidRange);
    }
    let roots = range.graphics_writes(stages, offset, end)?;
    let values: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("exact DWORD")))
        .collect();
    for root in roots {
        // SAFETY: the validated root and range belong to the bound signature; values are live during this call.
        unsafe {
            encoder.list.SetGraphicsRoot32BitConstants(
                root,
                values.len() as u32,
                values.as_ptr().cast(),
                offset / 4,
            );
        }
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
    super::raster::validate_raw_bindings(encoder.pipeline()?, &encoder.bound_slots)?;
    // SAFETY: an installed pipeline and checked ranges make this an append-only
    // command on the bundle owned exclusively by this encoder.
    unsafe {
        if let Some(root) = encoder.pipeline()?.0.vertex_index_root {
            let values = [vertices.start, instances.start, 0];
            encoder
                .list
                .SetGraphicsRoot32BitConstants(root, 3, values.as_ptr().cast(), 0);
        }
        encoder.list.DrawInstanced(
            vertices.end - vertices.start,
            instances.end - instances.start,
            vertices.start,
            instances.start,
        );
    }
    encoder.has_draw = true;
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
    if !super::raster::index_format_matches(
        encoder.pipeline()?.0.strip_index_format,
        encoder.index_format,
    ) {
        return Err(Error::InvalidState);
    }
    super::raster::validate_raw_bindings(encoder.pipeline()?, &encoder.bound_slots)?;
    // SAFETY: as above, validated indexed draw operands are copied into this
    // exclusively recorded bundle without borrowing caller memory.
    unsafe {
        if let Some(root) = encoder.pipeline()?.0.vertex_index_root {
            let values = [base_vertex as u32, instances.start, 0];
            encoder
                .list
                .SetGraphicsRoot32BitConstants(root, 3, values.as_ptr().cast(), 0);
        }
        encoder.list.DrawIndexedInstanced(
            indices.end - indices.start,
            instances.end - indices.start,
            indices.start,
            base_vertex,
            instances.start,
        );
    }
    encoder.has_draw = true;
    Ok(())
}
