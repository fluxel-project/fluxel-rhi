//! DX12 descriptor-table ABI.
//!
//! A pipeline-layout group ordinal is its HLSL/DXIL register space; a logical
//! `binding` is the register number in that space.  This is the same mapping
//! used by the former DX12 backend.  The first implementation deliberately
//! exposes only static uniform-buffer bindings: it creates real shader-visible
//! CBV tables and refuses every resource kind it cannot lower truthfully.

use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CONSTANT_BUFFER_VIEW_DESC, D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_DESC,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, ID3D12DescriptorHeap, ID3D12Device,
};

use super::{Dx12, Dx12BindGroup, Dx12BindGroupLayout, Dx12Buffer, Dx12Device};

#[path = "descriptor_array.rs"]
mod descriptor_array;

const VIEW_CAPACITY: u32 = 65_536;
const SAMPLER_CAPACITY: u32 = 2_048;
const CBV_ALIGNMENT: u64 = 256;

/// The two device-wide shader-visible heaps required by D3D12.  A command list
/// can bind only one heap of each type, so groups reserve subranges here rather
/// than allocating private shader-visible heaps.
#[derive(Clone)]
pub(crate) struct DescriptorHeaps {
    views: Arc<Heap>,
    samplers: Arc<Heap>,
}

struct Heap {
    native: ID3D12DescriptorHeap,
    increment: u32,
    capacity: u32,
    free: Mutex<Vec<Range<u32>>>,
}

impl Heap {
    fn allocate(&self, count: u32) -> Result<u32> {
        if count == 0 {
            return Ok(0);
        }
        let mut free = self.free.lock().map_err(|_| Error::DeviceLost)?;
        let index = free
            .iter()
            .position(|range| range.end - range.start >= count)
            .ok_or(Error::Unsupported("DX12 descriptor heap capacity"))?;
        let start = free[index].start;
        free[index].start += count;
        if free[index].start == free[index].end {
            free.remove(index);
        }
        Ok(start)
    }

    fn release(&self, start: u32, count: u32) {
        if count == 0 {
            return;
        }
        let Some(end) = start.checked_add(count) else {
            return;
        };
        if end > self.capacity {
            return;
        }
        let Ok(mut free) = self.free.lock() else {
            return;
        };
        let index = free.partition_point(|range| range.start < start);
        free.insert(index, start..end);
        let mut merged: Vec<Range<u32>> = Vec::with_capacity(free.len());
        for range in free.drain(..) {
            if let Some(previous) = merged.last_mut() {
                if range.start <= previous.end {
                    previous.end = previous.end.max(range.end);
                    continue;
                }
            }
            merged.push(range);
        }
        *free = merged;
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let mut handle = unsafe { self.native.GetCPUDescriptorHandleForHeapStart() };
        handle.ptr += usize::try_from(index.saturating_mul(self.increment)).unwrap_or(0);
        handle
    }
}

/// Creates bounded, shader-visible heaps for the lifetime of one DX12 device.
pub(crate) fn create_heaps(device: &ID3D12Device) -> Result<DescriptorHeaps> {
    fn create(
        device: &ID3D12Device,
        ty: windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE,
        capacity: u32,
    ) -> Result<Arc<Heap>> {
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: ty,
            NumDescriptors: capacity,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        let native = unsafe { device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&desc) }
            .map_err(native_error)?;
        let increment = unsafe { device.GetDescriptorHandleIncrementSize(ty) };
        Ok(Arc::new(Heap {
            native,
            increment,
            capacity,
            free: Mutex::new(vec![0..capacity]),
        }))
    }
    Ok(DescriptorHeaps {
        views: create(
            device,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            VIEW_CAPACITY,
        )?,
        samplers: create(device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, SAMPLER_CAPACITY)?,
    })
}

impl DescriptorHeaps {
    pub(crate) fn view_heap(&self) -> &ID3D12DescriptorHeap {
        &self.views.native
    }
    pub(crate) fn sampler_heap(&self) -> &ID3D12DescriptorHeap {
        &self.samplers.native
    }
}

/// One logical layout entry after validation. `binding` becomes `b<binding>`;
/// `space` is assigned from the pipeline-layout group position.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DescriptorClass {
    Cbv,
    Srv,
    Uav,
    Sampler,
}

#[derive(Clone)]
pub(crate) struct BindingSlot {
    pub(crate) binding: u32,
    pub(crate) min_size: u64,
    pub(crate) class: DescriptorClass,
    pub(crate) dynamic_offset: bool,
    /// Descriptors reserved in the table for this logical binding.
    pub(crate) descriptor_count: u32,
    /// The root range is unbounded. Each group may bind up to 32 populated
    /// descriptors for this slot.
    pub(crate) runtime_sized: bool,
    /// Required dimension for sampled texture SRVs.
    pub(crate) view_dimension: Option<api::TextureViewDimension>,
}

/// Buffer range captured by a group. Dynamic offsets are applied to `range`
/// only while a command is encoded; the group itself never mutates descriptors.
#[derive(Clone)]
pub(crate) struct UniformBinding {
    pub(crate) binding: u32,
    pub(crate) buffer: Dx12Buffer,
    pub(crate) range: Range<u64>,
    pub(crate) min_size: u64,
    pub(crate) dynamic_offset: bool,
}

pub(crate) struct BindGroupLayoutState {
    pub(crate) slots: Vec<BindingSlot>,
}

pub(crate) struct BindGroupState {
    pub(crate) layout: Arc<BindGroupLayoutState>,
    buffers: Vec<Dx12Buffer>,
    uniforms: Vec<UniformBinding>,
    _textures: Vec<super::Dx12TextureView>,
    _samplers: Vec<super::Dx12Sampler>,
    view_start: u32,
    view_count: u32,
    sampler_start: u32,
    sampler_count: u32,
    heaps: DescriptorHeaps,
}

impl Drop for BindGroupState {
    fn drop(&mut self) {
        self.heaps.views.release(self.view_start, self.view_count);
        self.heaps
            .samplers
            .release(self.sampler_start, self.sampler_count);
    }
}

impl BindGroupState {
    pub(crate) fn view_start(&self) -> u32 {
        self.view_start
    }
    pub(crate) fn view_gpu(
        &self,
    ) -> windows::Win32::Graphics::Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE {
        let mut handle = unsafe { self.heaps.views.native.GetGPUDescriptorHandleForHeapStart() };
        handle.ptr += u64::from(self.view_start.saturating_mul(self.heaps.views.increment));
        handle
    }
    pub(crate) fn heaps(&self) -> &DescriptorHeaps {
        &self.heaps
    }
    pub(crate) fn buffers(&self) -> &[Dx12Buffer] {
        &self.buffers
    }
    pub(crate) fn uniform(&self, binding: u32) -> Option<&UniformBinding> {
        self.uniforms
            .iter()
            .find(|uniform| uniform.binding == binding)
    }
    pub(crate) fn sampler_gpu(
        &self,
    ) -> windows::Win32::Graphics::Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE {
        let mut handle = unsafe {
            self.heaps
                .samplers
                .native
                .GetGPUDescriptorHandleForHeapStart()
        };
        handle.ptr += u64::from(
            self.sampler_start
                .saturating_mul(self.heaps.samplers.increment),
        );
        handle
    }
    pub(crate) fn has_views(&self) -> bool {
        self.view_count != 0
    }
    pub(crate) fn has_samplers(&self) -> bool {
        self.sampler_count != 0
    }
}

/// Validates the subset whose descriptor ABI is fully defined today.
pub(crate) fn create_bind_group_layout(
    _: &Dx12Device,
    desc: &api::BindGroupLayoutDesc,
) -> Result<Dx12BindGroupLayout> {
    let mut slots = Vec::with_capacity(desc.entries.len());
    let mut previous = None;
    for entry in &desc.entries {
        if previous.is_some_and(|value| value >= entry.binding) {
            return Err(Error::InvalidState);
        }
        previous = Some(entry.binding);
        if entry.partially_bound {
            return Err(Error::Unsupported("DX12 partially-bound descriptor arrays"));
        }
        let descriptor_count = descriptor_array::count(entry.count)?;
        match entry.ty {
            api::BindingType::UniformBuffer {
                dynamic_offset,
                min_size,
            } if min_size > 0 && !entry.nonuniform_indexing => {
                slots.push(BindingSlot {
                    binding: entry.binding,
                    min_size,
                    class: DescriptorClass::Cbv,
                    dynamic_offset,
                    descriptor_count: one_descriptor(entry.count)?,
                    runtime_sized: false,
                    view_dimension: None,
                });
            }
            api::BindingType::SampledTexture {
                sample: api::TextureSampleType::Float,
                dimension:
                    dimension @ (api::TextureViewDimension::D2 | api::TextureViewDimension::D2Array),
                multisampled: false,
            } => {
                slots.push(BindingSlot {
                    binding: entry.binding,
                    min_size: 0,
                    class: DescriptorClass::Srv,
                    dynamic_offset: false,
                    descriptor_count,
                    runtime_sized: matches!(entry.count, api::BindingCount::RuntimeSized),
                    view_dimension: Some(dimension),
                });
            }
            api::BindingType::StorageTexture {
                access:
                    api::StorageAccess::Read | api::StorageAccess::Write | api::StorageAccess::ReadWrite,
                ..
            } => {
                if !matches!(entry.count, api::BindingCount::One) || entry.nonuniform_indexing {
                    return Err(Error::Unsupported("DX12 storage texture descriptor arrays"));
                }
                slots.push(BindingSlot {
                    binding: entry.binding,
                    min_size: 0,
                    class: DescriptorClass::Uav,
                    dynamic_offset: false,
                    descriptor_count: 1,
                    runtime_sized: false,
                    view_dimension: None,
                });
            }
            api::BindingType::Sampler {
                filtering: _,
                comparison: false,
            } => {
                if !matches!(entry.count, api::BindingCount::One) || entry.nonuniform_indexing {
                    return Err(Error::Unsupported("DX12 sampler descriptor arrays"));
                }
                slots.push(BindingSlot {
                    binding: entry.binding,
                    min_size: 0,
                    class: DescriptorClass::Sampler,
                    dynamic_offset: false,
                    descriptor_count: 1,
                    runtime_sized: false,
                    view_dimension: None,
                });
            }
            _ => return Err(Error::Unsupported("DX12 bind-group binding type")),
        }
    }
    Ok(Dx12BindGroupLayout(Arc::new(BindGroupLayoutState {
        slots,
    })))
}

fn one_descriptor(count: api::BindingCount) -> Result<u32> {
    if matches!(count, api::BindingCount::One) {
        Ok(1)
    } else {
        Err(Error::Unsupported("DX12 buffer descriptor arrays"))
    }
}

/// Writes one CBV descriptor per layout entry into a reserved device heap range.
pub(crate) fn create_bind_group(
    device: &Dx12Device,
    desc: &api::BindGroupDesc<'_, Dx12>,
) -> Result<Dx12BindGroup> {
    let layout = unsafe { desc.layout.native_ref() }.0.clone();
    if desc.entries.len() != layout.slots.len() {
        return Err(Error::InvalidState);
    }
    let view_count =
        layout
            .slots
            .iter()
            .zip(desc.entries)
            .try_fold(0u32, |count, (slot, entry)| {
                if entry.binding != slot.binding {
                    return Err(Error::InvalidState);
                }
                if slot.class == DescriptorClass::Sampler || slot.dynamic_offset {
                    return Ok(count);
                }
                count
                    .checked_add(resource_descriptor_count(slot, &entry.resource)?)
                    .ok_or(Error::InvalidRange)
            })?;
    let sampler_count = layout
        .slots
        .iter()
        .filter(|slot| slot.class == DescriptorClass::Sampler)
        .map(|slot| slot.descriptor_count)
        .sum();
    let start = device.0.descriptors.views.allocate(view_count)?;
    let sampler_start = match device.0.descriptors.samplers.allocate(sampler_count) {
        Ok(value) => value,
        Err(error) => {
            device.0.descriptors.views.release(start, view_count);
            return Err(error);
        }
    };
    let mut buffers = Vec::with_capacity(layout.slots.len());
    let mut uniforms = Vec::new();
    let mut textures = Vec::new();
    let mut samplers = Vec::new();
    let mut view_index = 0u32;
    let mut sampler_index = 0u32;
    for (index, slot) in layout.slots.iter().enumerate() {
        let entry = &desc.entries[index];
        if entry.binding != slot.binding {
            device.0.descriptors.views.release(start, view_count);
            device
                .0
                .descriptors
                .samplers
                .release(sampler_start, sampler_count);
            return Err(Error::InvalidState);
        }
        let descriptor_count = resource_descriptor_count(slot, &entry.resource)?;
        if slot.class == DescriptorClass::Srv {
            validate_sampled_view_dimension(slot, &entry.resource)?;
        }
        match (slot.class, &entry.resource) {
            (DescriptorClass::Cbv, api::BindingResource::Buffer { buffer, range }) => {
                let native_buffer = unsafe { buffer.native_ref() };
                if native_buffer.usage.0 & api::BufferUsage::UNIFORM.0 == 0 {
                    return Err(Error::Unsupported(
                        "DX12 uniform binding requires BufferUsage::UNIFORM",
                    ));
                }
                if range.start > range.end
                    || range.end > native_buffer.size
                    || range.start % CBV_ALIGNMENT != 0
                {
                    return Err(Error::InvalidRange);
                }
                let bytes = range.end - range.start;
                if bytes < slot.min_size || bytes == 0 {
                    return Err(Error::InvalidRange);
                }
                if slot.dynamic_offset {
                    // Root CBVs are selected at bind time.  D3D12 still
                    // requires their GPU VA to be 256-byte aligned.
                    uniforms.push(UniformBinding {
                        binding: slot.binding,
                        buffer: native_buffer.clone(),
                        range: range.clone(),
                        min_size: slot.min_size,
                        dynamic_offset: true,
                    });
                    buffers.push(native_buffer.clone());
                    continue;
                }
                let native_bytes = bytes
                    .checked_add(CBV_ALIGNMENT - 1)
                    .ok_or(Error::InvalidRange)?
                    / CBV_ALIGNMENT
                    * CBV_ALIGNMENT;
                if native_bytes > native_buffer.size.saturating_sub(range.start)
                    || native_bytes > u64::from(u32::MAX)
                {
                    return Err(Error::InvalidRange);
                }
                let cbv = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                    BufferLocation: unsafe { native_buffer.resource.GetGPUVirtualAddress() }
                        .saturating_add(range.start),
                    SizeInBytes: native_bytes as u32,
                };
                unsafe {
                    device.0.native.CreateConstantBufferView(
                        Some(&cbv),
                        device.0.descriptors.views.cpu(start + view_index),
                    );
                }
                uniforms.push(UniformBinding {
                    binding: slot.binding,
                    buffer: native_buffer.clone(),
                    range: range.clone(),
                    min_size: slot.min_size,
                    dynamic_offset: false,
                });
                view_index += 1;
                buffers.push(native_buffer.clone());
            }
            (DescriptorClass::Srv, api::BindingResource::TextureView(view))
                if slot.descriptor_count == 1 =>
            {
                let native = unsafe { view.native_ref() };
                let source = native
                    .0
                    .srv
                    .ok_or(Error::Unsupported("DX12 sampled texture without SRV"))?;
                unsafe {
                    device.0.native.CopyDescriptorsSimple(
                        1,
                        device.0.descriptors.views.cpu(start + view_index),
                        source,
                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    );
                }
                view_index += 1;
                textures.push(native.clone());
            }
            (DescriptorClass::Srv, api::BindingResource::Array(values)) => {
                let retained = descriptor_array::write_sampled_array(
                    device,
                    start + view_index,
                    descriptor_count,
                    values,
                )?;
                view_index += descriptor_count;
                textures.extend(retained);
            }
            (DescriptorClass::Uav, api::BindingResource::TextureView(view)) => {
                let native = unsafe { view.native_ref() };
                let source = native
                    .0
                    .uav
                    .ok_or(Error::Unsupported("DX12 storage texture without UAV"))?;
                unsafe {
                    device.0.native.CopyDescriptorsSimple(
                        1,
                        device.0.descriptors.views.cpu(start + view_index),
                        source,
                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    );
                }
                view_index += 1;
                textures.push(native.clone());
            }
            (DescriptorClass::Sampler, api::BindingResource::Sampler(sampler)) => {
                let native = unsafe { sampler.native_ref() };
                unsafe {
                    device.0.native.CopyDescriptorsSimple(
                        1,
                        device
                            .0
                            .descriptors
                            .samplers
                            .cpu(sampler_start + sampler_index),
                        native.0.cpu,
                        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                    );
                }
                sampler_index += 1;
                samplers.push(native.clone());
            }
            _ => {
                return Err(Error::Unsupported(
                    "DX12 bind-group resource does not match layout",
                ));
            }
        }
    }
    Ok(Dx12BindGroup(Arc::new(BindGroupState {
        layout,
        buffers,
        uniforms,
        _textures: textures,
        _samplers: samplers,
        view_start: start,
        view_count,
        sampler_start,
        sampler_count,
        heaps: device.0.descriptors.clone(),
    })))
}

fn resource_descriptor_count(
    slot: &BindingSlot,
    resource: &api::BindingResource<'_, Dx12>,
) -> Result<u32> {
    if !slot.runtime_sized {
        return Ok(slot.descriptor_count);
    }
    let api::BindingResource::Array(values) = resource else {
        return Err(Error::InvalidState);
    };
    let count = u32::try_from(values.len()).map_err(|_| Error::InvalidRange)?;
    if count == 0 || count > descriptor_array::RUNTIME_SRV_CAPACITY {
        return Err(Error::Unsupported(
            "DX12 runtime sampled texture array length",
        ));
    }
    Ok(count)
}

fn validate_sampled_view_dimension(
    slot: &BindingSlot,
    resource: &api::BindingResource<'_, Dx12>,
) -> Result<()> {
    let expected = slot.view_dimension.ok_or(Error::InvalidState)?;
    match resource {
        api::BindingResource::TextureView(view) => validate_sampled_view(view, expected),
        api::BindingResource::Array(views) => {
            for view in views.iter().flatten() {
                let api::BindingResource::TextureView(view) = view else {
                    return Err(Error::Unsupported("DX12 sampled texture array resource"));
                };
                validate_sampled_view(view, expected)?;
            }
            Ok(())
        }
        _ => Err(Error::Unsupported("DX12 sampled texture binding resource")),
    }
}

fn validate_sampled_view(
    view: &api::TextureView<Dx12>,
    expected: api::TextureViewDimension,
) -> Result<()> {
    if unsafe { view.native_ref() }.0.desc.dimension != expected {
        return Err(Error::InvalidState);
    }
    Ok(())
}

fn native_error(error: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {error}"))
}
