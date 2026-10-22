//! DX12 descriptor-table ABI.
//!
//! A pipeline-layout group ordinal is its HLSL/DXIL register space; a logical
//! `binding` is the register number in that space.  This is the same mapping
//! used by the former DX12 backend. It allocates shader-visible descriptors
//! for the supported buffer, texture, sampler and acceleration bindings, and
//! rejects resource forms that cannot be lowered faithfully.

use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_BUFFER_SRV, D3D12_BUFFER_SRV_FLAG_NONE, D3D12_BUFFER_SRV_FLAG_RAW, D3D12_BUFFER_UAV,
    D3D12_BUFFER_UAV_FLAG_NONE, D3D12_BUFFER_UAV_FLAG_RAW, D3D12_CONSTANT_BUFFER_VIEW_DESC,
    D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
    D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
    D3D12_GPU_DESCRIPTOR_HANDLE, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV,
    D3D12_SHADER_RESOURCE_VIEW_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC_0, D3D12_SRV_DIMENSION_BUFFER,
    D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE, D3D12_UAV_DIMENSION_BUFFER,
    D3D12_UNORDERED_ACCESS_VIEW_DESC, D3D12_UNORDERED_ACCESS_VIEW_DESC_0, ID3D12DescriptorHeap,
    ID3D12Device, ID3D12Resource,
};

use super::{Dx12, Dx12BindGroup, Dx12BindGroupLayout, Dx12Buffer, Dx12Device};

#[path = "descriptor_array.rs"]
mod descriptor_array;
#[path = "descriptor/group.rs"]
mod group;
#[path = "descriptor/layout.rs"]
mod layout;
#[path = "descriptor/write.rs"]
mod write;
pub(crate) use group::create_bind_group;
pub(crate) use layout::create_bind_group_layout;

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

    fn gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: allocation bounds the index and the retained heap is shader visible.
        let mut handle = unsafe { self.native.GetGPUDescriptorHandleForHeapStart() };
        handle.ptr += u64::from(index) * u64::from(self.increment);
        handle
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        // SAFETY: this non-empty heap is retained by `self`; descriptor handles
        // are stable for its lifetime and index arithmetic is bounded by allocate.
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
        // SAFETY: `desc` is fully initialized and D3D12 returns an owned COM
        // heap whose lifetime is held by the returned Arc<Heap>.
        let native = unsafe { device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&desc) }
            .map_err(native_error)?;
        // SAFETY: the device owns the heap type and this query only reads its
        // immutable descriptor increment size.
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

/// One view and sampler slot in the device-wide heaps, retained until GPU completion.
/// Callers must keep this value alive with their submitted command list.
pub(crate) struct TemporaryDescriptorPair {
    heaps: DescriptorHeaps,
    view: u32,
    sampler: u32,
}

impl DescriptorHeaps {
    pub(crate) fn temporary_pair(&self) -> Result<TemporaryDescriptorPair> {
        let view = self.views.allocate(1)?;
        let sampler = match self.samplers.allocate(1) {
            Ok(slot) => slot,
            Err(error) => {
                self.views.release(view, 1);
                return Err(error);
            }
        };
        Ok(TemporaryDescriptorPair {
            heaps: self.clone(),
            view,
            sampler,
        })
    }
}

impl TemporaryDescriptorPair {
    pub(crate) fn view_cpu(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        self.heaps.views.cpu(self.view)
    }
    pub(crate) fn sampler_cpu(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        self.heaps.samplers.cpu(self.sampler)
    }
    pub(crate) fn view_gpu(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        self.heaps.views.gpu(self.view)
    }
    pub(crate) fn sampler_gpu(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        self.heaps.samplers.gpu(self.sampler)
    }
}

impl Drop for TemporaryDescriptorPair {
    fn drop(&mut self) {
        self.heaps.views.release(self.view, 1);
        self.heaps.samplers.release(self.sampler, 1);
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
    pub(crate) binding_type: api::BindingType,
    pub(crate) binding: u32,
    pub(crate) min_size: u64,
    pub(crate) class: DescriptorClass,
    pub(crate) dynamic_offset: bool,
    /// Descriptors reserved in the table for this logical binding.
    pub(crate) descriptor_count: u32,
    /// The binding uses a runtime-sized public array with a fixed capacity.
    pub(crate) runtime_sized: bool,
    /// Required dimension for sampled and storage texture views.
    pub(crate) view_dimension: Option<api::TextureViewDimension>,
    pub(crate) texture_sample: Option<api::TextureSampleType>,
    pub(crate) texture_format: Option<api::TextureFormat>,
    pub(crate) multisampled: bool,
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

#[derive(Clone)]
struct DynamicStorageBinding {
    binding: u32,
    buffer: Dx12Buffer,
    range: Range<u64>,
    view: api::StorageBufferView,
    class: DescriptorClass,
    table_index: u32,
}

/// Owns the per-command descriptor snapshot until GPU completion.
pub(crate) struct DynamicViewTable {
    heaps: DescriptorHeaps,
    start: u32,
    count: u32,
}
impl Drop for DynamicViewTable {
    fn drop(&mut self) {
        self.heaps.views.release(self.start, self.count);
    }
}
impl DynamicViewTable {
    pub(crate) fn gpu(&self) -> windows::Win32::Graphics::Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: the owned allocation remains alive for this table.
        let mut handle = unsafe { self.heaps.views.native.GetGPUDescriptorHandleForHeapStart() };
        handle.ptr += u64::from(self.start) * u64::from(self.heaps.views.increment);
        handle
    }
}

pub(crate) struct BindGroupLayoutState {
    pub(crate) slots: Vec<BindingSlot>,
}

pub(crate) struct BindGroupState {
    pub(crate) layout: Arc<BindGroupLayoutState>,
    buffers: Vec<Dx12Buffer>,
    uniforms: Vec<UniformBinding>,
    raw_storage_bindings: Vec<u32>,
    dynamic_storage: Vec<DynamicStorageBinding>,
    device: ID3D12Device,
    shadow_views: Option<ID3D12DescriptorHeap>,
    _textures: Vec<super::Dx12TextureView>,
    _accelerations: Vec<super::Dx12Acceleration>,
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
    /// Create an immutable, bounded descriptor-table snapshot for dynamic storage.
    /// `offsets` contains one (binding, offset) pair per dynamic storage binding.
    pub(crate) fn prepare_dynamic_views(
        &self,
        offsets: &[(u32, u32)],
    ) -> Result<Option<Arc<DynamicViewTable>>> {
        if offsets.len() != self.dynamic_storage.len() {
            return Err(Error::InvalidRange);
        }
        if offsets.is_empty() {
            return Ok(None);
        }
        let mut descriptors = Vec::with_capacity(offsets.len());
        for (binding, offset) in offsets {
            let value = self
                .dynamic_storage
                .iter()
                .find(|v| v.binding == *binding)
                .ok_or(Error::InvalidState)?;
            if offsets.iter().filter(|(b, _)| b == binding).count() != 1 {
                return Err(Error::InvalidRange);
            }
            let stride = match value.view {
                api::StorageBufferView::Raw => 4,
                api::StorageBufferView::Structured { stride } => stride,
            };
            let start = value
                .range
                .start
                .checked_add(u64::from(*offset))
                .ok_or(Error::InvalidRange)?;
            let end = value
                .range
                .end
                .checked_add(u64::from(*offset))
                .ok_or(Error::InvalidRange)?;
            if start % u64::from(stride) != 0 || end > value.buffer.size {
                return Err(Error::InvalidRange);
            }
            descriptors.push((
                value,
                start / u64::from(stride),
                u32::try_from((end - start) / u64::from(stride))
                    .map_err(|_| Error::InvalidRange)?,
                stride,
            ));
        }
        let start = self.heaps.views.allocate(self.view_count)?;
        let table = Arc::new(DynamicViewTable {
            heaps: self.heaps.clone(),
            start,
            count: self.view_count,
        });
        // SAFETY: the CPU-only shadow contains initialized descriptors, and the destination is a live allocation.
        unsafe {
            self.device.CopyDescriptorsSimple(
                self.view_count,
                self.heaps.views.cpu(start),
                self.shadow_views
                    .as_ref()
                    .ok_or(Error::InvalidState)?
                    .GetCPUDescriptorHandleForHeapStart(),
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            );
        }
        for (value, first, count, stride) in descriptors {
            let raw = matches!(value.view, api::StorageBufferView::Raw);
            let format = if raw {
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS
            } else {
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN
            };
            let structure = if raw { 0 } else { stride };
            let cpu = self.heaps.views.cpu(start + value.table_index);
            // SAFETY: range, allocation, descriptor kind, and resource lifetime were validated.
            unsafe {
                if value.class == DescriptorClass::Srv {
                    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: format,
                        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_SRV {
                                FirstElement: first,
                                NumElements: count,
                                StructureByteStride: structure,
                                Flags: if raw {
                                    D3D12_BUFFER_SRV_FLAG_RAW
                                } else {
                                    D3D12_BUFFER_SRV_FLAG_NONE
                                },
                            },
                        },
                    };
                    self.device
                        .CreateShaderResourceView(&value.buffer.resource, Some(&desc), cpu);
                } else {
                    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: format,
                        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_UAV {
                                FirstElement: first,
                                NumElements: count,
                                StructureByteStride: structure,
                                CounterOffsetInBytes: 0,
                                Flags: if raw {
                                    D3D12_BUFFER_UAV_FLAG_RAW
                                } else {
                                    D3D12_BUFFER_UAV_FLAG_NONE
                                },
                            },
                        },
                    };
                    self.device.CreateUnorderedAccessView(
                        &value.buffer.resource,
                        None::<&ID3D12Resource>,
                        Some(&desc),
                        cpu,
                    );
                }
            }
        }
        Ok(Some(table))
    }
    pub(crate) fn view_start(&self) -> u32 {
        self.view_start
    }
    pub(crate) fn view_gpu(
        &self,
    ) -> windows::Win32::Graphics::Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: the shader-visible view heap is retained by this bind group;
        // its allocation range was checked when the group was created.
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
    /// Whether every descriptor slot is a bound raw storage buffer view.
    pub(crate) fn storage_views_are_raw(&self, binding: u32) -> bool {
        self.raw_storage_bindings.contains(&binding)
    }
    pub(crate) fn uniform(&self, binding: u32) -> Option<&UniformBinding> {
        self.uniforms
            .iter()
            .find(|uniform| uniform.binding == binding)
    }
    pub(crate) fn sampler_gpu(
        &self,
    ) -> windows::Win32::Graphics::Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: equivalent sampler heap ownership/allocation invariant as
        // `view_gpu`; the resulting GPU handle borrows no Rust memory.
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

fn native_error(error: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {error}"))
}
