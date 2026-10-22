//! Creates bind groups and owns their descriptor reservations.

use super::*;

/// Returns both reservations on every failed group creation path.
struct ReservationGuard {
    heaps: DescriptorHeaps,
    views: (u32, u32),
    samplers: (u32, u32),
    committed: bool,
}
impl Drop for ReservationGuard {
    fn drop(&mut self) {
        if !self.committed {
            self.heaps.views.release(self.views.0, self.views.1);
            self.heaps
                .samplers
                .release(self.samplers.0, self.samplers.1);
        }
    }
}

/// Writes descriptor tables and retains every resource referenced by them.
pub(crate) fn create_bind_group(
    device: &Dx12Device,
    desc: &api::BindGroupDesc<'_, Dx12>,
) -> Result<Dx12BindGroup> {
    // SAFETY: Core validated the same-device layout; cloning retains its
    // immutable native metadata independently of the descriptor borrow.
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
                if slot.class == DescriptorClass::Sampler
                    || (slot.dynamic_offset && slot.class == DescriptorClass::Cbv)
                {
                    return Ok(count);
                }
                count
                    .checked_add(write::resource_descriptor_count(slot, &entry.resource)?)
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
    let mut reservation = ReservationGuard {
        heaps: device.0.descriptors.clone(),
        views: (start, view_count),
        samplers: (sampler_start, sampler_count),
        committed: false,
    };
    let shadow_views = if layout
        .slots
        .iter()
        .any(|slot| slot.dynamic_offset && slot.class != DescriptorClass::Cbv)
    {
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: view_count,
            Flags: windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        // SAFETY: initialized heap descriptor and live device.
        Some(
            unsafe {
                device
                    .0
                    .native
                    .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&desc)
            }
            .map_err(native_error)?,
        )
    } else {
        None
    };
    let view_cpu = |index: u32| {
        if let Some(heap) = &shadow_views {
            // SAFETY: shadow heap is retained through all descriptor writes.
            let mut handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
            handle.ptr += index as usize * device.0.descriptors.views.increment as usize;
            handle
        } else {
            device.0.descriptors.views.cpu(start + index)
        }
    };
    let mut buffers = Vec::with_capacity(layout.slots.len());
    let mut uniforms = Vec::new();
    let mut raw_storage_bindings = Vec::new();
    let mut dynamic_storage = Vec::new();
    let mut textures = Vec::new();
    let mut accelerations = Vec::new();
    let mut samplers = Vec::new();
    let mut view_index = 0u32;
    let mut sampler_index = 0u32;
    for (index, slot) in layout.slots.iter().enumerate() {
        let entry = &desc.entries[index];
        if entry.binding != slot.binding {
            return Err(Error::InvalidState);
        }
        if slot.view_dimension.is_some() {
            write::validate_texture_view_dimension(device, slot, &entry.resource)?;
        }
        let resources: Vec<Option<&api::BindingResource<'_, Dx12>>> = match &entry.resource {
            api::BindingResource::Array(values) => {
                let mut result: Vec<_> = values.iter().map(Option::as_ref).collect();
                result.resize(slot.descriptor_count as usize, None);
                result
            }
            resource => vec![Some(resource)],
        };
        if !resources.is_empty()
            && resources.iter().all(|resource| {
                matches!(
                    resource,
                    Some(api::BindingResource::StorageBufferView {
                        view: api::StorageBufferView::Raw,
                        ..
                    })
                )
            })
        {
            raw_storage_bindings.push(slot.binding);
        }
        for resource in resources {
            let Some(resource) = resource else {
                write::write_hole(
                    device,
                    slot.class,
                    view_cpu(view_index),
                    sampler_start + sampler_index,
                );
                if slot.class == DescriptorClass::Sampler {
                    sampler_index += 1;
                } else {
                    view_index += 1;
                }
                continue;
            };
            match (slot.class, resource) {
                (DescriptorClass::Srv, api::BindingResource::Acceleration(acceleration))
                    if slot.view_dimension.is_none() =>
                {
                    // SAFETY: Core checked the device identity; the bind group
                    // retains the resource until all recorded uses have retired.
                    let native = unsafe { acceleration.native_ref() };
                    if !native.0.is_top_level {
                        return Err(Error::InvalidState);
                    }
                    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
                        ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            // SAFETY: the native acceleration resource is retained below.
                            RaytracingAccelerationStructure:
                                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV {
                                    Location: unsafe { native.0.resource.GetGPUVirtualAddress() },
                                },
                        },
                    };
                    // SAFETY: DXR acceleration SRVs require a null resource and
                    // an address in the descriptor; the heap slot is allocated here.
                    unsafe {
                        device.0.native.CreateShaderResourceView(
                            None::<&ID3D12Resource>,
                            Some(&desc),
                            view_cpu(view_index),
                        );
                    }
                    view_index += 1;
                    accelerations.push(native.clone());
                }
                (
                    class @ (DescriptorClass::Srv | DescriptorClass::Uav),
                    api::BindingResource::StorageBufferView {
                        buffer,
                        range,
                        view,
                    },
                ) => {
                    // SAFETY: Core validates device ownership; the clone below retains the buffer.
                    let native = unsafe { buffer.native_ref() };
                    if class == DescriptorClass::Uav && native.memory == api::MemoryClass::Upload {
                        return Err(Error::Unsupported(
                            "DX12 upload storage buffers are read-only",
                        ));
                    }
                    if !native.usage.contains(api::BufferUsage::STORAGE)
                        || range.end > native.size
                        || range.start >= range.end
                        || range.end - range.start < slot.min_size
                    {
                        return Err(Error::InvalidRange);
                    }
                    let (stride, format, raw) = match view {
                        api::StorageBufferView::Raw => (
                            4u32,
                            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
                            true,
                        ),
                        api::StorageBufferView::Structured { stride }
                            if *stride > 0 && *stride <= 2048 && *stride % 4 == 0 =>
                        {
                            (
                                *stride,
                                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
                                false,
                            )
                        }
                        _ => return Err(Error::InvalidRange),
                    };
                    if range.start % u64::from(stride) != 0
                        || (range.end - range.start) % u64::from(stride) != 0
                    {
                        return Err(Error::InvalidRange);
                    }
                    let first = range.start / u64::from(stride);
                    let count = u32::try_from((range.end - range.start) / u64::from(stride))
                        .map_err(|_| Error::InvalidRange)?;
                    let structure = if raw { 0 } else { stride };
                    // SAFETY: the descriptor range and resource usage were validated;
                    // the allocated heap slot and resource are retained by the group.
                    unsafe {
                        if class == DescriptorClass::Srv {
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
                            device.0.native.CreateShaderResourceView(
                                &native.resource,
                                Some(&desc),
                                view_cpu(view_index),
                            );
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
                            device.0.native.CreateUnorderedAccessView(
                                &native.resource,
                                None::<&ID3D12Resource>,
                                Some(&desc),
                                view_cpu(view_index),
                            );
                        }
                    }
                    if slot.dynamic_offset {
                        dynamic_storage.push(DynamicStorageBinding {
                            binding: slot.binding,
                            buffer: native.clone(),
                            range: range.clone(),
                            view: *view,
                            class,
                            table_index: view_index,
                        });
                    }
                    view_index += 1;
                    buffers.push(native.clone());
                }
                (DescriptorClass::Cbv, api::BindingResource::Buffer { buffer, range }) => {
                    // SAFETY: Core guarantees the binding buffer belongs to this
                    // device; clones below retain it until bind-group destruction.
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
                    if native_bytes > native_buffer.allocation_size.saturating_sub(range.start)
                        || native_bytes > u64::from(u32::MAX)
                    {
                        return Err(Error::InvalidRange);
                    }
                    let cbv = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                        // SAFETY: the retained buffer is a live same-device COM
                        // resource, and its virtual address is immutable.
                        BufferLocation: unsafe { native_buffer.resource.GetGPUVirtualAddress() }
                            .saturating_add(range.start),
                        SizeInBytes: native_bytes as u32,
                    };
                    // SAFETY: the CBV range/alignment was checked above and the
                    // destination handle belongs to this group's reserved heap slot.
                    unsafe {
                        device
                            .0
                            .native
                            .CreateConstantBufferView(Some(&cbv), view_cpu(view_index));
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
                    if slot.view_dimension.is_some() =>
                {
                    // SAFETY: this same-device view is retained below; only its
                    // already-created SRV source descriptor is borrowed here.
                    let native = unsafe { view.native_ref() };
                    let source = native
                        .0
                        .srv
                        .ok_or(Error::Unsupported("DX12 sampled texture without SRV"))?;
                    // SAFETY: both descriptor handles belong to live same-device
                    // heaps and CopyDescriptorsSimple copies exactly one descriptor.
                    unsafe {
                        device.0.native.CopyDescriptorsSimple(
                            1,
                            view_cpu(view_index),
                            source,
                            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        );
                    }
                    view_index += 1;
                    textures.push(native.clone());
                }
                (DescriptorClass::Uav, api::BindingResource::TextureView(view)) => {
                    // SAFETY: same device/view lifetime invariant as the SRV path.
                    let native = unsafe { view.native_ref() };
                    let source = native
                        .0
                        .uav
                        .ok_or(Error::Unsupported("DX12 storage texture without UAV"))?;
                    // SAFETY: source UAV and reserved destination heap slot are
                    // retained for this group's lifetime.
                    unsafe {
                        device.0.native.CopyDescriptorsSimple(
                            1,
                            view_cpu(view_index),
                            source,
                            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        );
                    }
                    view_index += 1;
                    textures.push(native.clone());
                }
                (DescriptorClass::Sampler, api::BindingResource::Sampler(sampler)) => {
                    // SAFETY: Core validated sampler device identity; the native
                    // sampler descriptor heap is retained by the cloned sampler.
                    let native = unsafe { sampler.native_ref() };
                    // SAFETY: source and destination sampler heap handles are live
                    // and each describes one descriptor in the same D3D12 device.
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
    }
    if let Some(shadow) = &shadow_views {
        // SAFETY: shadow is CPU-only, fully initialized, and both ranges cover view_count descriptors.
        unsafe {
            device.0.native.CopyDescriptorsSimple(
                view_count,
                device.0.descriptors.views.cpu(start),
                shadow.GetCPUDescriptorHandleForHeapStart(),
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            );
        }
    }
    reservation.committed = true;
    Ok(Dx12BindGroup(Arc::new(BindGroupState {
        layout,
        buffers,
        uniforms,
        raw_storage_bindings,
        dynamic_storage,
        device: device.0.native.clone(),
        shadow_views,
        _textures: textures,
        _accelerations: accelerations,
        _samplers: samplers,
        view_start: start,
        view_count,
        sampler_start,
        sampler_count,
        heaps: device.0.descriptors.clone(),
    })))
}
