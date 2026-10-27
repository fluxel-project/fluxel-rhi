//! D3D12 shared-resource and shared-fence import/export.
//!
//! External resource handles are NT handles consumed by this process. Resource
//! state and producer ownership remain an explicit unsafe bridge obligation.
//! A D3D12 fence is queue-scoped, so acquire/release uses the encoder's narrow
//! native-list segmentation path to place `Wait` and `Signal` at the exact
//! recorded Core boundary without a second scheduler or command replay path.

use super::{
    Dx12, Dx12Buffer, Dx12Device, Dx12Texture, encoder,
    texture::{self, TextureState},
};
use fluxel_rhi_core::api::{self, Error, Result, Support};
use fluxel_rhi_core::backend::{ExternalBackend, ResourceBackend};
use std::{
    os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle},
    sync::{Arc, atomic::AtomicBool},
};
use windows::{
    Win32::{
        Foundation::{GENERIC_ALL, HANDLE},
        Graphics::{Direct3D12::*, Dxgi::Common::DXGI_FORMAT_UNKNOWN},
    },
    core::PCWSTR,
};

/// Owned Windows NT handles supplied by or returned to a platform bridge.
///
/// `Resource` must name an `ID3D12Resource` created with D3D12 sharing
/// support. `Fence` names a shared `ID3D12Fence` at the supplied timeline
/// value. Handles are consumed by import and closed by Rust ownership.
pub enum Dx12ExternalHandle {
    Resource(OwnedHandle),
    Fence { handle: OwnedHandle, value: u64 },
}

/// A shared D3D12 fence value. Imported values keep only an opened COM fence;
/// release values additionally carry an owned NT handle for the bridge.
pub enum Dx12ExternalSync {
    Imported {
        fence: ID3D12Fence,
        value: u64,
    },
    Exported {
        fence: ID3D12Fence,
        value: u64,
        handle: OwnedHandle,
    },
}

impl Dx12ExternalSync {
    /// Returns an exportable bridge payload. Imported synchronization cannot
    /// be re-exported implicitly because that would change handle ownership.
    pub fn into_handle(self) -> Option<Dx12ExternalHandle> {
        match self {
            Self::Exported { handle, value, .. } => {
                Some(Dx12ExternalHandle::Fence { handle, value })
            }
            Self::Imported { .. } => None,
        }
    }

    fn fence_value(&self) -> (&ID3D12Fence, u64) {
        match self {
            Self::Imported { fence, value } | Self::Exported { fence, value, .. } => {
                (fence, *value)
            }
        }
    }
}

fn native_handle(handle: &OwnedHandle) -> HANDLE {
    HANDLE(handle.as_raw_handle().cast())
}

unsafe fn owned_handle(handle: HANDLE) -> Result<OwnedHandle> {
    if handle.0.is_null() || handle.0 as isize == -1 {
        return Err(Error::Backend(
            "DX12 CreateSharedHandle returned an invalid handle".into(),
        ));
    }
    // SAFETY: CreateSharedHandle transfers one owned NT handle to this process.
    Ok(unsafe { OwnedHandle::from_raw_handle(handle.0) })
}

fn imported_fence(
    device: &Dx12Device,
    handle: OwnedHandle,
    value: u64,
) -> Result<Dx12ExternalSync> {
    if value == u64::MAX {
        return Err(Error::InvalidRange);
    }
    let mut fence = None;
    // SAFETY: the unsafe Core import contract supplies a valid compatible NT
    // handle. OpenSharedHandle only borrows it during this call; `handle` then
    // closes it while the resulting COM fence remains independently retained.
    unsafe {
        device
            .0
            .native
            .OpenSharedHandle::<ID3D12Fence>(native_handle(&handle), &mut fence)
    }
    .map_err(super::native_error)?;
    Ok(Dx12ExternalSync::Imported {
        fence: fence.ok_or(Error::InvalidState)?,
        value,
    })
}

fn exported_fence(device: &Dx12Device) -> Result<Dx12ExternalSync> {
    // SAFETY: a fresh device-owned shared fence is valid for the returned
    // native handle and has no outstanding signals.
    let fence = unsafe {
        device
            .0
            .native
            .CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_SHARED)
    }
    .map_err(super::native_error)?;
    let handle = unsafe {
        device
            .0
            .native
            .CreateSharedHandle(&fence, None, GENERIC_ALL.0, PCWSTR::null())
    }
    .map_err(super::native_error)?;
    Ok(Dx12ExternalSync::Exported {
        fence,
        value: 1,
        // SAFETY: `handle` was freshly returned by CreateSharedHandle above.
        handle: unsafe { owned_handle(handle) }?,
    })
}

fn resource(device: &Dx12Device, handle: OwnedHandle) -> Result<ID3D12Resource> {
    let mut resource = None;
    // SAFETY: the unsafe Core import contract supplies an NT handle naming a
    // shareable resource compatible with this device. The COM reference outlives
    // the handle, which is closed after this call.
    unsafe {
        device
            .0
            .native
            .OpenSharedHandle::<ID3D12Resource>(native_handle(&handle), &mut resource)
    }
    .map_err(super::native_error)?;
    resource.ok_or(Error::InvalidState)
}

fn imported_memory_class(resource: &ID3D12Resource) -> Result<api::MemoryClass> {
    let mut properties = D3D12_HEAP_PROPERTIES::default();
    let mut flags = D3D12_HEAP_FLAG_NONE;
    // SAFETY: output pointers are valid for this synchronous native query.
    unsafe { resource.GetHeapProperties(Some(&mut properties), Some(&mut flags)) }
        .map_err(super::native_error)?;
    let _ = flags;
    match properties.Type {
        D3D12_HEAP_TYPE_DEFAULT => Ok(api::MemoryClass::Device),
        D3D12_HEAP_TYPE_UPLOAD => Ok(api::MemoryClass::Upload),
        D3D12_HEAP_TYPE_READBACK => Ok(api::MemoryClass::Readback),
        D3D12_HEAP_TYPE_CUSTOM => match properties.CPUPageProperty {
            D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE => Ok(api::MemoryClass::Device),
            D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE => Ok(api::MemoryClass::Upload),
            D3D12_CPU_PAGE_PROPERTY_WRITE_BACK => Ok(api::MemoryClass::Readback),
            _ => Err(Error::Unsupported("DX12 external custom heap CPU policy")),
        },
        _ => Err(Error::Unsupported("DX12 external custom heap resource")),
    }
}

fn import_buffer(
    device: &Dx12Device,
    handle: OwnedHandle,
    desc: &api::ExternalResourceDesc,
) -> Result<Dx12Buffer> {
    if desc.size == 0 || desc.offset.checked_add(desc.size).is_none() {
        return Err(Error::InvalidRange);
    }
    let resource = resource(device, handle)?;
    // SAFETY: the opened resource is live for this query.
    let native = unsafe { resource.GetDesc() };
    if native.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
        || native.Format != DXGI_FORMAT_UNKNOWN
        || native.Width < desc.size
    {
        return Err(Error::InvalidState);
    }
    let memory = imported_memory_class(&resource)?;
    let buffer_desc = api::BufferDesc {
        label: None,
        size: desc.size,
        usage: desc.usage,
        memory,
    };
    if let Support::Unsupported(reason) = Dx12::buffer_support(device, &buffer_desc) {
        return Err(Error::Unsupported(reason));
    }
    let allocation_size = if desc.usage.contains(api::BufferUsage::UNIFORM) {
        desc.size.checked_add(255).ok_or(Error::InvalidRange)? & !255
    } else {
        desc.size
    };
    if native.Width < allocation_size {
        return Err(Error::InvalidState);
    }
    let mapped = desc.usage.contains(api::BufferUsage::MAP_READ)
        || desc.usage.contains(api::BufferUsage::MAP_WRITE);
    if mapped
        && !matches!(
            (
                memory,
                desc.usage.contains(api::BufferUsage::MAP_READ),
                desc.usage.contains(api::BufferUsage::MAP_WRITE)
            ),
            (api::MemoryClass::Readback, true, false) | (api::MemoryClass::Upload, false, true)
        )
    {
        return Err(Error::InvalidState);
    }
    if (desc.usage.contains(api::BufferUsage::STORAGE)
        || desc.usage.contains(api::BufferUsage::ACCEL_SCRATCH))
        && !native
            .Flags
            .contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    {
        return Err(Error::InvalidState);
    }
    Ok(Dx12Buffer {
        resource,
        size: desc.size,
        allocation_size,
        mapped: Arc::new(AtomicBool::new(false)),
        usage: desc.usage,
        memory,
        indirect_signatures: device.0.indirect_signatures.clone(),
        backing_heap: None,
        default_allocation: None,
    })
}

fn resource_dimension(dimension: api::TextureDimension) -> D3D12_RESOURCE_DIMENSION {
    match dimension {
        api::TextureDimension::D1 => D3D12_RESOURCE_DIMENSION_TEXTURE1D,
        api::TextureDimension::D2 => D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        api::TextureDimension::D3 => D3D12_RESOURCE_DIMENSION_TEXTURE3D,
    }
}

fn import_texture(
    device: &Dx12Device,
    handle: OwnedHandle,
    desc: &api::TextureDesc,
) -> Result<Dx12Texture> {
    if let api::Support::Unsupported(reason) = texture::support(device, desc) {
        return Err(Error::Unsupported(reason));
    }
    let allocation = texture::allocation_desc(device, desc)?;
    let resource = resource(device, handle)?;
    // SAFETY: the opened resource is live for this descriptor query.
    let native = unsafe { resource.GetDesc() };
    let expected_depth = if desc.dimension == api::TextureDimension::D3 {
        desc.extent[2]
    } else {
        desc.array_layers
    };
    if native.Dimension != resource_dimension(desc.dimension)
        || native.Width != u64::from(desc.extent[0])
        || native.Height != desc.extent[1]
        || native.DepthOrArraySize
            != u16::try_from(expected_depth).map_err(|_| Error::InvalidRange)?
        || native.MipLevels != u16::try_from(desc.mip_levels).map_err(|_| Error::InvalidRange)?
        || native.SampleDesc.Count != desc.samples
        || native.SampleDesc.Quality != allocation.resource.SampleDesc.Quality
        || native.Format != allocation.resource.Format
    {
        return Err(Error::InvalidState);
    }
    if desc.usage.contains(api::TextureUsage::STORAGE_BINDING)
        && !native
            .Flags
            .contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    {
        return Err(Error::InvalidState);
    }
    if desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
        && !native
            .Flags
            .contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
        && !native
            .Flags
            .contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
    {
        return Err(Error::InvalidState);
    }
    if desc.usage.contains(api::TextureUsage::TEXTURE_BINDING)
        && native
            .Flags
            .contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
    {
        return Err(Error::InvalidState);
    }
    Ok(Dx12Texture(Arc::new(TextureState {
        resource,
        desc: desc.clone(),
        sparse: None,
        backing_heap: None,
        default_allocation: None,
    })))
}

fn transition_texture(
    encoder: &mut encoder::Dx12Encoder,
    texture: &Dx12Texture,
    transition: &api::ExternalTextureTransition,
) -> Result<()> {
    let desc = &texture.0.desc;
    let expected = if texture::is_depth(desc.format) {
        texture::depth_aspects(desc.format)
    } else {
        api::TextureAspects::COLOR
    };
    if transition.aspects.0 == 0
        || transition.aspects.0 & !expected.0 != 0
        || transition.mips.start >= transition.mips.end
        || transition.mips.end > desc.mip_levels
        || transition.layers.start >= transition.layers.end
        || transition.layers.end > desc.array_layers
    {
        return Err(Error::InvalidRange);
    }
    let depth = texture::is_depth(desc.format);
    if [transition.layout.0, transition.layout.1]
        .into_iter()
        .any(|layout| {
            (depth && layout == api::ImageLayout::ColorAttachment)
                || (!depth && layout == api::ImageLayout::DepthStencilAttachment)
        })
    {
        return Err(Error::Unsupported(
            "DX12 external texture layout/format mismatch",
        ));
    }
    let before = if depth && transition.layout.0 == api::ImageLayout::Undefined {
        if transition.usage.from.accesses != api::AccessMask::EMPTY {
            return Err(Error::Unsupported("DX12 Undefined layout access"));
        }
        D3D12_RESOURCE_STATE_DEPTH_WRITE
    } else {
        encoder::texture_state(transition.layout.0, transition.usage.from)?
    };
    let after = encoder::texture_state(transition.layout.1, transition.usage.to)?;
    if encoder.is_compute {
        let allowed = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER.0
            | D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0
            | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0
            | D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT.0
            | D3D12_RESOURCE_STATE_COPY_SOURCE.0
            | D3D12_RESOURCE_STATE_COPY_DEST.0;
        if (before.0 | after.0) & !allowed != 0 {
            return Err(Error::Unsupported("DX12 compute queue resource state"));
        }
    }
    // External textures use the same resource-wide ordering as Core textures.
    if encoder::needs_uav_barrier(
        before,
        after,
        transition.usage.from.accesses,
        transition.usage.to.accesses,
    ) {
        encoder::uav_barrier(&encoder.list, &texture.0.resource);
    }
    for layer in transition.layers.clone() {
        for mip in transition.mips.clone() {
            for plane in 0..2 {
                let aspect = if plane == 0 {
                    if texture::is_depth(desc.format) {
                        api::TextureAspects::DEPTH.0
                    } else {
                        api::TextureAspects::COLOR.0
                    }
                } else {
                    api::TextureAspects::STENCIL.0
                };
                if transition.aspects.0 & aspect == 0 {
                    continue;
                }
                let subresource =
                    mip + layer * desc.mip_levels + plane * desc.mip_levels * desc.array_layers;
                encoder::barrier_sub(
                    &encoder.list,
                    &texture.0.resource,
                    subresource,
                    before,
                    after,
                );
            }
        }
    }
    encoder.textures.push(texture.clone());
    Ok(())
}

impl ExternalBackend for Dx12 {
    type NativeExternalHandle = Dx12ExternalHandle;
    type NativeExternalMemory = Dx12Buffer;
    type NativeExternalTexture = Dx12Texture;
    type NativeExternalSync = Dx12ExternalSync;

    fn external_buffer(memory: &Dx12Buffer) -> Result<Dx12Buffer> {
        Ok(memory.clone())
    }

    fn external_texture(texture: &Dx12Texture) -> Result<Dx12Texture> {
        Ok(texture.clone())
    }

    unsafe fn import_external_sync(
        device: &Dx12Device,
        handle: Dx12ExternalHandle,
    ) -> Result<Dx12ExternalSync> {
        match handle {
            Dx12ExternalHandle::Fence { handle, value } => imported_fence(device, handle, value),
            Dx12ExternalHandle::Resource(_) => Err(Error::InvalidState),
        }
    }

    unsafe fn import_memory(
        device: &Dx12Device,
        handle: Dx12ExternalHandle,
        desc: &api::ExternalResourceDesc,
    ) -> Result<Dx12Buffer> {
        match handle {
            Dx12ExternalHandle::Resource(handle) => import_buffer(device, handle, desc),
            Dx12ExternalHandle::Fence { .. } => Err(Error::InvalidState),
        }
    }

    unsafe fn import_texture(
        device: &Dx12Device,
        handle: Dx12ExternalHandle,
        desc: &api::TextureDesc,
    ) -> Result<Dx12Texture> {
        match handle {
            Dx12ExternalHandle::Resource(handle) => import_texture(device, handle, desc),
            Dx12ExternalHandle::Fence { .. } => Err(Error::InvalidState),
        }
    }

    fn acquire_external(
        encoder: &mut encoder::Dx12Encoder,
        memory: &Dx12Buffer,
        sync: Dx12ExternalSync,
    ) -> Result<()> {
        let (fence, value) = sync.fence_value();
        encoder.split_external_boundary(encoder::external_segment::QueueBoundary::Wait {
            fence: fence.clone(),
            value,
        })?;
        encoder.buffers.push(memory.clone());
        Ok(())
    }

    fn release_external(
        encoder: &mut encoder::Dx12Encoder,
        memory: &Dx12Buffer,
    ) -> Result<Dx12ExternalSync> {
        let sync = exported_fence(&encoder.device)?;
        let (fence, value) = sync.fence_value();
        encoder.split_external_boundary(encoder::external_segment::QueueBoundary::Signal {
            fence: fence.clone(),
            value,
        })?;
        encoder.buffers.push(memory.clone());
        Ok(sync)
    }

    fn acquire_external_texture(
        encoder: &mut encoder::Dx12Encoder,
        texture: &Dx12Texture,
        sync: Dx12ExternalSync,
        transition: &api::ExternalTextureTransition,
    ) -> Result<()> {
        let (fence, value) = sync.fence_value();
        encoder.split_external_boundary(encoder::external_segment::QueueBoundary::Wait {
            fence: fence.clone(),
            value,
        })?;
        transition_texture(encoder, texture, transition)
    }

    fn release_external_texture(
        encoder: &mut encoder::Dx12Encoder,
        texture: &Dx12Texture,
        transition: &api::ExternalTextureTransition,
    ) -> Result<Dx12ExternalSync> {
        transition_texture(encoder, texture, transition)?;
        let sync = exported_fence(&encoder.device)?;
        let (fence, value) = sync.fence_value();
        encoder.split_external_boundary(encoder::external_segment::QueueBoundary::Signal {
            fence: fence.clone(),
            value,
        })?;
        Ok(sync)
    }
}
