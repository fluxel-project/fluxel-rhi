//! Vulkan external memory and binary semaphore ownership.
//!
//! Imported OS handles describe allocations shared with an independent Vulkan
//! device. Queue-family transfers are recorded in the native command buffer;
//! their semaphore waits and signals are submitted with that work.

use super::DeviceState;
use super::{Vulkan, VulkanBuffer, VulkanDevice, VulkanEncoder, VulkanTexture, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::sync::Arc;

mod import;

#[cfg(windows)]
type OwnedOsHandle = std::os::windows::io::OwnedHandle;
#[cfg(unix)]
type OwnedOsHandle = std::os::fd::OwnedFd;

/// An owned `OPAQUE_WIN32` or `OPAQUE_FD` allocation handle and its metadata.
///
/// The bridge must provide the exact allocation size, memory type index and
/// dedicated-allocation status used by the producer. The imported buffer or
/// image must have matching creation parameters; opaque handles cannot expose
/// these properties for validation.
pub struct VulkanExternalMemoryHandle {
    pub handle: OwnedOsHandle,
    pub allocation_size: u64,
    pub memory_type_index: u32,
    pub dedicated: bool,
    /// Offset of the image within the imported allocation; ignored for buffers.
    pub image_offset: u64,
}

/// Platform handle consumed by the matching external import operation.
///
/// On Windows the NT handle is closed after Vulkan imports it. On Unix the FD
/// transfers to Vulkan on successful import and is closed on failure.
pub enum VulkanExternalHandle {
    Memory(VulkanExternalMemoryHandle),
    Semaphore(OwnedOsHandle),
}

pub(super) struct ExternalSemaphore {
    pub(super) device: Arc<DeviceState>,
    pub(super) semaphore: vk::Semaphore,
    exportable: bool,
}
impl Drop for ExternalSemaphore {
    fn drop(&mut self) {
        // SAFETY: each submitted encoder retains this Arc until completion.
        unsafe { self.device.device.destroy_semaphore(self.semaphore, None) };
    }
}

/// A binary semaphore retained through native queue completion.
pub struct VulkanExternalSync(pub(super) Arc<ExternalSemaphore>);

#[cfg(windows)]
const SEMAPHORE_HANDLE: vk::ExternalSemaphoreHandleTypeFlags =
    vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32;
#[cfg(unix)]
const SEMAPHORE_HANDLE: vk::ExternalSemaphoreHandleTypeFlags =
    vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD;

fn semaphore_supported(device: &VulkanDevice, features: vk::ExternalSemaphoreFeatureFlags) -> bool {
    if !device.0.external_semaphore_handles {
        return false;
    }
    let info = vk::PhysicalDeviceExternalSemaphoreInfo::default().handle_type(SEMAPHORE_HANDLE);
    let mut properties = vk::ExternalSemaphoreProperties::default();
    // SAFETY: physical device belongs to the retained instance and output lives through the call.
    unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_external_semaphore_properties(
                device.0.physical,
                &info,
                &mut properties,
            )
    };
    properties.external_semaphore_features.contains(features)
}

fn import_semaphore(device: &VulkanDevice, handle: OwnedOsHandle) -> Result<VulkanExternalSync> {
    if !semaphore_supported(device, vk::ExternalSemaphoreFeatureFlags::IMPORTABLE) {
        return Err(Error::Unsupported("Vulkan external semaphore import"));
    }
    // SAFETY: a valid device creates an ordinary binary semaphore.
    let semaphore = unsafe {
        device
            .0
            .device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
    }
    .map_err(vk_error)?;
    let owner = Arc::new(ExternalSemaphore {
        device: device.0.clone(),
        semaphore,
        exportable: false,
    });
    #[cfg(windows)]
    let result = {
        use std::os::windows::io::AsRawHandle;
        let loader = ash::khr::external_semaphore_win32::Device::new(
            &device.0.instance.instance,
            &device.0.device,
        );
        let info = vk::ImportSemaphoreWin32HandleInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(SEMAPHORE_HANDLE)
            .handle(handle.as_raw_handle() as isize);
        // SAFETY: the unsafe public import contract guarantees the external handle is valid.
        unsafe { loader.import_semaphore_win32_handle(&info) }
    };
    #[cfg(unix)]
    let result = {
        use std::os::fd::{AsRawFd, IntoRawFd};
        let loader = ash::khr::external_semaphore_fd::Device::new(
            &device.0.instance.instance,
            &device.0.device,
        );
        let info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(SEMAPHORE_HANDLE)
            .fd(handle.as_raw_fd());
        // SAFETY: the unsafe public import contract guarantees a valid external FD.
        let result = unsafe { loader.import_semaphore_fd(&info) };
        if result.is_ok() {
            let _ = handle.into_raw_fd();
        }
        result
    };
    result.map_err(vk_error)?;
    Ok(VulkanExternalSync(owner))
}

fn exportable_semaphore(device: &VulkanDevice) -> Result<VulkanExternalSync> {
    if !semaphore_supported(device, vk::ExternalSemaphoreFeatureFlags::EXPORTABLE) {
        return Err(Error::Unsupported("Vulkan external semaphore export"));
    }
    let mut export = vk::ExportSemaphoreCreateInfo::default().handle_types(SEMAPHORE_HANDLE);
    let info = vk::SemaphoreCreateInfo::default().push_next(&mut export);
    // SAFETY: export support was queried and the pNext chain stays live.
    let semaphore = unsafe { device.0.device.create_semaphore(&info, None) }.map_err(vk_error)?;
    Ok(VulkanExternalSync(Arc::new(ExternalSemaphore {
        device: device.0.clone(),
        semaphore,
        exportable: true,
    })))
}

impl VulkanExternalSync {
    /// Export an owned OS handle. The receiving queue must wait on it before
    /// accessing the resource; the signal occurs when the release is submitted.
    pub fn export_handle(&self) -> Result<OwnedOsHandle> {
        let owner = &self.0;
        if !owner.exportable {
            return Err(Error::InvalidState);
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::FromRawHandle;
            let loader = ash::khr::external_semaphore_win32::Device::new(
                &owner.device.instance.instance,
                &owner.device.device,
            );
            let info = vk::SemaphoreGetWin32HandleInfoKHR::default()
                .semaphore(owner.semaphore)
                .handle_type(SEMAPHORE_HANDLE);
            // SAFETY: this is an exportable semaphore on the retained device.
            let handle = unsafe { loader.get_semaphore_win32_handle(&info) }.map_err(vk_error)?;
            // SAFETY: Vulkan returns a newly owned NT handle on success.
            Ok(unsafe { OwnedOsHandle::from_raw_handle(handle as *mut std::ffi::c_void) })
        }
        #[cfg(unix)]
        {
            use std::os::fd::FromRawFd;
            let loader = ash::khr::external_semaphore_fd::Device::new(
                &owner.device.instance.instance,
                &owner.device.device,
            );
            let info = vk::SemaphoreGetFdInfoKHR::default()
                .semaphore(owner.semaphore)
                .handle_type(SEMAPHORE_HANDLE);
            // SAFETY: this is an exportable semaphore on the retained device.
            let fd = unsafe { loader.get_semaphore_fd(&info) }.map_err(vk_error)?;
            // SAFETY: Vulkan returns a newly owned FD on success.
            Ok(unsafe { OwnedOsHandle::from_raw_fd(fd) })
        }
    }
}

fn ready(encoder: &VulkanEncoder) -> Result<u32> {
    if encoder.raster || encoder.compute {
        return Err(Error::InvalidState);
    }
    Ok(if encoder.lane.ordinal == 1 {
        encoder.device.compute_family
    } else {
        encoder.device.graphics_family
    })
}

fn buffer_transfer(
    encoder: &mut VulkanEncoder,
    buffer: &VulkanBuffer,
    acquire: bool,
) -> Result<()> {
    let family = ready(encoder)?;
    if !Arc::ptr_eq(&encoder.device, &buffer.0.device) {
        return Err(Error::WrongDevice);
    }
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(if acquire {
            vk::AccessFlags::empty()
        } else {
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
        })
        .dst_access_mask(if acquire {
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
        } else {
            vk::AccessFlags::empty()
        })
        .src_queue_family_index(if acquire {
            vk::QUEUE_FAMILY_EXTERNAL
        } else {
            family
        })
        .dst_queue_family_index(if acquire {
            family
        } else {
            vk::QUEUE_FAMILY_EXTERNAL
        })
        .buffer(buffer.0.buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE);
    // SAFETY: imported buffer and recording command belong to this device; the bridge guarantees matching external ownership operations.
    unsafe {
        encoder.device.device.cmd_pipeline_barrier(
            encoder.command,
            if acquire {
                vk::PipelineStageFlags::TOP_OF_PIPE
            } else {
                vk::PipelineStageFlags::ALL_COMMANDS
            },
            if acquire {
                vk::PipelineStageFlags::ALL_COMMANDS
            } else {
                vk::PipelineStageFlags::BOTTOM_OF_PIPE
            },
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        )
    };
    encoder.buffers.push(buffer.clone());
    Ok(())
}

fn texture_transfer(
    encoder: &mut VulkanEncoder,
    texture: &VulkanTexture,
    state: &api::ExternalTextureTransition,
    acquire: bool,
) -> Result<()> {
    let family = ready(encoder)?;
    if !Arc::ptr_eq(&encoder.device, &texture.0.device) {
        return Err(Error::WrongDevice);
    }
    let desc = texture.desc();
    let valid_aspects = match desc.format {
        api::TextureFormat::Depth16Unorm
        | api::TextureFormat::Depth24Plus
        | api::TextureFormat::Depth32Float => api::TextureAspects::DEPTH,
        api::TextureFormat::Stencil8 => api::TextureAspects::STENCIL,
        api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8 => {
            api::TextureAspects::DEPTH | api::TextureAspects::STENCIL
        }
        _ => api::TextureAspects::COLOR,
    };
    if state.aspects.0 == 0
        || state.aspects.0 & !valid_aspects.0 != 0
        || state.mips.start >= state.mips.end
        || state.mips.end > desc.mip_levels
        || state.layers.start >= state.layers.end
        || state.layers.end > desc.array_layers
    {
        return Err(Error::InvalidRange);
    }
    if valid_aspects == (api::TextureAspects::DEPTH | api::TextureAspects::STENCIL)
        && state.aspects != valid_aspects
        && state.layout.0 != state.layout.1
        && !encoder.device.separate_depth_stencil_layouts
    {
        return Err(Error::Unsupported("Vulkan separate depth/stencil layouts"));
    }
    super::transition::validate_texture_use(texture, state.layout.0, state.usage.from)?;
    super::transition::validate_texture_use(texture, state.layout.1, state.usage.to)?;
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(super::transition::aspects(state.aspects)?)
        .base_mip_level(state.mips.start)
        .level_count(state.mips.end - state.mips.start)
        .base_array_layer(state.layers.start)
        .layer_count(state.layers.end - state.layers.start);
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(if acquire {
            vk::AccessFlags::empty()
        } else {
            super::transition::accesses(state.usage.from.accesses)?
        })
        .dst_access_mask(if acquire {
            super::transition::accesses(state.usage.to.accesses)?
        } else {
            vk::AccessFlags::empty()
        })
        .old_layout(super::transition::layout(state.layout.0))
        .new_layout(super::transition::layout(state.layout.1))
        .src_queue_family_index(if acquire {
            vk::QUEUE_FAMILY_EXTERNAL
        } else {
            family
        })
        .dst_queue_family_index(if acquire {
            family
        } else {
            vk::QUEUE_FAMILY_EXTERNAL
        })
        .image(texture.native())
        .subresource_range(range);
    let source = if acquire {
        vk::PipelineStageFlags::TOP_OF_PIPE
    } else {
        super::transition::stages(state.usage.from.stages)?
    };
    let dest = if acquire {
        super::transition::stages(state.usage.to.stages)?
    } else {
        vk::PipelineStageFlags::BOTTOM_OF_PIPE
    };
    // SAFETY: range and queue ownership are explicit; the bridge guarantees the matching external operation and lifetime.
    unsafe {
        encoder.device.device.cmd_pipeline_barrier(
            encoder.command,
            source,
            dest,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        )
    };
    encoder.textures.push(texture.clone());
    Ok(())
}

impl api::ExternalBackend for Vulkan {
    type NativeExternalHandle = VulkanExternalHandle;
    type NativeExternalMemory = VulkanBuffer;
    type NativeExternalTexture = VulkanTexture;
    type NativeExternalSync = VulkanExternalSync;
    fn external_buffer(memory: &VulkanBuffer) -> Result<VulkanBuffer> {
        Ok(memory.clone())
    }
    fn external_texture(texture: &VulkanTexture) -> Result<VulkanTexture> {
        Ok(texture.clone())
    }
    fn import_external_sync(
        device: &VulkanDevice,
        handle: VulkanExternalHandle,
    ) -> Result<VulkanExternalSync> {
        match handle {
            VulkanExternalHandle::Semaphore(handle) => import_semaphore(device, handle),
            _ => Err(Error::InvalidState),
        }
    }
    fn import_memory(
        device: &VulkanDevice,
        handle: VulkanExternalHandle,
        desc: &api::ExternalResourceDesc,
    ) -> Result<VulkanBuffer> {
        match handle {
            VulkanExternalHandle::Memory(handle) => import::import_buffer(device, handle, desc),
            _ => Err(Error::InvalidState),
        }
    }
    fn import_texture(
        device: &VulkanDevice,
        handle: VulkanExternalHandle,
        desc: &api::TextureDesc,
    ) -> Result<VulkanTexture> {
        match handle {
            VulkanExternalHandle::Memory(handle) => import::import_texture(device, handle, desc),
            _ => Err(Error::InvalidState),
        }
    }
    fn acquire_external(
        encoder: &mut VulkanEncoder,
        memory: &VulkanBuffer,
        sync: VulkanExternalSync,
    ) -> Result<()> {
        if !Arc::ptr_eq(&encoder.device, &sync.0.device) {
            return Err(Error::WrongDevice);
        }
        buffer_transfer(encoder, memory, true)?;
        encoder.external_waits.push(sync.0);
        Ok(())
    }
    fn release_external(
        encoder: &mut VulkanEncoder,
        memory: &VulkanBuffer,
    ) -> Result<VulkanExternalSync> {
        let sync = exportable_semaphore(&encoder.owner)?;
        buffer_transfer(encoder, memory, false)?;
        encoder.external_signals.push(sync.0.clone());
        Ok(sync)
    }
    fn acquire_external_texture(
        encoder: &mut VulkanEncoder,
        texture: &VulkanTexture,
        sync: VulkanExternalSync,
        state: &api::ExternalTextureTransition,
    ) -> Result<()> {
        if !Arc::ptr_eq(&encoder.device, &sync.0.device) {
            return Err(Error::WrongDevice);
        }
        texture_transfer(encoder, texture, state, true)?;
        encoder.external_waits.push(sync.0);
        Ok(())
    }
    fn release_external_texture(
        encoder: &mut VulkanEncoder,
        texture: &VulkanTexture,
        state: &api::ExternalTextureTransition,
    ) -> Result<VulkanExternalSync> {
        let sync = exportable_semaphore(&encoder.owner)?;
        texture_transfer(encoder, texture, state, false)?;
        encoder.external_signals.push(sync.0.clone());
        Ok(sync)
    }
}
