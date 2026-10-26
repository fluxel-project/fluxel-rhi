//! Native pipeline cache ownership and serialization.
use super::super::{DeviceState, Vulkan, VulkanDevice, vk_error};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct VulkanPipelineCache(Arc<Cache>);
struct Cache {
    device: Arc<DeviceState>,
    handle: vk::PipelineCache,
    // Vulkan externally synchronizes both pipeline creation and cache reads.
    gate: Mutex<()>,
}
impl Drop for Cache {
    fn drop(&mut self) {
        // SAFETY: this is the last cache owner and retains its device.
        unsafe { self.device.device.destroy_pipeline_cache(self.handle, None) }
    }
}
impl VulkanPipelineCache {
    pub(crate) fn with<R>(&self, f: impl FnOnce(vk::PipelineCache) -> R) -> R {
        let _guard = self.0.gate.lock().unwrap_or_else(|e| e.into_inner());
        f(self.0.handle)
    }
}
impl fluxel_rhi_core::backend::CacheBackend for Vulkan {
    fn create_cache(device: &VulkanDevice, data: Option<&[u8]>) -> Result<VulkanPipelineCache> {
        if let Some(bytes) = data {
            if bytes.len() < 32 {
                return Err(Error::InvalidRange);
            }
            let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            // SAFETY: the physical device belongs to the retained instance.
            let properties = unsafe {
                device
                    .0
                    .instance
                    .instance
                    .get_physical_device_properties(device.0.physical)
            };
            if word(0) < 32
                || word(0) as usize > bytes.len()
                || word(4) != 1
                || word(8) != properties.vendor_id
                || word(12) != properties.device_id
                || bytes[16..32] != properties.pipeline_cache_uuid
            {
                return Err(Error::InvalidRange);
            }
        }
        let info = vk::PipelineCacheCreateInfo::default().initial_data(data.unwrap_or_default());
        // SAFETY: initial data remains live for the synchronous call.
        let handle =
            unsafe { device.0.device.create_pipeline_cache(&info, None) }.map_err(vk_error)?;
        Ok(VulkanPipelineCache(Arc::new(Cache {
            device: device.0.clone(),
            handle,
            gate: Mutex::new(()),
        })))
    }
    fn serialize_cache(cache: &VulkanPipelineCache) -> Result<Vec<u8>> {
        cache.with(|handle| {
            // SAFETY: the cache and its device remain alive while the lock is held.
            unsafe { cache.0.device.device.get_pipeline_cache_data(handle) }.map_err(vk_error)
        })
    }
}

pub(crate) fn with_cache<R>(
    cache: Option<&api::PipelineCache<Vulkan>>,
    f: impl FnOnce(vk::PipelineCache) -> R,
) -> R {
    match cache {
        // SAFETY: Core validates the cache's device identity on pipeline creation.
        Some(cache) => unsafe { cache.native_ref() }.with(f),
        None => f(vk::PipelineCache::null()),
    }
}
