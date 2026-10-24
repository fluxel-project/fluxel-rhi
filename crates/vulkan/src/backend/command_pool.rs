//! Exclusive, fence-retired Vulkan command-pool pages.
//!
//! A lease is the sole mutable owner of one queue-family/command-level pool.
//! It may reset that pool only when the recorder was cancelled before submit or
//! when the fence-retained recorded work drops it.  Pages retain command
//! buffers across resets, avoiding both pool creation and buffer accumulation.

use crate::backend::vk_error;
use ash::vk;
use fluxel_rhi_core::api::Result;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

pub(crate) struct CommandPoolCache {
    inner: Arc<Inner>,
}

struct Inner {
    device: ash::Device,
    pages: Mutex<HashMap<Key, Vec<Page>>>,
    cached: Mutex<usize>,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct Key {
    family: u32,
    level: vk::CommandBufferLevel,
}

struct Page {
    pool: vk::CommandPool,
    commands: Vec<vk::CommandBuffer>,
    cursor: usize,
}

/// An exclusive native pool page. It is intentionally non-cloneable: Vulkan
/// externally synchronizes command-pool reset against all pool use.
pub(crate) struct CommandPoolLease {
    inner: Arc<Inner>,
    key: Key,
    page: Option<Page>,
}

// This bounds only inactive native pages, never command recording or public
// capability. Extra pages are destroyed rather than causing Unsupported.
const MAX_CACHED_PAGES: usize = 32;

impl CommandPoolCache {
    pub(crate) fn new(device: &ash::Device) -> Self {
        Self {
            inner: Arc::new(Inner {
                device: device.clone(),
                pages: Mutex::new(HashMap::new()),
                cached: Mutex::new(0),
            }),
        }
    }

    pub(crate) fn lease(
        &self,
        family: u32,
        level: vk::CommandBufferLevel,
    ) -> Result<CommandPoolLease> {
        let key = Key { family, level };
        let page = {
            let mut cached = self.inner.cached.lock().unwrap_or_else(|e| e.into_inner());
            let mut pages = self.inner.pages.lock().unwrap_or_else(|e| e.into_inner());
            let page = pages.get_mut(&key).and_then(Vec::pop);
            if page.is_some() {
                *cached -= 1;
            }
            page
        };
        let page = match page {
            Some(page) => page,
            None => {
                let info = vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT);
                // SAFETY: the cache retains this live logical-device dispatch
                // handle; family and level came from backend-owned queue data,
                // and the new pool is transferred to one exclusive lease.
                let pool = unsafe { self.inner.device.create_command_pool(&info, None) }
                    .map_err(vk_error)?;
                Page {
                    pool,
                    commands: Vec::new(),
                    cursor: 0,
                }
            }
        };
        Ok(CommandPoolLease {
            inner: self.inner.clone(),
            key,
            page: Some(page),
        })
    }

    /// Must run while the logical device is still live (`DeviceState::drop`).
    pub(crate) fn clear(&self) {
        let mut cached = self.inner.cached.lock().unwrap_or_else(|e| e.into_inner());
        let mut pages = self.inner.pages.lock().unwrap_or_else(|e| e.into_inner());
        for page in pages.drain().flat_map(|(_, pages)| pages) {
            // SAFETY: DeviceState clears only inactive cache pages before
            // destroying the logical device.
            unsafe { self.inner.device.destroy_command_pool(page.pool, None) };
        }
        *cached = 0;
    }
}

impl CommandPoolLease {
    /// Returns the next reusable command buffer in its reset initial state.
    /// Callers that need secondary-command inheritance must provide their own
    /// `vkBeginCommandBuffer` info after this step.
    pub(crate) fn allocate(&mut self) -> Result<vk::CommandBuffer> {
        let page = self.page.as_mut().expect("command pool lease is live");
        let command = if page.cursor == page.commands.len() {
            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(page.pool)
                .level(self.key.level)
                .command_buffer_count(1);
            // SAFETY: this lease exclusively owns a live pool of the selected
            // family/level; the allocation output is retained by that page.
            let command = unsafe { self.inner.device.allocate_command_buffers(&allocate) }
                .map_err(vk_error)?[0];
            page.commands.push(command);
            command
        } else {
            page.commands[page.cursor]
        };
        page.cursor += 1;
        Ok(command)
    }

    /// Returns the next reusable primary buffer and begins one-time recording.
    pub(crate) fn begin(&mut self) -> Result<vk::CommandBuffer> {
        let command = self.allocate()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: allocate returned an initial/reset command buffer that this
        // exclusive lease has not begun in its current recording epoch.
        unsafe { self.inner.device.begin_command_buffer(command, &begin) }.map_err(vk_error)?;
        Ok(command)
    }
}

impl Drop for CommandPoolLease {
    fn drop(&mut self) {
        let Some(mut page) = self.page.take() else {
            return;
        };
        // The lease is owned by either an unsubmitted recorder or fence-retired
        // RecordedWork, so no queue can access these buffers at reset time.
        if unsafe {
            // SAFETY: this lease belongs to cancelled recording or
            // fence-retired work, so no queue can access its pool now.
            self.inner
                .device
                .reset_command_pool(page.pool, vk::CommandPoolResetFlags::empty())
        }
        .is_err()
        {
            // SAFETY: on reset failure this lease remains the unique pool
            // owner; destruction prevents unsafe cache reuse.
            unsafe { self.inner.device.destroy_command_pool(page.pool, None) };
            return;
        }
        page.cursor = 0;
        let mut cached = self.inner.cached.lock().unwrap_or_else(|e| e.into_inner());
        if *cached == MAX_CACHED_PAGES {
            drop(cached);
            // SAFETY: this inactive page is uniquely owned by the lease.
            unsafe { self.inner.device.destroy_command_pool(page.pool, None) };
            return;
        }
        let mut pages = self.inner.pages.lock().unwrap_or_else(|e| e.into_inner());
        pages.entry(self.key).or_default().push(page);
        *cached += 1;
    }
}
