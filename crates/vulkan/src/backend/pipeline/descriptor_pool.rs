//! Fence-safe reuse of single-set Vulkan descriptor pools.
//!
//! A lease owns one pool while a bind group (or its sampler variant) owns its
//! sole descriptor set.  Its `Drop` can therefore reset and cache that pool:
//! encoder retention keeps the bind group alive until submitted work reaches
//! its fence.  This module deliberately has no scheduler or frame policy.

use ash::vk;
use fluxel_rhi_core::api::{Error, Result};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

/// Per-device cache of one-set descriptor-pool pages.  It owns only an ash
/// dispatch handle, never `Arc<DeviceState>`, so cached native objects cannot
/// make the logical-device ownership graph cyclic.
pub(crate) struct DescriptorPoolCache {
    inner: Arc<Inner>,
}

struct Inner {
    device: ash::Device,
    pools: Mutex<HashMap<Key, Vec<vk::DescriptorPool>>>,
    cached: Mutex<usize>,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct Key(Vec<(i32, u32)>);

/// One exclusive descriptor-pool page.  A page supplies exactly one set;
/// this avoids a general suballocator and makes reset safe at group drop.
pub(crate) struct DescriptorPoolLease {
    inner: Arc<Inner>,
    key: Key,
    pool: vk::DescriptorPool,
}

// A capped cache is a memory high-water guard, not an allocation limit: an
// over-cap pool is simply destroyed and every request still succeeds normally.
// At most one retained page per exact descriptor capacity and 32 total pages
// make the common repeated-layout path allocation-free while bounding idle
// driver memory across unusual layouts.
const MAX_CACHED_PAGES: usize = 32;

impl DescriptorPoolCache {
    pub(crate) fn new(device: &ash::Device) -> Self {
        Self {
            inner: Arc::new(Inner {
                device: device.clone(),
                pools: Mutex::new(HashMap::new()),
                cached: Mutex::new(0),
            }),
        }
    }

    pub(crate) fn lease(&self, sizes: &[vk::DescriptorPoolSize]) -> Result<DescriptorPoolLease> {
        let key = Key::from_sizes(sizes);
        let pool = {
            let mut cached = self.inner.cached.lock().unwrap_or_else(|e| e.into_inner());
            let mut pools = self.inner.pools.lock().unwrap_or_else(|e| e.into_inner());
            let pool = pools.get_mut(&key).and_then(Vec::pop);
            if pool.is_some() {
                *cached -= 1;
            }
            pool
        };
        let pool = match pool {
            Some(pool) => pool,
            None => unsafe {
                // SAFETY: this cache was created from the still-live logical
                // device, sizes describe one validated set, and ownership is
                // immediately transferred into the returned exclusive lease.
                self.inner.device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(sizes),
                    None,
                )
            }
            .map_err(|e| Error::Backend(format!("Vulkan: {e:?}")))?,
        };
        Ok(DescriptorPoolLease {
            inner: self.inner.clone(),
            key,
            pool,
        })
    }

    /// Must run while the logical device is still live (from `DeviceState::drop`).
    pub(crate) fn clear(&self) {
        let mut cached = self.inner.cached.lock().unwrap_or_else(|e| e.into_inner());
        let mut pools = self.inner.pools.lock().unwrap_or_else(|e| e.into_inner());
        for pool in pools.drain().flat_map(|(_, pools)| pools) {
            // SAFETY: DeviceState calls clear before vkDestroyDevice; cached
            // pages are inactive and no lease can reference this pool.
            unsafe { self.inner.device.destroy_descriptor_pool(pool, None) };
        }
        *cached = 0;
    }
}

impl Key {
    fn from_sizes(sizes: &[vk::DescriptorPoolSize]) -> Self {
        let mut values = sizes
            .iter()
            .map(|size| (size.ty.as_raw(), size.descriptor_count))
            .collect::<Vec<_>>();
        values.sort_unstable();
        Self(values)
    }
}

impl DescriptorPoolLease {
    pub(crate) fn pool(&self) -> vk::DescriptorPool {
        self.pool
    }
}

impl Drop for DescriptorPoolLease {
    fn drop(&mut self) {
        // Group/variant ownership and fence retention prove no command buffer
        // can still access this page when its only descriptor set is reset.
        if unsafe {
            // SAFETY: group/variant and fence retention prove no queue can
            // access this lease's only descriptor set at this point.
            self.inner
                .device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
        }
        .is_err()
        {
            // SAFETY: reset failed but this lease remains the unique owner;
            // destroying instead of caching prevents future reuse.
            unsafe { self.inner.device.destroy_descriptor_pool(self.pool, None) };
            return;
        }
        let mut cached = self.inner.cached.lock().unwrap_or_else(|e| e.into_inner());
        if *cached == MAX_CACHED_PAGES {
            drop(cached);
            // SAFETY: the inactive page is uniquely owned by this lease.
            unsafe { self.inner.device.destroy_descriptor_pool(self.pool, None) };
            return;
        }
        let mut pools = self.inner.pools.lock().unwrap_or_else(|e| e.into_inner());
        let pages = pools.entry(self.key.clone()).or_default();
        if pages.is_empty() {
            pages.push(self.pool);
            *cached += 1;
        } else {
            drop(pools);
            drop(cached);
            // SAFETY: another inactive page already represents this exact
            // capacity, and this unique inactive page needs no retention.
            unsafe { self.inner.device.destroy_descriptor_pool(self.pool, None) };
        }
    }
}
