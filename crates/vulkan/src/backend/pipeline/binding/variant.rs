//! Lazily materialized descriptor sets for the pipeline's sampler numeric types.
use super::*;
use std::collections::HashMap;

pub(super) enum Snapshot {
    Buffer {
        binding: u32,
        element: u32,
        ty: vk::DescriptorType,
        info: vk::DescriptorBufferInfo,
    },
    Image {
        binding: u32,
        element: u32,
        ty: vk::DescriptorType,
        info: vk::DescriptorImageInfo,
    },
    Sampler {
        binding: u32,
        element: u32,
        float: vk::Sampler,
        integer: vk::Sampler,
    },
    Acceleration {
        binding: u32,
        element: u32,
        handle: vk::AccelerationStructureKHR,
    },
}
pub(super) struct Variant {
    // Field order is lifetime-critical: return/reset this page before the
    // DeviceState owners below can be the final logical-device reference.
    #[allow(dead_code)]
    pool: super::super::descriptor_pool::DescriptorPoolLease,
    device: Arc<DeviceState>,
    _layout: VulkanBindGroupLayout,
    key: Vec<(u32, u8)>,
    sampler: Option<vk::Sampler>,
    set: vk::DescriptorSet,
}
impl Drop for Variant {
    fn drop(&mut self) {
        // The page lease resets only after this variant is no longer retained
        // by its fence-owned group; sampler lifetime remains native-local.
        if let Some(sampler) = self.sampler {
            unsafe { self.device.device.destroy_sampler(sampler, None) }
        }
    }
}
impl VulkanBindGroup {
    pub(crate) fn native_variant(
        &self,
        layout: &VulkanBindGroupLayout,
        modes: &HashMap<u32, u8>,
    ) -> Result<vk::DescriptorSet> {
        if layout.native() == self.layout() && modes.values().all(|&m| m == 0) {
            return Ok(self.native());
        }
        let mut key: Vec<_> = modes
            .iter()
            .filter(|(_, m)| **m != 0)
            .map(|(&b, &m)| (b, m))
            .collect();
        key.sort_unstable();
        if key.iter().any(|(_, m)| *m > 2) {
            return Err(Error::InvalidState);
        }
        let mut cache = self.0.variants.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = cache
            .iter()
            .find(|v| v._layout.native() == layout.native() && v.key == key)
        {
            return Ok(existing.set);
        }
        let value = create(&self.0, layout, key)?;
        let set = value.set;
        cache.push(value);
        Ok(set)
    }
}
fn create(group: &Group, layout: &VulkanBindGroupLayout, key: Vec<(u32, u8)>) -> Result<Variant> {
    let mut sizes = group.pool_sizes.clone();
    let mut mirror_count = 0u32;
    for entry in layout.0.entries.iter() {
        if layout.0.sampler_mirrors.contains_key(&entry.binding) {
            let n = match entry.count {
                api::BindingCount::One => 1,
                api::BindingCount::Fixed(n) => n,
                api::BindingCount::RuntimeSized { max_count } => max_count,
            };
            mirror_count = mirror_count.checked_add(n).ok_or(Error::InvalidRange)?;
        }
    }
    if mirror_count != 0 {
        if let Some(size) = sizes
            .iter_mut()
            .find(|s| s.ty == vk::DescriptorType::SAMPLER)
        {
            size.descriptor_count = size
                .descriptor_count
                .checked_add(mirror_count)
                .ok_or(Error::InvalidRange)?;
        } else {
            sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: mirror_count,
            });
        }
    }
    let pool = group.device.descriptor_pools.lease(&sizes)?;
    let native_pool = pool.pool();
    let mut variant = Variant {
        device: group.device.clone(),
        _layout: layout.clone(),
        key,
        pool,
        sampler: None,
        set: vk::DescriptorSet::null(),
    };
    let layouts = [layout.native()];
    let variable = group.variable_count.into_iter().collect::<Vec<_>>();
    let mut variable_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
        .descriptor_counts(&variable);
    let mut allocate = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(native_pool)
        .set_layouts(&layouts);
    if !variable.is_empty() {
        allocate = allocate.push_next(&mut variable_info);
    }
    // SAFETY: layout and pool outlive the allocated set via Variant.
    variant.set =
        unsafe { group.device.device.allocate_descriptor_sets(&allocate) }.map_err(vk_error)?[0];
    if mirror_count != 0 {
        // SAFETY: default sampler is valid and retained until the descriptor pool is destroyed.
        let sampler = unsafe {
            group
                .device
                .device
                .create_sampler(&vk::SamplerCreateInfo::default(), None)
        }
        .map_err(vk_error)?;
        variant.sampler = Some(sampler);
        for entry in layout.0.entries.iter() {
            let Some(&mirror) = layout.0.sampler_mirrors.get(&entry.binding) else {
                continue;
            };
            let count = match entry.count {
                api::BindingCount::One => 1,
                api::BindingCount::Fixed(n) => n,
                api::BindingCount::RuntimeSized { max_count } => max_count,
            };
            let info = vec![vk::DescriptorImageInfo::default().sampler(sampler); count as usize];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(variant.set)
                .dst_binding(mirror)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&info);
            // SAFETY: set, sampler, and data are valid for the duration of this write.
            update(group, write);
        }
    }
    for snapshot in &group.snapshots {
        let (binding, element) = match snapshot {
            Snapshot::Buffer {
                binding, element, ..
            }
            | Snapshot::Image {
                binding, element, ..
            }
            | Snapshot::Sampler {
                binding, element, ..
            }
            | Snapshot::Acceleration {
                binding, element, ..
            } => (*binding, *element),
        };
        let native = *layout
            .0
            .native_bindings
            .get(&binding)
            .ok_or(Error::InvalidState)?;
        let base = vk::WriteDescriptorSet::default()
            .dst_set(variant.set)
            .dst_binding(native)
            .dst_array_element(element);
        match snapshot {
            Snapshot::Buffer { ty, info, .. } => {
                let write = base
                    .descriptor_type(*ty)
                    .buffer_info(std::slice::from_ref(info));
                update(group, write);
            }
            Snapshot::Image { ty, info, .. } => {
                let write = base
                    .descriptor_type(*ty)
                    .image_info(std::slice::from_ref(info));
                update(group, write);
            }
            Snapshot::Sampler { float, integer, .. } => {
                let mode = variant
                    .key
                    .iter()
                    .find(|(b, _)| *b == binding)
                    .map_or(0, |(_, m)| *m);
                let info = vk::DescriptorImageInfo::default().sampler(if mode == 1 {
                    *integer
                } else {
                    *float
                });
                let write = base
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(std::slice::from_ref(&info));
                update(group, write);
                if let Some(&mirror) = layout.0.sampler_mirrors.get(&binding) {
                    let info = vk::DescriptorImageInfo::default().sampler(*integer);
                    let write = base
                        .dst_binding(mirror)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(std::slice::from_ref(&info));
                    update(group, write);
                }
            }
            Snapshot::Acceleration { handle, .. } => {
                let handles = [*handle];
                let mut info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                    .acceleration_structures(&handles);
                let write = base
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .push_next(&mut info);
                update(group, write);
            }
        }
    }
    Ok(variant)
}

fn update(group: &Group, write: vk::WriteDescriptorSet<'_>) {
    // SAFETY: the set is private during construction, descriptor data is live,
    // and the group retains each referenced native resource until retirement.
    unsafe { group.device.device.update_descriptor_sets(&[write], &[]) };
}
