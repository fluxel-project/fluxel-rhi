//! Vulkan sparse image residency and backing allocations.
use super::{
    DeviceState, Vulkan, VulkanDevice, VulkanTexture,
    resource::{self, TextureState},
    vk_error,
};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Clone)]
pub struct VulkanSparse(Arc<Allocation>);
struct Allocation {
    device: Arc<DeviceState>,
    memory: vk::DeviceMemory,
    size: u64,
    memory_type: u32,
    owners: Mutex<Vec<Owner>>,
}
#[derive(Clone)]
struct Owner {
    texture: usize,
    tile: usize,
    start: u64,
    end: u64,
    uncertain: bool,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        // SAFETY: mappings and submitted work retain this allocation until retirement.
        unsafe { self.device.device.free_memory(self.memory, None) }
    }
}
pub(super) struct SparseTextureState {
    texture_id: AtomicUsize,
    uncertain: AtomicBool,
    tiling: api::SparseTextureTiling,
    requirements: vk::MemoryRequirements,
    tail_offset: u64,
    tail_size: u64,
    mappings: Mutex<Vec<Option<(VulkanSparse, u64)>>>,
    starts: Vec<usize>,
}
impl Drop for SparseTextureState {
    fn drop(&mut self) {
        let texture = self.texture_id.load(Ordering::Relaxed);
        for (allocation, _) in self
            .mappings
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .flatten()
        {
            allocation
                .0
                .owners
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|o| o.texture != texture);
        }
    }
}
impl api::SparseBackend for Vulkan {
    fn create_sparse_texture(
        device: &VulkanDevice,
        desc: &api::TextureDesc,
    ) -> Result<VulkanTexture> {
        if !device.0.sparse_image {
            return Err(Error::Unsupported("Vulkan sparse image"));
        }
        if desc.dimension != api::TextureDimension::D2
            || desc.format != api::TextureFormat::Rgba8Unorm
            || desc.array_layers != 1
            || desc.samples != 1
            || desc.extent[2] != 1
            || !matches!(
                resource::texture_support(device, desc),
                api::Support::Supported(_)
            )
        {
            return Err(Error::Unsupported("Vulkan sparse texture descriptor"));
        }
        let usage = resource::sparse_image_usage(desc.usage)?;
        let info = vk::ImageCreateInfo::default()
            .flags(vk::ImageCreateFlags::SPARSE_BINDING | vk::ImageCreateFlags::SPARSE_RESIDENCY)
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: desc.extent[0],
                height: desc.extent[1],
                depth: 1,
            })
            .mip_levels(desc.mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: validated descriptor and enabled sparse device features.
        let image = unsafe { device.0.device.create_image(&info, None) }.map_err(vk_error)?;
        let setup = || -> Result<SparseTextureState> {
            // SAFETY: image belongs to the live device.
            let req = unsafe { device.0.device.get_image_memory_requirements(image) };
            // SAFETY: image remains live and belongs to this device.
            let sparse = unsafe { device.0.device.get_image_sparse_memory_requirements(image) };
            if sparse.len() != 1
                || sparse[0].format_properties.aspect_mask != vk::ImageAspectFlags::COLOR
            {
                return Err(Error::Unsupported("Vulkan sparse image aspects"));
            }
            let s = sparse[0];
            let gran = s.format_properties.image_granularity;
            if gran.width == 0
                || gran.height == 0
                || gran.depth != 1
                || req.alignment == 0
                || s.image_mip_tail_size % req.alignment != 0
            {
                return Err(Error::Unsupported("Vulkan sparse image granularity"));
            }
            let standard = s.image_mip_tail_first_lod.min(desc.mip_levels);
            let mut grids = Vec::new();
            let mut starts = Vec::new();
            let mut total = 0usize;
            for mip in 0..standard {
                let w = (desc.extent[0] >> mip).max(1);
                let h = (desc.extent[1] >> mip).max(1);
                let grid = [w.div_ceil(gran.width), h.div_ceil(gran.height), 1];
                starts.push(total);
                total = total
                    .checked_add(
                        (u64::from(grid[0]) * u64::from(grid[1]))
                            .try_into()
                            .map_err(|_| Error::InvalidRange)?,
                    )
                    .ok_or(Error::InvalidRange)?;
                grids.push(grid);
            }
            let tail = if standard < desc.mip_levels {
                let count = u32::try_from(s.image_mip_tail_size / req.alignment)
                    .map_err(|_| Error::InvalidRange)?;
                if count == 0 {
                    return Err(Error::Unsupported("Vulkan sparse empty mip tail"));
                }
                Some(api::SparsePackedMipTail {
                    first_mip: standard,
                    mip_count: desc.mip_levels - standard,
                    tile_count: count,
                })
            } else {
                None
            };
            starts.push(total);
            total = total
                .checked_add(tail.map_or(0, |t| t.tile_count as usize))
                .ok_or(Error::InvalidRange)?;
            Ok(SparseTextureState {
                texture_id: AtomicUsize::new(0),
                uncertain: AtomicBool::new(false),
                tiling: api::SparseTextureTiling {
                    tile_extent: [gran.width, gran.height, 1],
                    tile_grid: grids.first().copied().unwrap_or([0; 3]),
                    tile_bytes: req.alignment,
                    standard_mip_grids: grids,
                    packed_tail: tail,
                },
                requirements: req,
                tail_offset: s.image_mip_tail_offset,
                tail_size: s.image_mip_tail_size,
                mappings: Mutex::new(vec![None; total]),
                starts,
            })
        };
        let state = setup().inspect_err(|_| {
            // SAFETY: creation failed before the image escaped or was submitted.
            unsafe { device.0.device.destroy_image(image, None) }
        })?;
        let texture = VulkanTexture(Arc::new(TextureState {
            device: device.0.clone(),
            image,
            memory: None,
            sparse: Some(state),
            desc: desc.clone(),
        }));
        texture
            .0
            .sparse
            .as_ref()
            .unwrap()
            .texture_id
            .store(Arc::as_ptr(&texture.0) as usize, Ordering::Relaxed);
        Ok(texture)
    }
    fn sparse_texture_tiling(
        _: &VulkanDevice,
        texture: &VulkanTexture,
    ) -> Result<api::SparseTextureTiling> {
        Ok(texture
            .0
            .sparse
            .as_ref()
            .ok_or(Error::InvalidState)?
            .tiling
            .clone())
    }
    fn create_sparse_allocation(_: &VulkanDevice, _: u64) -> Result<VulkanSparse> {
        Err(Error::Unsupported(
            "Vulkan sparse allocation requires a texture",
        ))
    }
    fn create_sparse_allocation_for(
        device: &VulkanDevice,
        texture: &VulkanTexture,
        size: u64,
    ) -> Result<VulkanSparse> {
        let state = texture.0.sparse.as_ref().ok_or(Error::InvalidState)?;
        if size == 0 || size % state.requirements.alignment != 0 {
            return Err(Error::InvalidRange);
        }
        // SAFETY: physical device and instance are retained by device.
        let props = unsafe {
            device
                .0
                .instance
                .instance
                .get_physical_device_memory_properties(device.0.physical)
        };
        let index = (0..props.memory_type_count)
            .find(|i| {
                state.requirements.memory_type_bits & (1 << i) != 0
                    && props.memory_types[*i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or(Error::OutOfMemory)?;
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(index);
        // SAFETY: selected memory type satisfies the image requirements.
        let memory = unsafe { device.0.device.allocate_memory(&info, None) }.map_err(vk_error)?;
        Ok(VulkanSparse(Arc::new(Allocation {
            device: device.0.clone(),
            memory,
            size,
            memory_type: index,
            owners: Mutex::new(Vec::new()),
        })))
    }
}

/// Prepared mappings own their input texture and allocations until submission retires.
pub(super) struct PreparedSparseBind {
    texture: VulkanTexture,
    allocation: Option<VulkanSparse>,
    image_binds: Vec<vk::SparseImageMemoryBind>,
    opaque_binds: Vec<vk::SparseMemoryBind>,
    indices: Vec<usize>,
    allocation_offset: u64,
    retained: Vec<VulkanSparse>,
}
/// Simulate all packet bindings under the submission gate before native handoff.
pub(super) fn validate_transaction(binds: &[&PreparedSparseBind]) -> Result<()> {
    use std::collections::HashMap;
    let mut allocations: HashMap<usize, VulkanSparse> = HashMap::new();
    for bind in binds {
        if let Some(a) = &bind.allocation {
            allocations.insert(Arc::as_ptr(&a.0) as usize, a.clone());
        }
        let state = bind.texture.0.sparse.as_ref().expect("prepared image");
        if state.uncertain.load(Ordering::Relaxed) {
            return Err(Error::InvalidState);
        }
        let mappings = state.mappings.lock().unwrap_or_else(|e| e.into_inner());
        for (a, _) in mappings.iter().flatten() {
            allocations.insert(Arc::as_ptr(&a.0) as usize, a.clone());
        }
    }
    let mut owners: HashMap<usize, Vec<Owner>> = allocations
        .iter()
        .map(|(&id, a)| {
            (
                id,
                a.0.owners.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            )
        })
        .collect();
    for bind in binds {
        let texture = Arc::as_ptr(&bind.texture.0) as usize;
        let bytes = bind
            .texture
            .0
            .sparse
            .as_ref()
            .expect("prepared image")
            .tiling
            .tile_bytes;
        for (position, &tile) in bind.indices.iter().enumerate() {
            for entries in owners.values_mut() {
                entries.retain(|o| o.uncertain || o.texture != texture || o.tile != tile);
            }
            if let Some(allocation) = &bind.allocation {
                let entries = owners
                    .get_mut(&(Arc::as_ptr(&allocation.0) as usize))
                    .expect("collected allocation");
                let start = bind.allocation_offset + position as u64 * bytes;
                let end = start + bytes;
                if entries.iter().any(|o| start < o.end && o.start < end) {
                    return Err(Error::InvalidState);
                }
                entries.push(Owner {
                    texture,
                    tile,
                    start,
                    end,
                    uncertain: false,
                });
            }
        }
    }
    Ok(())
}
pub(super) fn prepare_bind(bind: &api::SparseBind<Vulkan>) -> Result<PreparedSparseBind> {
    // SAFETY: packet validation checked both handles belong to this device.
    let texture = unsafe { bind.resource.native_ref() }.clone();
    let state = texture.0.sparse.as_ref().ok_or(Error::InvalidState)?;
    // SAFETY: Core validated the optional allocation's device identity.
    let allocation = bind
        .allocation
        .as_ref()
        .map(|a| unsafe { a.native_ref() }.clone());
    if bind.tiles.is_empty()
        || bind.allocation_offset % state.tiling.tile_bytes != 0
        || (allocation.is_none() && bind.allocation_offset != 0)
    {
        return Err(Error::InvalidRange);
    }
    if allocation
        .as_ref()
        .is_some_and(|a| state.requirements.memory_type_bits & (1 << a.0.memory_type) == 0)
    {
        return Err(Error::InvalidRange);
    }
    let mut image_binds = Vec::new();
    let mut opaque_binds = Vec::new();
    let mut indices = Vec::new();
    let mut offset = bind.allocation_offset;
    for tile in &bind.tiles {
        if tile.layer != 0 {
            return Err(Error::InvalidRange);
        }
        let memory = allocation
            .as_ref()
            .map_or(vk::DeviceMemory::null(), |a| a.0.memory);
        if let Some(grid) = state.tiling.standard_mip_grids.get(tile.mip as usize) {
            if tile.origin[2] != 0
                || tile.count[2] != 1
                || tile.count[0] == 0
                || tile.count[1] == 0
                || tile.origin[0]
                    .checked_add(tile.count[0])
                    .is_none_or(|v| v > grid[0])
                || tile.origin[1]
                    .checked_add(tile.count[1])
                    .is_none_or(|v| v > grid[1])
            {
                return Err(Error::InvalidRange);
            }
            let gran = state.tiling.tile_extent;
            let w = (texture.0.desc.extent[0] >> tile.mip).max(1);
            let h = (texture.0.desc.extent[1] >> tile.mip).max(1);
            for y in tile.origin[1]..tile.origin[1] + tile.count[1] {
                for x in tile.origin[0]..tile.origin[0] + tile.count[0] {
                    let px = x * gran[0];
                    let py = y * gran[1];
                    image_binds.push(vk::SparseImageMemoryBind {
                        subresource: vk::ImageSubresource {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: tile.mip,
                            array_layer: 0,
                        },
                        offset: vk::Offset3D {
                            x: px as i32,
                            y: py as i32,
                            z: 0,
                        },
                        extent: vk::Extent3D {
                            width: gran[0].min(w - px),
                            height: gran[1].min(h - py),
                            depth: 1,
                        },
                        memory,
                        memory_offset: if allocation.is_some() { offset } else { 0 },
                        flags: vk::SparseMemoryBindFlags::empty(),
                    });
                    indices.push(state.starts[tile.mip as usize] + (y * grid[0] + x) as usize);
                    offset = offset
                        .checked_add(state.tiling.tile_bytes)
                        .ok_or(Error::InvalidRange)?;
                }
            }
        } else if let Some(tail) = state.tiling.packed_tail.filter(|t| t.first_mip == tile.mip) {
            if tile.origin != [0, 0, 0] || tile.count != [tail.tile_count, 1, 1] {
                return Err(Error::InvalidRange);
            }
            opaque_binds.push(vk::SparseMemoryBind {
                resource_offset: state.tail_offset,
                size: state.tail_size,
                memory,
                memory_offset: if allocation.is_some() { offset } else { 0 },
                flags: vk::SparseMemoryBindFlags::empty(),
            });
            indices.extend(
                state.starts.last().copied().unwrap()
                    ..state.starts.last().copied().unwrap() + tail.tile_count as usize,
            );
            offset = offset
                .checked_add(state.tail_size)
                .ok_or(Error::InvalidRange)?;
        } else {
            return Err(Error::InvalidRange);
        }
    }
    if allocation.as_ref().is_some_and(|a| offset > a.0.size) {
        return Err(Error::InvalidRange);
    }
    let mut sorted = indices.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|x| x[0] == x[1]) {
        return Err(Error::InvalidRange);
    }
    Ok(PreparedSparseBind {
        texture,
        allocation,
        image_binds,
        opaque_binds,
        indices,
        allocation_offset: bind.allocation_offset,
        retained: Vec::new(),
    })
}
impl PreparedSparseBind {
    /// Submit under the device and queue gates. Keep this object until completion,
    /// including on an indeterminate error, because Vulkan may have accepted work.
    pub(super) fn submit(
        &mut self,
        device: &VulkanDevice,
        queue: vk::Queue,
        waits: &[(vk::Semaphore, u64)],
        signals: &[(vk::Semaphore, u64)],
    ) -> Result<()> {
        let state = self
            .texture
            .0
            .sparse
            .as_ref()
            .expect("prepared sparse image");
        let mut mappings = state.mappings.lock().unwrap_or_else(|e| e.into_inner());
        // Retain both previous and replacement allocations before native handoff.
        self.retained.extend(
            self.indices
                .iter()
                .filter_map(|&i| mappings[i].as_ref().map(|m| m.0.clone())),
        );
        if let Some(a) = &self.allocation {
            self.retained.push(a.clone());
        }
        let image_info = [vk::SparseImageMemoryBindInfo::default()
            .image(self.texture.0.image)
            .binds(&self.image_binds)];
        let opaque_info = [vk::SparseImageOpaqueMemoryBindInfo::default()
            .image(self.texture.0.image)
            .binds(&self.opaque_binds)];
        let ws: Vec<_> = waits.iter().map(|x| x.0).collect();
        let wv: Vec<_> = waits.iter().map(|x| x.1).collect();
        let ss: Vec<_> = signals.iter().map(|x| x.0).collect();
        let sv: Vec<_> = signals.iter().map(|x| x.1).collect();
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wv)
            .signal_semaphore_values(&sv);
        let mut info = vk::BindSparseInfo::default()
            .wait_semaphores(&ws)
            .signal_semaphores(&ss)
            .push_next(&mut timeline);
        if !self.image_binds.is_empty() {
            info = info.image_binds(&image_info);
        }
        if !self.opaque_binds.is_empty() {
            info = info.image_opaque_binds(&opaque_info);
        }
        // SAFETY: all ranges and memory types were preflighted; caller serializes queue access and retains this object.
        let submitted = unsafe {
            device
                .0
                .device
                .queue_bind_sparse(queue, &[info], vk::Fence::null())
        };
        let texture_id = Arc::as_ptr(&self.texture.0) as usize;
        if let Err(error) = submitted {
            // The driver may have consumed the operation. Reserve both old and
            // possible new intervals permanently and reject further binds of
            // this texture; the caller retains this prepared operation.
            state.uncertain.store(true, Ordering::Relaxed);
            for &i in &self.indices {
                if let Some((old, _)) = &mappings[i] {
                    for owner in old
                        .0
                        .owners
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .iter_mut()
                    {
                        if owner.texture == texture_id && owner.tile == i {
                            owner.uncertain = true;
                        }
                    }
                }
            }
            if let Some(allocation) = &self.allocation {
                let mut owners = allocation
                    .0
                    .owners
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                for (position, &tile) in self.indices.iter().enumerate() {
                    let start = self.allocation_offset + position as u64 * state.tiling.tile_bytes;
                    owners.push(Owner {
                        texture: texture_id,
                        tile,
                        start,
                        end: start + state.tiling.tile_bytes,
                        uncertain: true,
                    });
                }
            }
            return Err(vk_error(error));
        }
        for (position, &i) in self.indices.iter().enumerate() {
            if let Some((old, _)) = mappings[i].take() {
                old.0
                    .owners
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .retain(|o| o.texture != texture_id || o.tile != i);
            }
            if let Some(allocation) = &self.allocation {
                let start = self.allocation_offset + position as u64 * state.tiling.tile_bytes;
                allocation
                    .0
                    .owners
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(Owner {
                        texture: texture_id,
                        tile: i,
                        start,
                        end: start + state.tiling.tile_bytes,
                        uncertain: false,
                    });
                mappings[i] = Some((allocation.clone(), start));
            }
        }
        Ok(())
    }
}
