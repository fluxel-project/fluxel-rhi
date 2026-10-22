//! Sparse texture allocations and queue mappings.
use super::*;
use std::sync::Mutex;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

pub(crate) struct SparseAllocationState {
    pub(crate) heap: ID3D12Heap,
    pub(crate) tiles: u64,
}
pub(crate) struct SparseTextureState {
    pub(crate) tiling: api::SparseTextureTiling,
    pub(crate) mappings: Mutex<Vec<Option<Dx12Sparse>>>,
    pub(crate) standard_starts: Vec<u32>,
    pub(crate) packed_start: u32,
}
pub(crate) struct SparseRetained {
    pub(crate) texture: Dx12Texture,
    pub(crate) heaps: Vec<Dx12Sparse>,
}

pub(crate) fn validate_bind(bind: &api::SparseBind<Dx12>) -> api::Result<()> {
    // SAFETY: Core packet validation checked ownership; this borrow is read-only.
    let texture = unsafe { bind.resource.native_ref() };
    let state = texture.0.sparse.as_ref().ok_or(api::Error::Unsupported(
        "DX12 sparse bind requires a reserved texture",
    ))?;
    if bind.tiles.is_empty()
        || bind.allocation_offset % state.tiling.tile_bytes != 0
        || (bind.allocation.is_none() && bind.allocation_offset != 0)
    {
        return Err(api::Error::InvalidRange);
    }
    let mut total = 0u64;
    for tile in &bind.tiles {
        if tile.layer != 0 {
            return Err(api::Error::InvalidRange);
        }
        let n = if let Some(grid) = state.tiling.standard_mip_grids.get(tile.mip as usize) {
            if tile.origin[2] != 0
                || tile.count[2] != 1
                || tile.count[0] == 0
                || tile.count[1] == 0
                || tile.count[1] > u16::MAX as u32
                || tile.origin[0]
                    .checked_add(tile.count[0])
                    .is_none_or(|x| x > grid[0])
                || tile.origin[1]
                    .checked_add(tile.count[1])
                    .is_none_or(|y| y > grid[1])
            {
                return Err(api::Error::InvalidRange);
            }
            u64::from(tile.count[0]) * u64::from(tile.count[1])
        } else if let Some(tail) = state
            .tiling
            .packed_tail
            .filter(|tail| tail.first_mip == tile.mip)
        {
            if tile.origin != [0, 0, 0] || tile.count != [tail.tile_count, 1, 1] {
                return Err(api::Error::InvalidRange);
            }
            u64::from(tail.tile_count)
        } else {
            return Err(api::Error::InvalidRange);
        };
        total = total.checked_add(n).ok_or(api::Error::InvalidRange)?;
    }
    if total > u64::from(u32::MAX) {
        return Err(api::Error::InvalidRange);
    }
    if let Some(allocation) = &bind.allocation {
        // SAFETY: same-device ownership was validated by Core.
        let allocation = unsafe { allocation.native_ref() };
        let start = bind.allocation_offset / state.tiling.tile_bytes;
        if start > u32::MAX as u64
            || start
                .checked_add(total)
                .is_none_or(|end| end > allocation.0.tiles || end > u32::MAX as u64)
        {
            return Err(api::Error::InvalidRange);
        }
    }
    Ok(())
}

/// Issues a previously validated mapping while the submission and queue gates
/// are held. Returned ownership protects old and new heaps until a fence.
pub(crate) fn issue_bind(device: &Dx12Device, bind: &api::SparseBind<Dx12>) -> SparseRetained {
    // SAFETY: packet validation established both handles belong to this device.
    let texture = unsafe { bind.resource.native_ref() }.clone();
    let allocation = bind
        .allocation
        .as_ref()
        .map(|a| unsafe { a.native_ref() }.clone());
    let state = texture.0.sparse.as_ref().expect("validated sparse texture");
    let mut mappings = state.mappings.lock().unwrap_or_else(|p| p.into_inner());
    let mut retained = Vec::new();
    if let Some(heap) = &allocation {
        retained.push(heap.clone());
    }
    let mut offset = (bind.allocation_offset / state.tiling.tile_bytes) as u32;
    for tile in &bind.tiles {
        let standard = state.tiling.standard_mip_grids.get(tile.mip as usize);
        let count = tile.count[0] * tile.count[1];
        let coordinate = D3D12_TILED_RESOURCE_COORDINATE {
            X: tile.origin[0],
            Y: tile.origin[1],
            Z: 0,
            Subresource: tile.mip,
        };
        let region = D3D12_TILE_REGION_SIZE {
            NumTiles: count,
            UseBox: standard.is_some().into(),
            Width: tile.count[0],
            Height: tile.count[1] as u16,
            Depth: 1,
        };
        let flags = if allocation.is_some() {
            D3D12_TILE_RANGE_FLAG_NONE
        } else {
            D3D12_TILE_RANGE_FLAG_NULL
        };
        let heap = allocation.as_ref().map(|a| &a.0.heap);
        // SAFETY: all bounds, heap offset, and tile counts were preflighted;
        // queue access is serialized and native objects are retained below.
        unsafe {
            device.0.queue.UpdateTileMappings(
                &texture.0.resource,
                1,
                Some(&coordinate),
                Some(&region),
                heap,
                1,
                Some(&flags),
                Some(&offset),
                Some(&count),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        if let Some(grid) = standard {
            let start = state.standard_starts[tile.mip as usize];
            for y in tile.origin[1]..tile.origin[1] + tile.count[1] {
                for x in tile.origin[0]..tile.origin[0] + tile.count[0] {
                    let index = (start + y * grid[0] + x) as usize;
                    if let Some(old) = std::mem::replace(&mut mappings[index], allocation.clone()) {
                        retained.push(old);
                    }
                }
            }
        } else {
            for index in state.packed_start..state.packed_start + count {
                if let Some(old) =
                    std::mem::replace(&mut mappings[index as usize], allocation.clone())
                {
                    retained.push(old);
                }
            }
        }
        offset += count;
    }
    drop(mappings);
    SparseRetained {
        texture,
        heaps: retained,
    }
}

impl api::SparseBackend for Dx12 {
    fn create_sparse_texture(
        device: &Dx12Device,
        desc: &api::TextureDesc,
    ) -> api::Result<Dx12Texture> {
        if !device.0.sparse_supported {
            return Err(api::Error::Unsupported("DX12 tiled resources"));
        }
        if desc.dimension != api::TextureDimension::D2
            || desc.format != api::TextureFormat::Rgba8Unorm
            || desc.extent[0] == 0
            || desc.extent[1] == 0
            || desc.extent[0] > 16384
            || desc.extent[1] > 16384
            || desc.extent[2] != 1
            || desc.array_layers != 1
            || desc.mip_levels == 0
            || desc.mip_levels > 1 + desc.extent[0].max(desc.extent[1]).ilog2()
            || desc.samples != 1
            || !desc.view_formats.is_empty()
            || desc.usage.0
                & !(api::TextureUsage::COPY_SRC.0
                    | api::TextureUsage::COPY_DST.0
                    | api::TextureUsage::TEXTURE_BINDING.0)
                != 0
        {
            return Err(api::Error::Unsupported("DX12 sparse RGBA8 D2 subset"));
        }
        let native = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: u64::from(desc.extent[0]),
            Height: desc.extent[1],
            DepthOrArraySize: 1,
            MipLevels: desc.mip_levels as u16,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut resource = None;
        // SAFETY: descriptor is fully initialized and result storage is valid.
        unsafe {
            device.0.native.CreateReservedResource(
                &native,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
        }
        .map_err(native_error)?;
        let resource: ID3D12Resource = resource.ok_or(api::Error::OutOfMemory)?;
        let mut count = 0;
        let mut packed = D3D12_PACKED_MIP_INFO::default();
        let mut shape = D3D12_TILE_SHAPE::default();
        let mut sub_count = desc.mip_levels;
        let mut sub = vec![D3D12_SUBRESOURCE_TILING::default(); sub_count as usize];
        // SAFETY: outputs match the query and sub contains sub_count elements.
        unsafe {
            device.0.native.GetResourceTiling(
                &resource,
                Some(&mut count),
                Some(&mut packed),
                Some(&mut shape),
                Some(&mut sub_count),
                0,
                sub.as_mut_ptr(),
            );
        }
        let standard_count = u32::from(packed.NumStandardMips);
        if sub_count != desc.mip_levels
            || standard_count + u32::from(packed.NumPackedMips) != desc.mip_levels
            || count == 0
            || shape.WidthInTexels == 0
            || shape.HeightInTexels == 0
            || shape.DepthInTexels != 1
        {
            return Err(api::Error::Unsupported("DX12 sparse tiling metadata"));
        }
        let mut grids = Vec::new();
        let mut starts = Vec::new();
        for item in sub.iter().take(standard_count as usize) {
            let tiles = u64::from(item.WidthInTiles)
                * u64::from(item.HeightInTiles)
                * u64::from(item.DepthInTiles);
            if item.WidthInTiles == 0
                || item.HeightInTiles == 0
                || item.DepthInTiles != 1
                || u64::from(item.StartTileIndexInOverallResource) + tiles > u64::from(count)
            {
                return Err(api::Error::Unsupported("DX12 sparse standard mip tiling"));
            }
            grids.push([
                item.WidthInTiles,
                u32::from(item.HeightInTiles),
                u32::from(item.DepthInTiles),
            ]);
            starts.push(item.StartTileIndexInOverallResource);
        }
        let packed_tail = if packed.NumPackedMips == 0 {
            None
        } else {
            if packed.NumTilesForPackedMips == 0
                || packed
                    .StartTileIndexInOverallResource
                    .checked_add(packed.NumTilesForPackedMips)
                    .is_none_or(|end| end > count)
            {
                return Err(api::Error::Unsupported("DX12 sparse packed mip tiling"));
            }
            Some(api::SparsePackedMipTail {
                first_mip: standard_count,
                mip_count: u32::from(packed.NumPackedMips),
                tile_count: packed.NumTilesForPackedMips,
            })
        };
        let tiling = api::SparseTextureTiling {
            tile_extent: [
                shape.WidthInTexels,
                shape.HeightInTexels,
                shape.DepthInTexels,
            ],
            tile_grid: grids.first().copied().unwrap_or([0, 0, 0]),
            tile_bytes: 65536,
            standard_mip_grids: grids,
            packed_tail,
        };
        Ok(Dx12Texture(Arc::new(super::texture::TextureState {
            resource,
            desc: desc.clone(),
            sparse: Some(SparseTextureState {
                tiling,
                mappings: Mutex::new(vec![None; count as usize]),
                standard_starts: starts,
                packed_start: packed.StartTileIndexInOverallResource,
            }),
        })))
    }
    fn sparse_texture_tiling(
        _: &Dx12Device,
        texture: &Dx12Texture,
    ) -> api::Result<api::SparseTextureTiling> {
        texture
            .0
            .sparse
            .as_ref()
            .map(|s| s.tiling.clone())
            .ok_or(api::Error::Unsupported("DX12 texture is not sparse"))
    }
    fn create_sparse_allocation(device: &Dx12Device, size: u64) -> api::Result<Dx12Sparse> {
        if !device.0.sparse_supported {
            return Err(api::Error::Unsupported("DX12 tiled resources"));
        }
        if size == 0 || size % 65536 != 0 {
            return Err(api::Error::InvalidRange);
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: size,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let mut heap = None;
        // SAFETY: descriptor and output pointer are valid for this synchronous call.
        unsafe { device.0.native.CreateHeap(&desc, &mut heap) }.map_err(native_error)?;
        Ok(Dx12Sparse(Arc::new(SparseAllocationState {
            heap: heap.ok_or(api::Error::OutOfMemory)?,
            tiles: size / 65536,
        })))
    }
}
