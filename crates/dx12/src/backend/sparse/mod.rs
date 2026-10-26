//! Sparse texture allocations and queue mappings.
use super::*;
#[path = "array.rs"]
mod array;
use std::sync::Mutex;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R32_FLOAT,
    DXGI_FORMAT_R32_TYPELESS,
};

pub(crate) struct SparseAllocationState {
    pub(crate) heap: ID3D12Heap,
    pub(crate) tiles: u64,
    render_target: bool,
}
pub(crate) struct SparseTextureState {
    pub(crate) tiling: api::SparseTextureTiling,
    pub(crate) mappings: Mutex<Vec<Option<Dx12Sparse>>>,
    pub(crate) standard_starts: Vec<u32>,
    pub(crate) packed_start: u32,
    pub(crate) layer_stride: u32,
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
        if tile
            .aspect
            .is_some_and(|a| a != state.tiling.aspects[0].aspect)
        {
            return Err(api::Error::InvalidRange);
        }
        if tile.layer >= texture.0.desc.array_layers {
            return Err(api::Error::InvalidRange);
        }
        let n = if let Some(grid) = state.tiling.standard_mip_grids.get(tile.mip as usize) {
            if tile.count[2] == 0
                || tile.count[2] > u16::MAX as u32
                || tile.origin[2]
                    .checked_add(tile.count[2])
                    .is_none_or(|z| z > grid[2])
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
            u64::from(tile.count[0]) * u64::from(tile.count[1]) * u64::from(tile.count[2])
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
        if allocation.0.render_target
            != (super::texture::is_depth(texture.0.desc.format)
                || texture
                    .0
                    .desc
                    .usage
                    .contains(api::TextureUsage::RENDER_ATTACHMENT))
        {
            return Err(api::Error::InvalidState);
        }
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
        let count = tile.count[0] * tile.count[1] * tile.count[2];
        let coordinate = D3D12_TILED_RESOURCE_COORDINATE {
            X: tile.origin[0],
            Y: tile.origin[1],
            Z: tile.origin[2],
            Subresource: tile.mip + tile.layer * texture.0.desc.mip_levels,
        };
        let region = D3D12_TILE_REGION_SIZE {
            NumTiles: count,
            UseBox: standard.is_some().into(),
            Width: tile.count[0],
            Height: tile.count[1] as u16,
            Depth: tile.count[2] as u16,
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
            let start = state.standard_starts[tile.mip as usize] + tile.layer * state.layer_stride;
            for z in tile.origin[2]..tile.origin[2] + tile.count[2] {
                for y in tile.origin[1]..tile.origin[1] + tile.count[1] {
                    for x in tile.origin[0]..tile.origin[0] + tile.count[0] {
                        let index = (start + (z * grid[1] + y) * grid[0] + x) as usize;
                        if let Some(old) =
                            std::mem::replace(&mut mappings[index], allocation.clone())
                        {
                            retained.push(old);
                        }
                    }
                }
            }
        } else {
            let start = state.packed_start + tile.layer * state.layer_stride;
            for index in start..start + count {
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

/// Verifies that `GetResourceTiling` repeated the first-slice layout for
/// every array slice and marked every packed mip with its sentinel value.
fn validate_array_tiling_metadata(
    sub: &[D3D12_SUBRESOURCE_TILING],
    mip_levels: u32,
    array_layers: u32,
    standard_mips: u32,
    layer_stride: u32,
) -> api::Result<()> {
    let expected = usize::try_from(
        mip_levels
            .checked_mul(array_layers)
            .ok_or(api::Error::InvalidRange)?,
    )
    .map_err(|_| api::Error::InvalidRange)?;
    if sub.len() != expected || standard_mips > mip_levels {
        return Err(api::Error::Unsupported("DX12 sparse array tiling metadata"));
    }
    for layer in 0..array_layers {
        let base = usize::try_from(layer * mip_levels).map_err(|_| api::Error::InvalidRange)?;
        for mip in 0..mip_levels {
            let item = sub[base + mip as usize];
            if mip < standard_mips {
                let first = sub[mip as usize];
                if item.WidthInTiles != first.WidthInTiles
                    || item.HeightInTiles != first.HeightInTiles
                    || item.DepthInTiles != first.DepthInTiles
                    || item.StartTileIndexInOverallResource
                        != first.StartTileIndexInOverallResource + layer * layer_stride
                {
                    return Err(api::Error::Unsupported(
                        "DX12 sparse array standard mip metadata",
                    ));
                }
            } else if item.StartTileIndexInOverallResource != D3D12_PACKED_TILE {
                return Err(api::Error::Unsupported(
                    "DX12 sparse array packed mip metadata",
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/backend/sparse.rs"]
mod tests;

impl fluxel_rhi_core::backend::SparseBackend for Dx12 {
    fn create_sparse_texture(
        device: &Dx12Device,
        desc: &api::TextureDesc,
    ) -> api::Result<Dx12Texture> {
        if !device.0.sparse_supported {
            return Err(api::Error::Unsupported("DX12 tiled resources"));
        }
        let is_3d = desc.dimension == api::TextureDimension::D3;
        if !matches!(
            desc.dimension,
            api::TextureDimension::D2 | api::TextureDimension::D3
        ) || desc.extent[0] == 0
            || desc.extent[1] == 0
            || desc.extent[0] > 16384
            || desc.extent[1] > 16384
            || desc.extent[2] == 0
            || (is_3d && (desc.extent[2] > 2048 || desc.array_layers != 1))
            || (!is_3d && desc.extent[2] != 1)
            || desc.array_layers == 0
            || desc.array_layers > 2048
            || desc.mip_levels == 0
            || desc.mip_levels
                > 1 + desc.extent[0]
                    .max(desc.extent[1])
                    .max(desc.extent[2])
                    .ilog2()
            || desc.samples != 1
            || !desc.view_formats.is_empty()
            || desc.usage.0
                & !(api::TextureUsage::COPY_SRC.0
                    | api::TextureUsage::COPY_DST.0
                    | api::TextureUsage::TEXTURE_BINDING.0
                    | api::TextureUsage::STORAGE_BINDING.0
                    | api::TextureUsage::RENDER_ATTACHMENT.0)
                != 0
        {
            return Err(api::Error::Unsupported("DX12 sparse D2 texture descriptor"));
        }
        let block_bytes = super::texture::bc_block_bytes(desc.format);
        let depth = super::texture::is_depth(desc.format);
        if super::texture::color_texel_bytes(desc.format).is_none()
            && block_bytes.is_none()
            && !depth
        {
            return Err(api::Error::Unsupported("DX12 sparse texture format"));
        }
        if depth && desc.usage.contains(api::TextureUsage::STORAGE_BINDING) {
            return Err(api::Error::Unsupported("DX12 sparse depth storage binding"));
        }
        if block_bytes.is_some() && desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT) {
            return Err(api::Error::Unsupported("DX12 sparse compressed attachment"));
        }
        if block_bytes.is_some()
            && (desc.extent[0] % 4 != 0
                || desc.extent[1] % 4 != 0
                || desc.usage.contains(api::TextureUsage::STORAGE_BINDING))
        {
            return Err(api::Error::Unsupported("DX12 sparse BC texture layout"));
        }
        // Only validated DXGI-mappable formats may reach the infallible
        // private mapping. Public ETC/ASTC descriptors must return a
        // structured unsupported result rather than panic at this boundary.
        let format = super::texture::dxgi_format(desc.format);
        let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: format,
            ..Default::default()
        };
        // SAFETY: the feature query uses the corresponding initialized structure.
        let supported = unsafe {
            device.0.native.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                &mut support as *mut _ as *mut _,
                std::mem::size_of_val(&support) as u32,
            )
        }
        .is_ok()
            && support.Support1.contains(if is_3d {
                D3D12_FORMAT_SUPPORT1_TEXTURE3D
            } else {
                D3D12_FORMAT_SUPPORT1_TEXTURE2D
            })
            && support.Support2.contains(D3D12_FORMAT_SUPPORT2_TILED)
            && (!desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
                || support.Support1.contains(if depth {
                    D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL
                } else {
                    D3D12_FORMAT_SUPPORT1_RENDER_TARGET
                }))
            && (!desc.usage.contains(api::TextureUsage::STORAGE_BINDING)
                || (support
                    .Support1
                    .contains(D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)
                    && support
                        .Support2
                        .contains(D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD)
                    && support
                        .Support2
                        .contains(D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)))
            && (!desc.usage.contains(api::TextureUsage::TEXTURE_BINDING)
                || depth
                || support
                    .Support1
                    .contains(if super::texture::integer_color(desc.format) {
                        D3D12_FORMAT_SUPPORT1_SHADER_LOAD
                    } else {
                        D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE
                    }));
        if !supported {
            return Err(api::Error::Unsupported("DX12 sparse tiled format support"));
        }
        if depth && desc.usage.contains(api::TextureUsage::TEXTURE_BINDING) {
            let mut srv_support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
                Format: if desc.format == api::TextureFormat::Depth16Unorm {
                    DXGI_FORMAT_R16_UNORM
                } else {
                    DXGI_FORMAT_R32_FLOAT
                },
                ..Default::default()
            };
            // SAFETY: this feature query writes the matching initialized structure.
            let srv_ok = unsafe {
                device.0.native.CheckFeatureSupport(
                    D3D12_FEATURE_FORMAT_SUPPORT,
                    &mut srv_support as *mut _ as *mut _,
                    std::mem::size_of_val(&srv_support) as u32,
                )
            }
            .is_ok()
                && srv_support
                    .Support1
                    .contains(D3D12_FORMAT_SUPPORT1_SHADER_LOAD);
            if !srv_ok {
                return Err(api::Error::Unsupported("DX12 sparse depth sample format"));
            }
        }
        let native = D3D12_RESOURCE_DESC {
            Dimension: if is_3d {
                D3D12_RESOURCE_DIMENSION_TEXTURE3D
            } else {
                D3D12_RESOURCE_DIMENSION_TEXTURE2D
            },
            Alignment: 0,
            Width: u64::from(desc.extent[0]),
            Height: desc.extent[1],
            DepthOrArraySize: if is_3d {
                desc.extent[2] as u16
            } else {
                desc.array_layers as u16
            },
            MipLevels: desc.mip_levels as u16,
            Format: if depth && desc.usage.contains(api::TextureUsage::TEXTURE_BINDING) {
                if desc.format == api::TextureFormat::Depth16Unorm {
                    DXGI_FORMAT_R16_TYPELESS
                } else {
                    DXGI_FORMAT_R32_TYPELESS
                }
            } else {
                format
            },
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
            Flags: if depth {
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
            } else {
                let mut flags = D3D12_RESOURCE_FLAG_NONE;
                if desc.usage.contains(api::TextureUsage::STORAGE_BINDING) {
                    flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                }
                if desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT) {
                    flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
                }
                flags
            },
        };
        if !is_3d {
            array::validate_array_mips(device, desc, native, depth)?;
        }
        let mut resource = None;
        // SAFETY: descriptor is fully initialized and result storage is valid.
        unsafe {
            device.0.native.CreateReservedResource(
                &native,
                if depth {
                    D3D12_RESOURCE_STATE_DEPTH_WRITE
                } else {
                    D3D12_RESOURCE_STATE_COMMON
                },
                None,
                &mut resource,
            )
        }
        .map_err(|error| {
            // Pre-tier-4 devices reject arrays whose mip chain enters a packed tail.
            if desc.array_layers > 1
                && desc.mip_levels > 1
                && error.code() == windows::core::HRESULT(0x80070057u32 as i32)
            {
                api::Error::Unsupported("DX12 sparse array mip layout")
            } else {
                native_error(error)
            }
        })?;
        let resource: ID3D12Resource = resource.ok_or(api::Error::OutOfMemory)?;
        let mut count = 0;
        let mut packed = D3D12_PACKED_MIP_INFO::default();
        let mut shape = D3D12_TILE_SHAPE::default();
        let expected_subresources = desc
            .mip_levels
            .checked_mul(desc.array_layers)
            .ok_or(api::Error::InvalidRange)?;
        let mut sub_count = expected_subresources;
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
        if count % desc.array_layers != 0 {
            return Err(api::Error::Unsupported("DX12 sparse array tiling"));
        }
        let layer_stride = count / desc.array_layers;
        let packed_mips = u32::from(packed.NumPackedMips);
        let packed_tiles = packed.NumTilesForPackedMips;
        let packed_arrays_allowed =
            desc.array_layers == 1 || array::supports_packed_array_tails(device)?;
        if sub_count != expected_subresources
            || standard_count + packed_mips != desc.mip_levels
            || (!packed_arrays_allowed && desc.array_layers > 1 && packed_mips != 0)
            || count == 0
            || shape.WidthInTexels == 0
            || shape.HeightInTexels == 0
            || shape.DepthInTexels == 0
            || (!is_3d && shape.DepthInTexels != 1)
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
                || item.DepthInTiles == 0
                || (!is_3d && item.DepthInTiles != 1)
                || u64::from(item.StartTileIndexInOverallResource) + tiles > u64::from(layer_stride)
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
        // The packed-mip descriptor is per array slice. Tier 4 permits these
        // tails for texture arrays; older tiers were rejected before creation.
        validate_array_tiling_metadata(
            &sub,
            desc.mip_levels,
            desc.array_layers,
            standard_count,
            layer_stride,
        )?;
        let packed_tail = if packed_mips == 0 {
            None
        } else {
            if packed.NumTilesForPackedMips == 0
                || packed
                    .StartTileIndexInOverallResource
                    .checked_add(packed_tiles)
                    .is_none_or(|end| end > layer_stride)
            {
                return Err(api::Error::Unsupported("DX12 sparse packed mip tiling"));
            }
            Some(api::SparsePackedMipTail {
                first_mip: standard_count,
                mip_count: packed_mips,
                tile_count: packed_tiles,
            })
        };
        let tiling = api::SparseTextureTiling::from_aspects(vec![api::SparseAspectTiling {
            aspect: if depth {
                api::TextureAspects::DEPTH
            } else {
                api::TextureAspects::COLOR
            },
            packed_tail_shared_by_layers: false,
            tile_extent: [
                shape.WidthInTexels,
                shape.HeightInTexels,
                shape.DepthInTexels,
            ],
            tile_grid: grids.first().copied().unwrap_or([0, 0, 0]),
            tile_bytes: 65536,
            standard_mip_grids: grids,
            packed_tail,
        }])
        .expect("one sparse aspect");
        Ok(Dx12Texture(Arc::new(super::texture::TextureState {
            resource,
            default_allocation: None,
            backing_heap: None,
            desc: desc.clone(),
            sparse: Some(SparseTextureState {
                tiling,
                mappings: Mutex::new(vec![None; count as usize]),
                standard_starts: starts,
                packed_start: packed.StartTileIndexInOverallResource,
                layer_stride,
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
        allocate(device, size, false)
    }
    fn create_sparse_allocation_for(
        device: &Dx12Device,
        texture: &Dx12Texture,
        size: u64,
    ) -> api::Result<Dx12Sparse> {
        if texture.0.sparse.is_none() {
            return Err(api::Error::Unsupported("DX12 texture is not sparse"));
        }
        allocate(
            device,
            size,
            super::texture::is_depth(texture.0.desc.format)
                || texture
                    .0
                    .desc
                    .usage
                    .contains(api::TextureUsage::RENDER_ATTACHMENT),
        )
    }
}

fn allocate(device: &Dx12Device, size: u64, render_target: bool) -> api::Result<Dx12Sparse> {
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
        Flags: if render_target {
            D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES
        } else {
            D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES
        },
    };
    let mut heap = None;
    // SAFETY: descriptor and output pointer are valid for this synchronous call.
    unsafe { device.0.native.CreateHeap(&desc, &mut heap) }.map_err(native_error)?;
    Ok(Dx12Sparse(Arc::new(SparseAllocationState {
        heap: heap.ok_or(api::Error::OutOfMemory)?,
        tiles: size / 65536,
        render_target,
    })))
}
