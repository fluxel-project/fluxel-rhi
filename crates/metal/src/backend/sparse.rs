//! Placement sparse texture mappings for the restricted Metal 4 path.
//!
//! Core requires an exact allocation and byte offset per mapping. Metal 3's
//! resource-state encoder allocates tiles from the texture's own sparse heap
//! without selecting an allocation offset. Metal 4 placement heaps can express
//! that offset; submission must establish ordering and completion separately.
use super::{Metal, MetalDevice};
use fluxel_rhi_core::api;
use fluxel_rhi_core::api::{Error, Result, SparseBackend};
#[cfg(target_vendor = "apple")]
use std::sync::{Arc, Mutex};

#[cfg(any(target_vendor = "apple", test))]
mod validation;

/// Placement heap backing a set of sparse texture tiles.
#[derive(Clone)]
pub struct MetalSparseAllocation {
    #[cfg(target_vendor = "apple")]
    pub(super) heap: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLHeap>>,
    pub(super) size: u64,
}

pub(super) struct SparseState {
    tiling: api::SparseTextureTiling,
    // The predicted mapping frontier is updated in native submission order.
    #[cfg(target_vendor = "apple")]
    mappings: Mutex<validation::MappingState<MetalSparseAllocation>>,
}

impl SparseBackend for Metal {
    fn create_sparse_texture(
        device: &MetalDevice,
        desc: &api::TextureDesc,
    ) -> Result<super::MetalTexture> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLDevice as _, MTLSparsePageSize, MTLTexture as _, MTLTextureType};
            if !placement_supported(device) {
                return Err(Error::Unsupported(
                    "Metal placement sparse requires Metal 4",
                ));
            }
            if desc.dimension != api::TextureDimension::D2
                || desc.format != api::TextureFormat::Rgba8Unorm
                || desc.array_layers != 1
                || desc.samples != 1
                || desc.extent[2] != 1
            {
                return Err(Error::Unsupported("Metal sparse texture descriptor"));
            }
            let mut texture = super::texture::create_sparse_texture(device, desc)?;
            // SAFETY: descriptor is a supported 2D RGBA8 single-sampled texture.
            let tile = unsafe {
                device
                    ._state
                    ._device
                    .sparseTileSizeWithTextureType_pixelFormat_sampleCount_sparsePageSize(
                        MTLTextureType::Type2D,
                        super::texture::metal_format(desc.format).unwrap(),
                        1,
                        MTLSparsePageSize::Size64,
                    )
            };
            let bytes = device
                ._state
                ._device
                .sparseTileSizeInBytesForSparsePageSize(MTLSparsePageSize::Size64);
            let first = texture
                .raw
                .firstMipmapInTail()
                .min(desc.mip_levels as usize);
            let tail_bytes = texture.raw.tailSizeInBytes();
            if tile.width == 0
                || tile.height == 0
                || tile.depth != 1
                || bytes == 0
                || tail_bytes % bytes != 0
            {
                return Err(Error::Unsupported("Metal sparse tile geometry"));
            }
            let width = u32::try_from(tile.width).map_err(|_| Error::InvalidRange)?;
            let height = u32::try_from(tile.height).map_err(|_| Error::InvalidRange)?;
            let standard = u32::try_from(first).map_err(|_| Error::InvalidRange)?;
            let grids: Vec<_> = (0..standard)
                .map(|m| {
                    [
                        (desc.extent[0] >> m).max(1).div_ceil(width),
                        (desc.extent[1] >> m).max(1).div_ceil(height),
                        1,
                    ]
                })
                .collect();
            let tail = if standard < desc.mip_levels {
                let count = u32::try_from(tail_bytes / bytes).map_err(|_| Error::InvalidRange)?;
                if count == 0 {
                    return Err(Error::Unsupported("Metal sparse empty mip tail"));
                }
                Some(api::SparsePackedMipTail {
                    first_mip: standard,
                    mip_count: desc.mip_levels - standard,
                    tile_count: count,
                })
            } else {
                None
            };
            texture.sparse = Some(Arc::new(SparseState {
                tiling: api::SparseTextureTiling::from_aspects(vec![api::SparseAspectTiling {
                    aspect: api::TextureAspects::COLOR,
                    packed_tail_shared_by_layers: false,
                    tile_extent: [width, height, 1],
                    tile_grid: grids.first().copied().unwrap_or([0, 0, 0]),
                    tile_bytes: bytes as u64,
                    standard_mip_grids: grids,
                    packed_tail: tail,
                }])
                .expect("one sparse aspect"),
                mappings: Mutex::new(validation::MappingState::new()),
            }));
            Ok(texture)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, desc);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn sparse_texture_tiling(
        _: &MetalDevice,
        texture: &super::MetalTexture,
    ) -> Result<api::SparseTextureTiling> {
        texture
            .sparse
            .as_ref()
            .map(|s| s.tiling.clone())
            .ok_or(Error::InvalidState)
    }
    fn create_sparse_allocation(device: &MetalDevice, size: u64) -> Result<MetalSparseAllocation> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLDevice as _, MTLHeapDescriptor, MTLHeapType, MTLSparsePageSize};
            if !placement_supported(device) {
                return Err(Error::Unsupported(
                    "Metal placement sparse requires Metal 4",
                ));
            }
            let size = usize::try_from(size).map_err(|_| Error::InvalidRange)?;
            let page = device
                ._state
                ._device
                .sparseTileSizeInBytesForSparsePageSize(MTLSparsePageSize::Size64);
            if page == 0 || size % page != 0 {
                return Err(Error::InvalidRange);
            }
            let desc = MTLHeapDescriptor::new();
            desc.setType(MTLHeapType::Placement);
            desc.setSize(size);
            desc.setMaxCompatiblePlacementSparsePageSize(MTLSparsePageSize::Size64);
            let heap = device
                ._state
                ._device
                .newHeapWithDescriptor(&desc)
                .ok_or(Error::OutOfMemory)?;
            Ok(MetalSparseAllocation {
                heap,
                size: size as u64,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, size);
            Err(Error::Unsupported("Metal requires an Apple target"))
        }
    }
}

#[cfg(target_vendor = "apple")]
pub(super) struct PreparedSparseBind {
    texture: super::MetalTexture,
    allocation: Option<MetalSparseAllocation>,
    operations: Vec<objc2_metal::MTL4UpdateSparseTextureMappingOperation>,
    regions: Vec<validation::Region>,
    replaced: Mutex<Vec<MetalSparseAllocation>>,
}

#[cfg(target_vendor = "apple")]
pub(super) fn prepare_bind(bind: &api::SparseBind<Metal>) -> Result<PreparedSparseBind> {
    use objc2_metal::{
        MTL4UpdateSparseTextureMappingOperation as Op, MTLOrigin, MTLRegion, MTLSize,
        MTLSparseTextureMappingMode,
    };
    // SAFETY: Core checks both object identities when binding the submission.
    let texture = unsafe { bind.resource.native_ref() }.clone();
    let allocation = bind
        .allocation
        .as_ref()
        .map(|a| unsafe { a.native_ref() }.clone());
    let state = texture.sparse.as_ref().ok_or(Error::InvalidState)?;
    let regions = validation::validate_regions(
        &state.tiling,
        &bind.tiles,
        allocation.as_ref().map(|a| a.size),
        bind.allocation_offset,
    )?;
    let mode = if allocation.is_some() {
        MTLSparseTextureMappingMode::Map
    } else {
        MTLSparseTextureMappingMode::Unmap
    };
    let operations = regions
        .iter()
        .map(|r| {
            Ok(Op {
                mode,
                textureLevel: r.mip as usize,
                textureSlice: 0,
                textureRegion: MTLRegion {
                    origin: MTLOrigin {
                        x: r.origin[0] as usize,
                        y: r.origin[1] as usize,
                        z: 0,
                    },
                    size: MTLSize {
                        width: r.count[0] as usize,
                        height: r.count[1] as usize,
                        depth: 1,
                    },
                },
                heapOffset: usize::try_from(r.heap_tile_offset).map_err(|_| Error::InvalidRange)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PreparedSparseBind {
        texture,
        allocation,
        operations,
        regions,
        replaced: Mutex::new(Vec::new()),
    })
}

#[cfg(target_vendor = "apple")]
impl PreparedSparseBind {
    /// Enqueue after all validation and waits. The caller must retain `self`
    /// until a later GPU completion marker and must synchronize ResourceState.
    pub(super) fn enqueue(
        &self,
        queue: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4CommandQueue>,
    ) {
        use objc2_metal::MTL4CommandQueue as _;
        let old = self
            .texture
            .sparse
            .as_ref()
            .unwrap()
            .mappings
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .replace(&self.regions, self.allocation.as_ref());
        self.replaced
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .extend(old);
        // SAFETY: prepare_bind validated all bounds and lifetimes. The caller
        // provides the queue dependency and keeps self alive through completion.
        unsafe {
            queue.updateTextureMappings_heap_operations_count(
                &self.texture.raw,
                self.allocation.as_ref().map(|a| &*a.heap),
                std::ptr::NonNull::new(self.operations.as_ptr().cast_mut())
                    .expect("nonempty operations"),
                self.operations.len(),
            );
        }
    }
}

#[cfg(target_vendor = "apple")]
pub(super) fn placement_supported(device: &MetalDevice) -> bool {
    use objc2::{msg_send, sel};
    let raw = &*device._state._device;
    // SAFETY: Objective-C respondsToSelector is defined on NSObject; the
    // newer selector is only invoked after confirming its presence.
    unsafe {
        let responds: bool = msg_send![raw, respondsToSelector: sel!(supportsPlacementSparse)];
        if !responds {
            return false;
        }
        msg_send![raw, supportsPlacementSparse]
    }
}
