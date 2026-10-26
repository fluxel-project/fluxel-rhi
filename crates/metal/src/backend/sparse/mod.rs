//! Placement sparse texture mappings through Metal 4.
//!
//! Core requires an exact allocation and byte offset per mapping. Metal 3's
//! resource-state encoder allocates tiles from the texture's own sparse heap
//! without selecting an allocation offset. Metal 4 placement heaps can express
//! that offset; submission must establish ordering and completion separately.
use super::{Metal, MetalDevice};
use fluxel_rhi_core::api;
use fluxel_rhi_core::api::{Error, Result};
use fluxel_rhi_core::backend::SparseBackend;
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
            use api::TextureFormat::*;
            use objc2_metal::MTLGPUFamily;
            let bc = matches!(
                desc.format,
                Bc1RgbaUnorm
                    | Bc1RgbaUnormSrgb
                    | Bc2RgbaUnorm
                    | Bc2RgbaUnormSrgb
                    | Bc3RgbaUnorm
                    | Bc3RgbaUnormSrgb
                    | Bc4RUnorm
                    | Bc4RSnorm
                    | Bc5RgUnorm
                    | Bc5RgSnorm
                    | Bc6hRgbUfloat
                    | Bc6hRgbFloat
                    | Bc7RgbaUnorm
                    | Bc7RgbaUnormSrgb
            );
            // Apple marks ordinary color formats sparse starting with Apple6,
            // and BC on Apple9 or Apple7/8 devices with BC support.
            // Apple7 permits depth/stencil only on placement sparse textures.
            let ordinary = matches!(
                desc.format,
                R8Unorm
                    | R8Snorm
                    | R8Uint
                    | R8Sint
                    | Rg8Unorm
                    | Rg8Snorm
                    | Rg8Uint
                    | Rg8Sint
                    | Rgba8Unorm
                    | Rgba8UnormSrgb
                    | Rgba8Snorm
                    | Rgba8Uint
                    | Rgba8Sint
                    | Bgra8Unorm
                    | Bgra8UnormSrgb
                    | R16Unorm
                    | R16Snorm
                    | R16Uint
                    | R16Sint
                    | R16Float
                    | Rg16Unorm
                    | Rg16Snorm
                    | Rg16Uint
                    | Rg16Sint
                    | Rg16Float
                    | Rgba16Unorm
                    | Rgba16Snorm
                    | Rgba16Uint
                    | Rgba16Sint
                    | Rgba16Float
                    | R32Uint
                    | R32Sint
                    | R32Float
                    | Rg32Uint
                    | Rg32Sint
                    | Rg32Float
                    | Rgba32Uint
                    | Rgba32Sint
                    | Rgba32Float
                    | Rgb10a2Unorm
                    | Rgb10a2Uint
                    | Rg11b10Ufloat
                    | Rgb9e5Ufloat
                    | Etc2Rgb8Unorm
                    | Etc2Rgb8UnormSrgb
                    | Etc2Rgba8Unorm
                    | Etc2Rgba8UnormSrgb
                    | Etc2Rgb8A1Unorm
                    | Etc2Rgb8A1UnormSrgb
                    | EacR11Unorm
                    | EacR11Snorm
                    | EacRg11Unorm
                    | EacRg11Snorm
                    | Astc { .. }
            );
            let depth = matches!(desc.format, Depth16Unorm | Depth32Float | Stencil8);
            let native = &device._state._device;
            use objc2::runtime::NSObjectProtocol as _;
            let bc_sparse = native.supportsFamily(MTLGPUFamily::Apple9)
                || (native.supportsFamily(MTLGPUFamily::Apple7)
                    && native.respondsToSelector(objc2::sel!(supportsBCTextureCompression))
                    && native.supportsBCTextureCompression());
            if desc.samples != 1
                || !matches!(
                    desc.dimension,
                    api::TextureDimension::D2 | api::TextureDimension::D3
                )
                || !(ordinary && native.supportsFamily(MTLGPUFamily::Apple6)
                    || bc && bc_sparse
                    || depth && native.supportsFamily(MTLGPUFamily::Apple7))
            {
                return Err(Error::Unsupported(
                    "Metal device lacks sparse format or texture dimension",
                ));
            }
            let mut texture = super::texture::create_sparse_texture(device, desc)?;
            let texture_type = match (desc.dimension, desc.array_layers > 1) {
                (api::TextureDimension::D2, false) => MTLTextureType::Type2D,
                (api::TextureDimension::D2, true) => MTLTextureType::Type2DArray,
                (api::TextureDimension::D3, false) => MTLTextureType::Type3D,
                _ => return Err(Error::InvalidRange),
            };
            // SAFETY: the family and format support were checked, and creation
            // validated the texture type and sample count.
            let tile = unsafe {
                native.sparseTileSizeWithTextureType_pixelFormat_sampleCount_sparsePageSize(
                    texture_type,
                    super::texture::metal_format(desc.format).ok_or(Error::InvalidRange)?,
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
            let mut tail_bytes = texture.raw.tailSizeInBytes();
            if desc.array_layers > 1 && first < desc.mip_levels as usize {
                // Metal 4 maps a specific array slice. Query the corresponding
                // single-slice sparse texture instead of interpreting the
                // array-wide tailSizeInBytes as a per-slice allocation size.
                let mut single_desc = desc.clone();
                single_desc.array_layers = 1;
                let single = super::texture::create_sparse_texture(device, &single_desc)?;
                let single_first = single.raw.firstMipmapInTail().min(desc.mip_levels as usize);
                let single_tail = single.raw.tailSizeInBytes();
                // SAFETY: the same validated format and sample count are used
                // for a single 2D slice of this texture.
                let single_tile = unsafe {
                    native.sparseTileSizeWithTextureType_pixelFormat_sampleCount_sparsePageSize(
                        MTLTextureType::Type2D,
                        super::texture::metal_format(desc.format).ok_or(Error::InvalidRange)?,
                        1,
                        MTLSparsePageSize::Size64,
                    )
                };
                if single_first != first
                    || single_tile.width != tile.width
                    || single_tile.height != tile.height
                    || single_tile.depth != tile.depth
                    || (tail_bytes != single_tail
                        && single_tail.checked_mul(desc.array_layers as usize) != Some(tail_bytes))
                {
                    return Err(Error::Unsupported("Metal sparse array tail geometry"));
                }
                tail_bytes = single_tail;
            }
            if tile.width == 0
                || tile.height == 0
                || tile.depth == 0
                || bytes == 0
                || tail_bytes % bytes != 0
            {
                return Err(Error::Unsupported("Metal sparse tile geometry"));
            }
            let width = u32::try_from(tile.width).map_err(|_| Error::InvalidRange)?;
            let height = u32::try_from(tile.height).map_err(|_| Error::InvalidRange)?;
            let depth = u32::try_from(tile.depth).map_err(|_| Error::InvalidRange)?;
            let standard = u32::try_from(first).map_err(|_| Error::InvalidRange)?;
            let grids: Vec<_> = (0..standard)
                .map(|m| {
                    [
                        (desc.extent[0] >> m).max(1).div_ceil(width),
                        (desc.extent[1] >> m).max(1).div_ceil(height),
                        if desc.dimension == api::TextureDimension::D3 {
                            (desc.extent[2] >> m).max(1).div_ceil(depth)
                        } else {
                            1
                        },
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
                    aspect: match desc.format {
                        Depth16Unorm | Depth32Float => api::TextureAspects::DEPTH,
                        Stencil8 => api::TextureAspects::STENCIL,
                        _ => api::TextureAspects::COLOR,
                    },
                    packed_tail_shared_by_layers: false,
                    tile_extent: [width, height, depth],
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
        texture.desc().array_layers,
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
                textureSlice: r.layer as usize,
                textureRegion: MTLRegion {
                    origin: MTLOrigin {
                        x: r.origin[0] as usize,
                        y: r.origin[1] as usize,
                        z: r.origin[2] as usize,
                    },
                    size: MTLSize {
                        width: r.count[0] as usize,
                        height: r.count[1] as usize,
                        depth: r.count[2] as usize,
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
        let mut mappings = self
            .texture
            .sparse
            .as_ref()
            .unwrap()
            .mappings
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let old = mappings.replace(&self.regions, self.allocation.as_ref());
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
        // Keep frontier mutation and native queue insertion in the same order
        // even when completion callbacks enqueue bindings concurrently.
        drop(mappings);
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
