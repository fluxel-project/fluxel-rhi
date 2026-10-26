//! Validation of the Core sparse rectangle and allocation-offset contract.
//!
//! The returned offsets are in tiles, as required by Metal 4. The caller must
//! additionally synchronize mapping operations and retain all mapped heaps.

use fluxel_rhi_core::api::{Error, Result, SparseTextureTiling, SparseTile};
use std::collections::{HashMap, HashSet};

/// Host-side mapping frontier. Mutations must be made in native execution order.
pub(super) struct MappingState<T> {
    tiles: HashMap<(u32, u32, u32, u32, u32), T>,
}
impl<T: Clone> MappingState<T> {
    pub(super) fn new() -> Self {
        Self {
            tiles: HashMap::new(),
        }
    }

    /// The removed references must remain live until this mapping completes.
    pub(super) fn replace(&mut self, regions: &[Region], allocation: Option<&T>) -> Vec<T> {
        let mut old = Vec::new();
        for region in regions {
            for z in region.origin[2]..region.origin[2] + region.count[2] {
                for y in region.origin[1]..region.origin[1] + region.count[1] {
                    for x in region.origin[0]..region.origin[0] + region.count[0] {
                        let key = (region.mip, region.layer, x, y, z);
                        let previous = match allocation {
                            Some(a) => self.tiles.insert(key, a.clone()),
                            None => self.tiles.remove(&key),
                        };
                        if let Some(previous) = previous {
                            old.push(previous);
                        }
                    }
                }
            }
        }
        old
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Region {
    pub mip: u32,
    pub layer: u32,
    pub origin: [u32; 3],
    pub count: [u32; 3],
    pub heap_tile_offset: u64,
}

pub(super) fn validate_regions(
    tiling: &SparseTextureTiling,
    layers: u32,
    tiles: &[SparseTile],
    allocation_size: Option<u64>,
    allocation_offset: u64,
) -> Result<Vec<Region>> {
    let page = tiling.tile_bytes;
    if page == 0
        || tiles.is_empty()
        || allocation_offset % page != 0
        || (allocation_size.is_none() && allocation_offset != 0)
    {
        return Err(Error::InvalidRange);
    }
    let mut cursor = allocation_offset / page;
    let mut used = HashSet::new();
    let mut out = Vec::with_capacity(tiles.len());
    for tile in tiles {
        if tile.aspect.is_some_and(|a| a != tiling.aspects[0].aspect) {
            return Err(Error::InvalidRange);
        }
        if tile.layer >= layers {
            return Err(Error::InvalidRange);
        }
        if let Some(grid) = tiling.standard_mip_grids.get(tile.mip as usize) {
            if (0..3).any(|i| {
                tile.count[i] == 0
                    || tile.origin[i]
                        .checked_add(tile.count[i])
                        .is_none_or(|end| end > grid[i])
            }) {
                return Err(Error::InvalidRange);
            }
            for z in tile.origin[2]..tile.origin[2] + tile.count[2] {
                for y in tile.origin[1]..tile.origin[1] + tile.count[1] {
                    for x in tile.origin[0]..tile.origin[0] + tile.count[0] {
                        if !used.insert((tile.mip, tile.layer, x, y, z)) {
                            return Err(Error::InvalidRange);
                        }
                    }
                }
            }
        } else if let Some(tail) = tiling.packed_tail.filter(|t| t.first_mip == tile.mip) {
            if tile.origin != [0, 0, 0]
                || tile.count != [tail.tile_count, 1, 1]
                || tail.tile_count == 0
                || !used.insert((tile.mip, tile.layer, 0, 0, 0))
            {
                return Err(Error::InvalidRange);
            }
        } else {
            return Err(Error::InvalidRange);
        }
        let count = u64::from(tile.count[0])
            .checked_mul(u64::from(tile.count[1]))
            .and_then(|n| n.checked_mul(u64::from(tile.count[2])))
            .ok_or(Error::InvalidRange)?;
        out.push(Region {
            mip: tile.mip,
            layer: tile.layer,
            origin: tile.origin,
            count: tile.count,
            heap_tile_offset: if allocation_size.is_some() { cursor } else { 0 },
        });
        cursor = cursor.checked_add(count).ok_or(Error::InvalidRange)?;
    }
    if allocation_size.is_some_and(|size| cursor.checked_mul(page).is_none_or(|end| end > size)) {
        return Err(Error::InvalidRange);
    }
    Ok(out)
}

#[cfg(test)]
#[path = "../../tests/backend/sparse/validation.rs"]
mod tests;
