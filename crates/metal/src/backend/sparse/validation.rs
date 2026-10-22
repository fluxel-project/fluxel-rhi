//! Validation of the Core sparse rectangle and allocation-offset contract.
//!
//! The returned offsets are in tiles, as required by Metal 4. The caller must
//! additionally synchronize mapping operations and retain all mapped heaps.

use fluxel_rhi_core::api::{Error, Result, SparseTextureTiling, SparseTile};
use std::collections::{HashMap, HashSet};

/// Host-side mapping frontier. Mutations must be made in native execution order.
pub(super) struct MappingState<T> {
    tiles: HashMap<(u32, u32, u32), T>,
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
            for y in region.origin[1]..region.origin[1] + region.count[1] {
                for x in region.origin[0]..region.origin[0] + region.count[0] {
                    let key = (region.mip, x, y);
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
        old
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Region {
    pub mip: u32,
    pub origin: [u32; 3],
    pub count: [u32; 3],
    pub heap_tile_offset: u64,
}

pub(super) fn validate_regions(
    tiling: &SparseTextureTiling,
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
        if tile.layer != 0 || tile.origin[2] != 0 || tile.count[2] != 1 {
            return Err(Error::InvalidRange);
        }
        if let Some(grid) = tiling.standard_mip_grids.get(tile.mip as usize) {
            if tile.count[0] == 0
                || tile.count[1] == 0
                || tile.origin[0]
                    .checked_add(tile.count[0])
                    .is_none_or(|x| x > grid[0])
                || tile.origin[1]
                    .checked_add(tile.count[1])
                    .is_none_or(|y| y > grid[1])
            {
                return Err(Error::InvalidRange);
            }
            for y in tile.origin[1]..tile.origin[1] + tile.count[1] {
                for x in tile.origin[0]..tile.origin[0] + tile.count[0] {
                    if !used.insert((tile.mip, x, y)) {
                        return Err(Error::InvalidRange);
                    }
                }
            }
        } else if let Some(tail) = tiling.packed_tail.filter(|t| t.first_mip == tile.mip) {
            if tile.origin != [0, 0, 0]
                || tile.count != [tail.tile_count, 1, 1]
                || tail.tile_count == 0
                || !used.insert((tile.mip, 0, 0))
            {
                return Err(Error::InvalidRange);
            }
        } else {
            return Err(Error::InvalidRange);
        }
        let count = u64::from(tile.count[0])
            .checked_mul(u64::from(tile.count[1]))
            .ok_or(Error::InvalidRange)?;
        out.push(Region {
            mip: tile.mip,
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
mod tests {
    use super::*;
    use fluxel_rhi_core::api::SparsePackedMipTail;
    fn tiling() -> SparseTextureTiling {
        SparseTextureTiling {
            tile_extent: [128, 128, 1],
            tile_grid: [4, 2, 1],
            tile_bytes: 65536,
            standard_mip_grids: vec![[4, 2, 1]],
            packed_tail: Some(SparsePackedMipTail {
                first_mip: 1,
                mip_count: 3,
                tile_count: 2,
            }),
        }
    }
    fn tile(mip: u32, origin: [u32; 3], count: [u32; 3]) -> SparseTile {
        SparseTile {
            mip,
            layer: 0,
            origin,
            count,
        }
    }
    #[test]
    fn offsets_follow_standard_rectangles_and_packed_tail() {
        let tiles = [tile(0, [1, 0, 0], [2, 2, 1]), tile(1, [0, 0, 0], [2, 1, 1])];
        let regions = validate_regions(&tiling(), &tiles, Some(8 * 65536), 65536).unwrap();
        assert_eq!(
            (regions[0].heap_tile_offset, regions[1].heap_tile_offset),
            (1, 5)
        );
        assert!(validate_regions(&tiling(), &tiles, Some(6 * 65536), 65536).is_err());
        assert!(validate_regions(&tiling(), &tiles, Some(8 * 65536), 1).is_err());
    }
    #[test]
    fn rejects_overlap_and_partial_tail_and_checks_unmap_offset() {
        let overlap = [tile(0, [0, 0, 0], [2, 1, 1]), tile(0, [1, 0, 0], [1, 1, 1])];
        assert!(validate_regions(&tiling(), &overlap, None, 0).is_err());
        assert!(validate_regions(&tiling(), &[tile(1, [0, 0, 0], [1, 1, 1])], None, 0).is_err());
        assert!(
            validate_regions(&tiling(), &[tile(0, [0, 0, 0], [1, 1, 1])], None, 65536).is_err()
        );
    }

    #[test]
    fn mapped_heaps_release_only_after_unmap_completion_and_last_tile() {
        use std::sync::Arc;
        let a = Arc::new(());
        let b = Arc::new(());
        let weak_a = Arc::downgrade(&a);
        let mut state = MappingState::new();
        let first = validate_regions(
            &tiling(),
            &[tile(0, [0, 0, 0], [2, 1, 1])],
            Some(2 * 65536),
            0,
        )
        .unwrap();
        assert!(state.replace(&first, Some(&a)).is_empty());
        drop(a);
        let first_tile =
            validate_regions(&tiling(), &[tile(0, [0, 0, 0], [1, 1, 1])], Some(65536), 0).unwrap();
        let retained = state.replace(&first_tile, Some(&b));
        drop(retained);
        assert!(weak_a.upgrade().is_some()); // second tile remains mapped
        let second_tile =
            validate_regions(&tiling(), &[tile(0, [1, 0, 0], [1, 1, 1])], None, 0).unwrap();
        let pending = state.replace(&second_tile, None);
        assert!(weak_a.upgrade().is_some()); // old heap held by in-flight operation
        drop(pending);
        assert!(weak_a.upgrade().is_none());
    }
}
