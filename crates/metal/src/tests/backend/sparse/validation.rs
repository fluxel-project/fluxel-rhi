//! Unit tests for the corresponding implementation module.

use super::*;
use fluxel_rhi_core::api::{SparseAspectTiling, SparsePackedMipTail, TextureAspects};
fn tiling() -> SparseTextureTiling {
    SparseTextureTiling::from_aspects(vec![SparseAspectTiling {
        aspect: TextureAspects::COLOR,
        packed_tail_shared_by_layers: false,
        tile_extent: [128, 128, 1],
        tile_grid: [4, 2, 1],
        tile_bytes: 65536,
        standard_mip_grids: vec![[4, 2, 1]],
        packed_tail: Some(SparsePackedMipTail {
            first_mip: 1,
            mip_count: 3,
            tile_count: 2,
        }),
    }])
    .unwrap()
}
fn tile(mip: u32, origin: [u32; 3], count: [u32; 3]) -> SparseTile {
    SparseTile {
        aspect: None,
        mip,
        layer: 0,
        origin,
        count,
    }
}
#[test]
fn offsets_follow_standard_rectangles_and_packed_tail() {
    let tiles = [tile(0, [1, 0, 0], [2, 2, 1]), tile(1, [0, 0, 0], [2, 1, 1])];
    let regions = validate_regions(&tiling(), 1, &tiles, Some(8 * 65536), 65536).unwrap();
    assert_eq!(
        (regions[0].heap_tile_offset, regions[1].heap_tile_offset),
        (1, 5)
    );
    assert!(validate_regions(&tiling(), 1, &tiles, Some(6 * 65536), 65536).is_err());
    assert!(validate_regions(&tiling(), 1, &tiles, Some(8 * 65536), 1).is_err());
}
#[test]
fn rejects_overlap_and_partial_tail_and_checks_unmap_offset() {
    let overlap = [tile(0, [0, 0, 0], [2, 1, 1]), tile(0, [1, 0, 0], [1, 1, 1])];
    assert!(validate_regions(&tiling(), 1, &overlap, None, 0).is_err());
    assert!(validate_regions(&tiling(), 1, &[tile(1, [0, 0, 0], [1, 1, 1])], None, 0).is_err());
    assert!(
        validate_regions(&tiling(), 1, &[tile(0, [0, 0, 0], [1, 1, 1])], None, 65536).is_err()
    );
}

#[test]
fn layers_and_depth_tiles_have_independent_ownership() {
    use std::sync::Arc;
    let mut geometry = tiling();
    geometry.standard_mip_grids[0] = [2, 1, 2];
    let mut state = MappingState::new();
    let a = Arc::new(());
    let weak = Arc::downgrade(&a);
    let first = tile(0, [0, 0, 0], [1, 1, 1]);
    let mut second = tile(0, [0, 0, 1], [1, 1, 1]);
    second.layer = 1;
    let bind = validate_regions(&geometry, 2, &[first, second], Some(2 * 65536), 0).unwrap();
    assert_eq!(bind[1].heap_tile_offset, 1);
    state.replace(&bind, Some(&a));
    drop(a);
    let unmap =
        validate_regions(&geometry, 2, &[tile(0, [0, 0, 0], [1, 1, 1])], None, 0).unwrap();
    drop(state.replace(&unmap, None));
    assert!(weak.upgrade().is_some());
    let mut second = tile(0, [0, 0, 1], [1, 1, 1]);
    second.layer = 1;
    let unmap = validate_regions(&geometry, 2, &[second], None, 0).unwrap();
    let pending = state.replace(&unmap, None);
    assert!(weak.upgrade().is_some());
    drop(pending);
    assert!(weak.upgrade().is_none());
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
        1,
        &[tile(0, [0, 0, 0], [2, 1, 1])],
        Some(2 * 65536),
        0,
    )
    .unwrap();
    assert!(state.replace(&first, Some(&a)).is_empty());
    drop(a);
    let first_tile = validate_regions(
        &tiling(),
        1,
        &[tile(0, [0, 0, 0], [1, 1, 1])],
        Some(65536),
        0,
    )
    .unwrap();
    let retained = state.replace(&first_tile, Some(&b));
    drop(retained);
    assert!(weak_a.upgrade().is_some()); // second tile remains mapped
    let second_tile =
        validate_regions(&tiling(), 1, &[tile(0, [1, 0, 0], [1, 1, 1])], None, 0).unwrap();
    let pending = state.replace(&second_tile, None);
    assert!(weak_a.upgrade().is_some()); // old heap held by in-flight operation
    drop(pending);
    assert!(weak_a.upgrade().is_none());
}
