//! Sparse-texture packed-tail metadata construction and rejection coverage.

use super::*;

fn standard(start: u32) -> D3D12_SUBRESOURCE_TILING {
    D3D12_SUBRESOURCE_TILING {
        WidthInTiles: 2,
        HeightInTiles: 1,
        DepthInTiles: 1,
        StartTileIndexInOverallResource: start,
    }
}

#[test]
fn array_packed_tail_metadata_repeats_each_slice() {
    let mut packed = D3D12_SUBRESOURCE_TILING::default();
    packed.StartTileIndexInOverallResource = D3D12_PACKED_TILE;
    let metadata = [standard(0), packed, standard(8), packed];
    validate_array_tiling_metadata(&metadata, 2, 2, 1, 8).unwrap();
}

#[test]
fn array_packed_tail_metadata_rejects_non_sentinel_tail() {
    let metadata = [standard(0), standard(2), standard(8), standard(10)];
    assert!(validate_array_tiling_metadata(&metadata, 2, 2, 1, 8).is_err());
}
