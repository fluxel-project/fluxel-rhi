//! Sparse native aspect selection checks.
use super::*;
#[test]
fn select_independent_or_combined_sparse_aspects() {
    let group = |aspect, native_aspect, base| AspectState {
        tiling: api::SparseAspectTiling {
            aspect,
            tile_extent: [64, 64, 1],
            tile_grid: [1, 1, 1],
            tile_bytes: 65536,
            standard_mip_grids: vec![[1, 1, 1]],
            packed_tail: None,
            packed_tail_shared_by_layers: false,
        },
        native_aspect,
        starts: vec![0, 1],
        base,
        per_layer_tiles: 1,
        tail_offset: 0,
        tail_size: 0,
        tail_stride: 0,
    };
    let separate = vec![
        group(api::TextureAspects::DEPTH, vk::ImageAspectFlags::DEPTH, 0),
        group(
            api::TextureAspects::STENCIL,
            vk::ImageAspectFlags::STENCIL,
            4,
        ),
    ];
    assert_eq!(select_aspect(&separate, None).unwrap().base, 0);
    assert_eq!(
        select_aspect(&separate, Some(api::TextureAspects::STENCIL))
            .unwrap()
            .base,
        4
    );
    assert!(matches!(
        select_aspect(
            &separate,
            Some(api::TextureAspects::DEPTH | api::TextureAspects::STENCIL)
        ),
        Err(Error::InvalidRange)
    ));
    let combined = vec![group(
        api::TextureAspects::DEPTH | api::TextureAspects::STENCIL,
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        0,
    )];
    assert!(select_aspect(&combined, Some(api::TextureAspects::DEPTH)).is_err());
    assert!(
        select_aspect(
            &combined,
            Some(api::TextureAspects::DEPTH | api::TextureAspects::STENCIL)
        )
        .is_ok()
    );
}
#[test]
fn shared_tail_requires_layer_zero() {
    assert!(validate_tail_layer(true, 0).is_ok());
    assert!(matches!(
        validate_tail_layer(true, 1),
        Err(Error::InvalidRange)
    ));
    assert!(validate_tail_layer(false, 1).is_ok());
}
