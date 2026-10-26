//! Unit tests for the parent module.

use super::super::tests::type_contract_tests::NativeShared;
use super::*;
impl SparseBackend for NativeShared {
    fn create_sparse_allocation(_: &Arc<()>, _: u64) -> Result<Arc<()>> {
        Ok(Arc::new(()))
    }
}
#[test]
fn packed_tail_region_covers_the_whole_reported_tail() {
    let tail = SparsePackedMipTail {
        first_mip: 2,
        mip_count: 7,
        tile_count: 3,
    };
    let region = SparseTile::packed_tail(tail);
    assert_eq!(region.mip, 2);
    assert_eq!(region.layer, 0);
    assert_eq!(region.origin, [0, 0, 0]);
    assert_eq!(region.count, [3, 1, 1]);
}
#[test]
fn primary_tiling_fields_match_first_aspect_group() {
    let group = SparseAspectTiling {
        aspect: TextureAspects::DEPTH | TextureAspects::STENCIL,
        tile_extent: [64, 32, 1],
        tile_grid: [3, 4, 1],
        tile_bytes: 65536,
        standard_mip_grids: vec![[3, 4, 1]],
        packed_tail: None,
        packed_tail_shared_by_layers: false,
    };
    let tiling = SparseTextureTiling::from_aspects(vec![group.clone()]).unwrap();
    assert_eq!(tiling.tile_extent, group.tile_extent);
    assert_eq!(tiling.standard_mip_grids, group.standard_mip_grids);
    assert_eq!(
        tiling.aspects[0].aspect,
        TextureAspects::DEPTH | TextureAspects::STENCIL
    );
    assert!(SparseTextureTiling::from_aspects(vec![]).is_none());
}
#[test]
fn sparse_contract_checks_empty_allocation_and_texture_device() {
    let id = DeviceId::allocate().unwrap();
    let other = DeviceId::allocate().unwrap();
    let caps = DeviceCapabilities {
        sparse: true,
        ..DeviceCapabilities::default()
    };
    // SAFETY: each mock handle is paired with its unique device identity.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    let foreign = unsafe { Texture::<NativeShared>::from_native(other, Arc::new(())) };
    let sparse = device.sparse().unwrap();
    assert!(matches!(sparse.allocation(0), Err(Error::InvalidRange)));
    assert!(matches!(sparse.tiling(&foreign), Err(Error::WrongDevice)));
}
