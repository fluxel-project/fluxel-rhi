//! CPU row staging must preserve distinct padded depth rows.
use super::*;
use crate::backend::api::{
    ContextEpoch, ContextStamp, DeviceIdentity, GlExtent3d, GlPixelFormat, GlPixelLayout,
    GlRepackPolicy, GlTextureAspect, GlTextureSubresource, TextureId,
};

#[test]
fn padded_depth_rows_retain_distinct_values() {
    let stamp = ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL);
    let region = GlTextureRegion {
        subresource: GlTextureSubresource {
            texture: TextureId::new(stamp, 0, 0),
            aspect: GlTextureAspect::DepthOnly,
            mip_level: 1,
            base_layer: 1,
            layer_count: 1,
        },
        origin: [1, 1, 0],
        extent: GlExtent3d {
            width: 2,
            height: 2,
            depth_or_layers: 1,
        },
    };
    let layout = GlPixelLayout {
        format: GlPixelFormat::Depth16,
        bytes_per_row: 16,
        rows_per_image: 2,
        offset: 4,
        alignment: 4,
        repack: GlRepackPolicy::Disallow,
    };
    let mut bytes = [0u8; 32];
    bytes[4..8].copy_from_slice(&[0, 0, 1, 0]);
    bytes[20..24].copy_from_slice(&[0, 128, 255, 255]);
    assert_eq!(
        packed_unpack_rows("fixture", layout, region, &bytes).unwrap(),
        [0, 0, 1, 0, 0, 128, 255, 255]
    );
}
