use super::*;

#[test]
fn compressed_formats_preserve_block_and_color_space() {
    use crate::backend::api::{GlAstcBlock as B, GlCompressedColorSpace as C};
    assert_eq!(
        lower_texture_format(TextureFormat::Bc6hRgbFloat),
        Ok(GlFormat::Bc6hRgbSfloat)
    );
    assert_eq!(
        lower_texture_format(TextureFormat::Etc2Rgb8A1UnormSrgb),
        Ok(GlFormat::Etc2Rgb8A1Srgb)
    );
    assert_eq!(
        lower_texture_format(TextureFormat::Astc {
            block: [10, 6],
            srgb: false,
            hdr: true
        }),
        Ok(GlFormat::Astc {
            block: B::B10x6,
            color_space: C::Hdr
        })
    );
    assert!(
        lower_texture_format(TextureFormat::Astc {
            block: [5, 6],
            srgb: false,
            hdr: false
        })
        .is_err()
    );
    assert!(
        lower_texture_format(TextureFormat::Astc {
            block: [4, 4],
            srgb: true,
            hdr: true
        })
        .is_err()
    );
}

#[test]
fn array_upload_z_becomes_the_subresource_base_once() {
    assert_eq!(
        texture_region_coordinates(GlTextureDimension::D2Array, [7, 11, 3], [32, 16, 1],),
        (3, 1, [7, 11, 0]),
    );
}

#[test]
fn volume_upload_keeps_z_as_a_region_coordinate() {
    assert_eq!(
        texture_region_coordinates(GlTextureDimension::D3, [7, 11, 3], [32, 16, 2],),
        (0, 1, [7, 11, 3]),
    );
}
