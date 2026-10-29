//! Secondary raster-pass inheritance and attachment signature validation.

use super::*;

fn signature(depth: Option<api::TextureFormat>) -> api::SecondaryRasterDesc {
    api::SecondaryRasterDesc {
        color_formats: vec![Some(api::TextureFormat::Rgba8Unorm)],
        depth_stencil_format: depth,
        samples: 1,
        multiview_mask: None,
    }
}

#[test]
fn secondary_signature_preserves_multiple_slots_and_holes() {
    let mut desc = signature(None);
    desc.color_formats = vec![
        Some(api::TextureFormat::Rgba8Unorm),
        None,
        Some(api::TextureFormat::Bgra8Unorm),
    ];
    assert!(validate_signature(&desc).is_ok());
    desc.color_formats = vec![None; 9];
    assert!(validate_signature(&desc).is_err());
}

#[test]
fn inheritance_compares_every_color_slot() {
    let mut desc = signature(None);
    desc.color_formats = vec![
        Some(api::TextureFormat::Rgba8Unorm),
        None,
        Some(api::TextureFormat::Bgra8Unorm),
    ];
    assert!(validate_inheritance(Some(&desc), &desc.color_formats, None, 1, None).is_ok());
    assert!(matches!(
        validate_inheritance(
            Some(&desc),
            &[
                Some(api::TextureFormat::Rgba8Unorm),
                None,
                Some(api::TextureFormat::Rgba8Unorm)
            ],
            None,
            1,
            None
        ),
        Err(Error::InvalidState)
    ));
}

#[test]
fn secondary_signature_accepts_depth32_inheritance() {
    assert!(validate_signature(&signature(Some(api::TextureFormat::Depth32Float))).is_ok());
}

#[test]
fn secondary_inheritance_must_match_the_real_depth_attachment() {
    let inherited = signature(Some(api::TextureFormat::Depth32Float));
    assert!(
        validate_inheritance(
            Some(&inherited),
            &[Some(api::TextureFormat::Rgba8Unorm)],
            Some(api::TextureFormat::Depth32Float),
            1,
            None,
        )
        .is_ok()
    );
    assert!(matches!(
        validate_inheritance(
            Some(&inherited),
            &[Some(api::TextureFormat::Rgba8Unorm)],
            None,
            1,
            None,
        ),
        Err(Error::InvalidState)
    ));
}
