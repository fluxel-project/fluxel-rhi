//! Clear validation rejects incompatible descriptions before native execution.
use super::*;
#[test]
fn rejects_invalid_clear_descriptions() {
    let d = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    };
    let color = d.create_texture(&desc).unwrap();
    let no_attachment = d
        .create_texture(&TextureDesc {
            usage: TextureUsage::COPY_SRC,
            ..desc.clone()
        })
        .unwrap();
    let depth = d
        .create_texture(&TextureDesc {
            format: TextureFormat::Depth32Float,
            ..desc
        })
        .unwrap();
    let float = TextureClearValue::Color(ClearColor::Float([0.0; 4]));
    let cases = [
        (&color, 0..2, 0..1, float),
        (&color, 0..1, 0..2, float),
        (&color, 0..1, 0..1, TextureClearValue::Depth(0.5)),
        (&depth, 0..1, 0..1, float),
        (
            &color,
            0..1,
            0..1,
            TextureClearValue::Color(ClearColor::Uint([0; 4])),
        ),
        (&no_attachment, 0..1, 0..1, float),
    ];
    for (texture, mips, layers, value) in cases {
        let mut e = d.encoder(lane).unwrap();
        assert!(matches!(
            e.clear_texture(texture, mips, layers, value),
            Err(Error::InvalidRange | Error::Unsupported(_))
        ));
    }
    let mut encoder = d.encoder(lane).unwrap();
    assert!(matches!(
        encoder.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &depth,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::DEPTH_WRITE
                },
                to: TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::DEPTH_WRITE
                }
            },
            layout: Some((ImageLayout::Undefined, ImageLayout::DepthStencilAttachment)),
            ownership: None,
        })]),
        Err(Error::Unsupported(_))
    ));
}
