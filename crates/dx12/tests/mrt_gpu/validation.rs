//! Attachment and secondary inheritance validation.
use super::*;

#[test]
fn rejects_invalid_attachment_configurations_and_secondary_signature() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let texture = |size, samples| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [size, size, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap()
    };
    let textures = [texture(16, 1), texture(8, 1), texture(16, 4)];
    let view = |texture: &Texture<_>| {
        d.create_view(
            texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap()
    };
    let views = [view(&textures[0]), view(&textures[1]), view(&textures[2])];
    let attachment = |view| {
        Some(ColorAttachment {
            view: AttachmentView::Texture(view),
            resolve_target: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
        })
    };
    let area = Rect {
        x: 0,
        y: 0,
        width: 16,
        height: 16,
    };
    for (a, b) in [(0, 0), (0, 1), (0, 2)] {
        let colors = [attachment(&views[a]), attachment(&views[b])];
        let mut encoder = d.encoder(lane).unwrap();
        let result = encoder.begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: area,
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        });
        assert!(result.is_err(), "invalid attachments {a}/{b} were accepted");
    }
    let colors = [attachment(&views[0])];
    let mut encoder = d.encoder(lane).unwrap();
    let result = encoder.begin_raster(&RasterPassDesc {
        label: None,
        colors: &colors,
        depth_stencil: None,
        render_area: Rect { x: 1, ..area },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    });
    assert!(result.is_err(), "out-of-bounds render area was accepted");
    drop(result);
    let sig = SecondaryRasterDesc {
        color_formats: vec![None, Some(TextureFormat::Rgba8Unorm)],
        depth_stencil_format: None,
        samples: 1,
        multiview_mask: None,
    };
    let mut encoder = d.encoder(lane).unwrap();
    let result = encoder.begin_raster(&RasterPassDesc {
        label: None,
        colors: &colors,
        depth_stencil: None,
        render_area: area,
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: Some(sig),
    });
    assert!(
        result.is_err(),
        "incompatible secondary color slots were accepted"
    );
}
