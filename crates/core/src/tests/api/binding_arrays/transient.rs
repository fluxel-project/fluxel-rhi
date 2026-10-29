//! Transient attachment texture creation and sampling constraints.
use super::{GROUP_CALLS, LAYOUT_CALLS, TEXTURE_CALLS, device, entry};
use crate::api::tests::type_contract_tests::NativeShared;
use crate::{api::*, backend::*};
use std::sync::Arc;

#[test]
fn transient_attachment_rejects_incompatible_creation_and_sampling_before_backend() {
    TEXTURE_CALLS.with(|calls| calls.set(0));
    GROUP_CALLS.with(|calls| calls.set(0));
    let device = device(true);
    let base = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [16, 16, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::TRANSIENT_ATTACHMENT | TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    };
    let mut invalid = base.clone();
    invalid.usage |= TextureUsage::TEXTURE_BINDING;
    assert!(matches!(
        device.create_texture(&invalid),
        Err(Error::InvalidState)
    ));
    assert!(matches!(
        device.texture_support(&invalid),
        Support::Unsupported(_)
    ));
    TEXTURE_CALLS.with(|calls| assert_eq!(calls.get(), 0));

    let texture = device.create_texture(&base).unwrap();
    let view = device
        .create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: TextureSampleType::Float,
                    dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    assert!(matches!(
        device.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            }],
        }),
        Err(Error::InvalidState)
    ));
    GROUP_CALLS.with(|calls| assert_eq!(calls.get(), 0));

    let lane = LaneId {
        device: device.id(),
        ordinal: 0,
    };
    let mut encoder = device.encoder(lane).unwrap();
    let invalid_load = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Load,
        store: StoreOp::Discard,
    })];
    let raster = |colors| RasterPassDesc {
        label: None,
        colors,
        depth_stencil: None,
        render_area: Rect {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    };
    assert!(matches!(
        encoder.begin_raster(&raster(&invalid_load)),
        Err(Error::InvalidState)
    ));

    let persistent_desc = TextureDesc {
        usage: TextureUsage::RENDER_ATTACHMENT,
        ..base.clone()
    };
    let persistent = device.create_texture(&persistent_desc).unwrap();
    let persistent_view = device
        .create_view(
            &persistent,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let transient_resolve = [Some(ColorAttachment {
        view: AttachmentView::Texture(&persistent_view),
        resolve_target: Some(AttachmentView::Texture(&view)),
        load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
        store: StoreOp::Store,
    })];
    assert!(matches!(
        encoder.begin_raster(&raster(&transient_resolve)),
        Err(Error::InvalidState)
    ));
}
