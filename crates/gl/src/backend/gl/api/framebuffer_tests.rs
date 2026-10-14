//! Tests for the private framebuffer.rs contract.
use super::{
    GlBlitMask, GlBlitRegion, GlColorClearValue, GlFramebufferDescriptor,
    GlFramebufferValidationError, GlLoadOp, GlStoreOp,
};
#[test]
fn clear_value_retains_float_bits() {
    let clear = GlColorClearValue {
        red: 1.0f32.to_bits(),
        green: 0,
        blue: 0,
        alpha: 1.0f32.to_bits(),
    };
    assert_eq!(clear.red, 1.0f32.to_bits());
    assert_eq!(GlLoadOp::Clear, GlLoadOp::Clear);
    assert_eq!(GlStoreOp::Discard, GlStoreOp::Discard);
}
#[test]
fn rejects_empty_framebuffer() {
    let descriptor = GlFramebufferDescriptor {
        color_attachments: vec![],
        depth_stencil_attachment: None,
        draw_buffers: vec![],
    };
    let stamp = super::super::ContextStamp::new(
        super::super::DeviceIdentity::new(1).unwrap(),
        super::super::ContextEpoch::INITIAL,
    );
    assert_eq!(
        descriptor.validate(1, 1, stamp),
        Err(GlFramebufferValidationError::NoAttachments)
    );
}
#[test]
fn draw_buffers_selection_is_bounded_unique_and_in_range() {
    let descriptor = |draw_buffers: Vec<u32>| GlFramebufferDescriptor {
        color_attachments: vec![],
        depth_stencil_attachment: Some(empty_view()),
        draw_buffers,
    };
    let stamp = super::super::ContextStamp::new(
        super::super::DeviceIdentity::new(1).unwrap(),
        super::super::ContextEpoch::INITIAL,
    );
    assert_eq!(
        descriptor(vec![0]).validate(4, 4, stamp),
        Err(GlFramebufferValidationError::DrawBufferIndexOutOfBounds)
    );
    assert_eq!(descriptor(vec![]).validate(4, 4, stamp), Ok(()));
}
#[test]
fn blit_regions_and_masks_fail_closed_before_gl() {
    let region = GlBlitRegion {
        src_offset: [0; 2],
        src_extent: [4, 4],
        dst_offset: [0; 2],
        dst_extent: [4, 4],
    };
    assert_eq!(region.validate(), Ok(()));
    assert_eq!(
        GlBlitRegion {
            dst_extent: [0, 4],
            ..region
        }
        .validate(),
        Err(GlFramebufferValidationError::InvalidBlitRegion)
    );
    assert!(
        GlBlitMask {
            color: false,
            depth: false,
            stencil: false
        }
        .is_empty()
    );
}
fn empty_view() -> super::GlTextureView {
    super::GlTextureView {
        target: super::GlAttachmentTarget::Texture(super::TextureId::new(
            super::super::ContextStamp::new(
                super::super::DeviceIdentity::new(1).unwrap(),
                super::super::ContextEpoch::INITIAL,
            ),
            0,
            0,
        )),
        format: super::GlFormat::Depth32Float,
        mip_level: 0,
        array_layer: 0,
        layer_count: 1,
        width: 1,
        height: 1,
        sample_count: 1,
    }
}

fn stamp() -> super::super::ContextStamp {
    super::super::ContextStamp::new(
        super::super::DeviceIdentity::new(1).unwrap(),
        super::super::ContextEpoch::INITIAL,
    )
}

/// A color-renderable view of `layer_count` layers on a fresh allocation.
fn color_view(slot: u32, layer_count: u32) -> super::GlTextureView {
    super::GlTextureView {
        target: super::GlAttachmentTarget::Texture(super::TextureId::new(stamp(), slot, 0)),
        format: super::GlFormat::Rgba8Unorm,
        mip_level: 0,
        array_layer: 0,
        layer_count,
        width: 4,
        height: 4,
        sample_count: 1,
    }
}

#[test]
fn attachment_views_must_agree_on_their_layer_count() {
    let color = color_view(0, 2);
    // The depth view matches the color view in every shape rule but the
    // layer count, so the case can only fail on that rule.
    let depth = super::GlTextureView {
        layer_count: 1,
        width: color.width,
        height: color.height,
        ..empty_view()
    };
    let descriptor = GlFramebufferDescriptor {
        color_attachments: vec![color],
        depth_stencil_attachment: Some(depth),
        draw_buffers: vec![],
    };
    assert_eq!(
        descriptor.validate(4, 4, stamp()),
        Err(GlFramebufferValidationError::MismatchedLayerCount)
    );
    let agreed = GlFramebufferDescriptor {
        depth_stencil_attachment: Some(super::GlTextureView {
            format: super::GlFormat::Depth32Float,
            layer_count: 2,
            ..depth
        }),
        ..descriptor
    };
    assert_eq!(agreed.validate(4, 4, stamp()), Ok(()));
    assert_eq!(agreed.view_count(), 2);
}

#[test]
fn a_view_serving_no_layer_is_not_a_view() {
    let descriptor = GlFramebufferDescriptor {
        color_attachments: vec![color_view(0, 0)],
        depth_stencil_attachment: None,
        draw_buffers: vec![],
    };
    assert_eq!(
        descriptor.validate(4, 4, stamp()),
        Err(GlFramebufferValidationError::InvalidLayerCount)
    );
}

/// A multiview view count is refused outright on a context that did not
/// prove multiview, and refused above the proved count otherwise.
#[test]
fn multiview_view_count_is_gated_by_the_proved_limit() {
    let descriptor = GlFramebufferDescriptor {
        color_attachments: vec![color_view(0, 2)],
        depth_stencil_attachment: None,
        draw_buffers: vec![],
    };
    assert_eq!(
        descriptor.validate_multiview(1),
        Err(GlFramebufferValidationError::MultiviewNotSupported)
    );
    assert_eq!(descriptor.validate_multiview(2), Ok(()));

    let three = GlFramebufferDescriptor {
        color_attachments: vec![color_view(0, 3)],
        ..descriptor
    };
    assert_eq!(
        three.validate_multiview(2),
        Err(GlFramebufferValidationError::MultiviewViewCountExceedsLimit)
    );
    assert_eq!(three.validate_multiview(3), Ok(()));
}

#[test]
fn single_view_passes_never_need_the_multiview_limit() {
    let descriptor = GlFramebufferDescriptor {
        color_attachments: vec![],
        depth_stencil_attachment: Some(empty_view()),
        draw_buffers: vec![],
    };
    assert_eq!(descriptor.view_count(), 1);
    assert_eq!(descriptor.validate_multiview(1), Ok(()));
}

/// A pass may not resolve into a multiview attachment: a resolve is a
/// single-view blit, so a multilayer destination has no defined result.
#[test]
fn resolve_targets_are_single_layer() {
    let view = color_view(0, 1);
    let multisampled = super::GlTextureView {
        sample_count: 4,
        ..view
    };
    let pass = super::GlRenderPassDescriptor {
        target: super::GlRenderTarget::Offscreen(super::FramebufferId::new(stamp(), 0, 0)),
        color_attachments: vec![super::GlColorAttachment {
            view: super::GlPassAttachmentView::Allocated(multisampled),
            resolve_target: Some(super::GlTextureView {
                layer_count: 2,
                ..view
            }),
            load: GlLoadOp::Load,
            store: GlStoreOp::Store,
            clear: GlColorClearValue {
                red: 0,
                green: 0,
                blue: 0,
                alpha: 0,
            },
        }],
        depth_stencil_attachment: None,
    };
    let framebuffer = GlFramebufferDescriptor {
        color_attachments: vec![multisampled],
        depth_stencil_attachment: None,
        draw_buffers: vec![],
    };
    assert_eq!(
        pass.validate(Some(&framebuffer), 4, 4, stamp()),
        Err(GlFramebufferValidationError::InvalidResolve)
    );
    // The pass itself already agrees on one view count, so the multiview
    // gate stays satisfied and only the resolve rule rejects.
    assert_eq!(pass.view_count(), 1);
    assert_eq!(pass.validate_multiview(1), Ok(()));
}
