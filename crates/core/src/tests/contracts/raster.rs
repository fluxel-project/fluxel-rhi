//! Raster and texture-clear admission contracts.
use super::super::support::*;
use crate::{api::*, backend::*};
use std::{
    marker::PhantomData,
    ops::Range,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[test]
fn dynamic_depth_bias_requires_dynamic_pipeline_and_rebind_restores_initial_value() {
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.dynamic_depth_bias = true;
    // SAFETY: this test pairs the mock native value and capabilities with the ID.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    let initial = DynamicDepthBias {
        constant: 1,
        slope_scale: 0.5,
        clamp: 0.0,
    };
    let changed = DynamicDepthBias {
        constant: -2,
        slope_scale: 1.25,
        clamp: 0.0,
    };
    // SAFETY: this mock pipeline is only used with the same test device.
    let dynamic = unsafe {
        RasterPipeline::<NativeShared>::from_native_with_dynamic_depth_bias(
            id,
            Arc::new(()),
            Some(initial),
        )
    };
    // SAFETY: this mock static pipeline is only used with the same test device.
    let static_pipeline = unsafe { RasterPipeline::<NativeShared>::from_native(id, Arc::new(())) };
    let lane = LaneId {
        device: id,
        ordinal: 0,
    };
    let mut encoder = CommandEncoder::new(&device, lane).unwrap();
    let desc = RasterPassDesc {
        label: None,
        colors: &[],
        depth_stencil: None,
        render_area: Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    };
    let mut pass = encoder.begin_raster(&desc).unwrap();
    DEPTH_BIAS_VALUES.with(|values| values.borrow_mut().clear());
    assert_eq!(pass.set_depth_bias(changed), Err(Error::InvalidState));
    pass.set_pipeline(&dynamic).unwrap();
    pass.set_depth_bias(changed).unwrap();
    pass.set_pipeline(&dynamic).unwrap();
    pass.set_pipeline(&static_pipeline).unwrap();
    assert_eq!(pass.set_depth_bias(changed), Err(Error::InvalidState));
    DEPTH_BIAS_VALUES.with(|values| {
        assert_eq!(*values.borrow(), vec![initial, changed, initial]);
    });
    pass.finish().unwrap();
}

#[test]
fn raster_rect_clear_rejects_empty_overflow_and_outside_ranges_before_spi() {
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.rect_attachment_clear = true;
    // SAFETY: this fixture pairs the native value and capabilities with its ID.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    let lane = LaneId {
        device: id,
        ordinal: 0,
    };
    let mut encoder = CommandEncoder::new(&device, lane).unwrap();
    // SAFETY: this wrapper is only a test identity used by the mock pass.
    let view = unsafe { TextureView::<NativeShared>::from_native(id, Arc::new(())) };
    let color_attachment = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Load,
        store: StoreOp::Store,
    })];
    let desc = RasterPassDesc {
        label: None,
        colors: &color_attachment,
        depth_stencil: None,
        render_area: Rect {
            x: 10,
            y: 20,
            width: 30,
            height: 40,
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    };
    let mut pass = encoder.begin_raster(&desc).unwrap();
    assert_eq!(
        pass.clear_rect(
            Rect {
                x: 10,
                y: 20,
                width: 0,
                height: 1,
            },
            &[],
            None,
            None,
        ),
        Err(Error::InvalidRange)
    );
    let invalid_target = [ColorAttachmentClear {
        attachment: 1,
        value: ClearColor::Float([1.0; 4]),
    }];
    assert_eq!(
        pass.clear_rect(
            Rect {
                x: 10,
                y: 20,
                width: 1,
                height: 1,
            },
            &invalid_target,
            None,
            None,
        ),
        Err(Error::InvalidRange)
    );
    assert_eq!(
        pass.clear_rect(
            Rect {
                x: u32::MAX,
                y: 20,
                width: 2,
                height: 1,
            },
            &[],
            Some(0.5),
            None,
        ),
        Err(Error::InvalidRange)
    );
    assert_eq!(
        pass.clear_rect(
            Rect {
                x: 39,
                y: 20,
                width: 2,
                height: 1,
            },
            &[],
            Some(0.5),
            None,
        ),
        Err(Error::InvalidRange)
    );
    assert_eq!(
        pass.clear_rect(
            Rect {
                x: 10,
                y: 20,
                width: 1,
                height: 1,
            },
            &[],
            None,
            None,
        ),
        Err(Error::InvalidRange)
    );
    CLEAR_RECT_CALLS.with(|calls| calls.set(0));
    assert_eq!(
        pass.clear_rect(
            Rect {
                x: 12,
                y: 23,
                width: 1,
                height: 1,
            },
            &[],
            None,
            Some(0x1ff),
        ),
        Err(Error::InvalidRange)
    );
    CLEAR_RECT_CALLS.with(|calls| assert_eq!(calls.get(), 0));
    let valid_target = [ColorAttachmentClear {
        attachment: 0,
        value: ClearColor::Float([0.25, 0.5, 0.75, 1.0]),
    }];
    CLEAR_RECT_CALLS.with(|calls| calls.set(0));
    pass.clear_rect(
        Rect {
            x: 12,
            y: 23,
            width: 2,
            height: 3,
        },
        &valid_target,
        None,
        None,
    )
    .unwrap();
    CLEAR_RECT_CALLS.with(|calls| assert_eq!(calls.get(), 1));
    pass.finish().unwrap();
}

#[test]
fn raster_rect_clear_rejects_read_only_depth_and_stencil_before_spi() {
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.rect_attachment_clear = true;
    // SAFETY: this fixture pairs the native value and capabilities with its ID.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    let lane = LaneId {
        device: id,
        ordinal: 0,
    };
    let mut encoder = CommandEncoder::new(&device, lane).unwrap();
    // SAFETY: the mock pass uses this view only as an attachment identity.
    let view = unsafe { TextureView::<NativeShared>::from_native(id, Arc::new(())) };
    let depth_stencil = DepthStencilAttachment {
        view: AttachmentView::Texture(&view),
        depth: Some((LoadOp::Load, StoreOp::Store)),
        stencil: Some((LoadOp::Load, StoreOp::Store)),
        depth_read_only: true,
        stencil_read_only: true,
    };
    let desc = RasterPassDesc {
        label: None,
        colors: &[],
        depth_stencil: Some(depth_stencil),
        render_area: Rect {
            x: 0,
            y: 0,
            width: 4,
            height: 4,
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    };
    let mut pass = encoder.begin_raster(&desc).unwrap();
    let rect = Rect {
        x: 1,
        y: 1,
        width: 1,
        height: 1,
    };
    assert_eq!(
        pass.clear_rect(rect, &[], Some(0.5), None),
        Err(Error::InvalidState)
    );
    assert_eq!(
        pass.clear_rect(rect, &[], None, Some(1)),
        Err(Error::InvalidState)
    );
    pass.finish().unwrap();
}
#[test]
fn clear_texture_rejects_invalid_shape_before_backend() {
    let id = DeviceId::allocate().unwrap();
    let other = DeviceId::allocate().unwrap();
    // SAFETY: each mock object is paired with its allocated device identity.
    let device = unsafe {
        Device::<NativeShared>::from_native(id, Arc::new(()), DeviceCapabilities::default())
    };
    let texture = unsafe { Texture::<NativeShared>::from_native(id, Arc::new(())) };
    let foreign = unsafe { Texture::<NativeShared>::from_native(other, Arc::new(())) };
    let mut encoder = CommandEncoder::new(
        &device,
        LaneId {
            device: id,
            ordinal: 0,
        },
    )
    .unwrap();
    let color = TextureClearValue::Color(ClearColor::Float([0.0; 4]));
    assert!(matches!(
        encoder.clear_texture(&foreign, 0..1, 0..1, color),
        Err(Error::WrongDevice)
    ));
    assert!(matches!(
        encoder.clear_texture(&texture, 0..0, 0..1, color),
        Err(Error::InvalidRange)
    ));
    assert!(matches!(
        encoder.clear_texture(&texture, 0..1, 1..1, color),
        Err(Error::InvalidRange)
    ));
    assert!(matches!(
        encoder.clear_texture(&texture, 0..1, 0..1, TextureClearValue::Depth(f32::NAN)),
        Err(Error::InvalidRange)
    ));
    assert!(matches!(
        encoder.clear_texture(&texture, 0..1, 0..1, TextureClearValue::Depth(1.1)),
        Err(Error::InvalidRange)
    ));
    assert!(matches!(
        encoder.clear_texture(&texture, 0..1, 0..1, color),
        Err(Error::Unsupported(_))
    ));
}
