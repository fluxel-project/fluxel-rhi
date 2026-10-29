//! Type profile, marker, and dynamic raster-state contracts.
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

fn assert_send_sync<T: Send + Sync>() {}
fn assert_send<T: Send>() {}
fn generic_renderer_optional_compute<B: Backend + CommandBackend>(
    device: &Device<B>,
    encoder: &mut CommandEncoder<B>,
) -> Result<()> {
    if let Some(compute) = device.compute() {
        let mut pass = compute.begin(encoder)?;
        pass.dispatch([1, 1, 1])?;
        pass.finish()?;
    }
    Ok(())
}
#[test]
fn concrete_shared_and_local_profiles_typecheck() {
    assert_send_sync::<Device<NativeShared>>();
    assert_send_sync::<Buffer<NativeShared>>();
    assert_send_sync::<Queue<NativeShared>>();
    assert_send::<CommandEncoder<NativeShared>>();
    assert_send::<RecordedWork<NativeShared>>();
    assert_send_sync::<CompletionPoint<NativeShared>>();
    let id = DeviceId::allocate().unwrap();
    // SAFETY: this fixture creates matching native values for the allocated device ID.
    let native = unsafe {
        Device::<NativeShared>::from_native(id, Arc::new(()), DeviceCapabilities::default())
    };
    // SAFETY: this fixture creates matching native values for the allocated device ID.
    let local =
        unsafe { Device::<GlLocal>::from_native(id, Rc::new(()), DeviceCapabilities::default()) };
    assert!(native.memory_observation().is_none());
    assert!(local.memory_observation().is_none());
    let lane = LaneId {
        device: id,
        ordinal: 0,
    };
    let mut encoder = CommandEncoder::<NativeShared>::new(&native, lane).unwrap();
    generic_renderer_optional_compute(&native, &mut encoder).unwrap();
    let _ = CommandEncoder::<GlLocal>::new(&local, lane).unwrap();
}
#[test]
fn raster_draw_rejects_inverted_ranges_before_backend() {
    DRAW_CALLS.with(|calls| calls.set(0));
    let id = DeviceId::allocate().unwrap();
    // SAFETY: the fixture pairs this native device with its newly allocated ID.
    let device = unsafe {
        Device::<NativeShared>::from_native(id, Arc::new(()), DeviceCapabilities::default())
    };
    let lane = LaneId {
        device: id,
        ordinal: 0,
    };
    let mut encoder = CommandEncoder::new(&device, lane).unwrap();
    let colors = [];
    let mut pass = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
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
        })
        .unwrap();
    assert!(matches!(
        pass.draw(Range { start: 4, end: 3 }, 0..1),
        Err(Error::InvalidRange)
    ));
    assert!(matches!(
        pass.draw_indexed(0..1, 0, Range { start: 2, end: 1 }),
        Err(Error::InvalidRange)
    ));
    DRAW_CALLS.with(|calls| assert_eq!(calls.get(), 0));
    pass.draw(0..3, 0..1).unwrap();
    DRAW_CALLS.with(|calls| assert_eq!(calls.get(), 1));
    pass.finish().unwrap();
}
#[test]
fn raster_pass_label_is_recorded_as_balanced_native_group() {
    DEBUG_PUSHES.with(|count| count.set(0));
    DEBUG_POPS.with(|count| count.set(0));
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.debug_markers = true;
    caps.debug_markers_raster_pass = true;
    // SAFETY: the mock backend does not issue GPU work or retain resources.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    let lane = LaneId {
        device: id,
        ordinal: 0,
    };
    let mut encoder = CommandEncoder::new(&device, lane).unwrap();
    let colors = [];
    let pass = encoder
        .begin_raster(&RasterPassDesc {
            label: Some("opaque forward pass"),
            colors: &colors,
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
        })
        .unwrap();
    pass.finish().unwrap();
    DEBUG_PUSHES.with(|count| assert_eq!(count.get(), 1));
    DEBUG_POPS.with(|count| assert_eq!(count.get(), 1));

    let mut encoder = CommandEncoder::new(&device, lane).unwrap();
    let mut pass = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
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
        })
        .unwrap();
    assert!(matches!(pass.pop_debug_group(), Err(Error::InvalidState)));
    pass.push_debug_group("unfinished child").unwrap();
    assert!(matches!(pass.finish(), Err(Error::InvalidState)));
    DEBUG_PUSHES.with(|count| assert_eq!(count.get(), 2));
    DEBUG_POPS.with(|count| assert_eq!(count.get(), 1));
}
#[test]
fn aggregate_debug_marker_capability_is_derived_from_scopes() {
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.debug_markers_raster_pass = true;
    // SAFETY: this mock capability record has no native GPU operations.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(device.capabilities().debug_markers);

    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.debug_markers = true;
    // SAFETY: this mock capability record has no native GPU operations.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(!device.capabilities().debug_markers);
}

#[test]
fn dynamic_depth_bias_values_are_finite_and_pipeline_clone_preserves_mode() {
    let initial = DynamicDepthBias {
        constant: 2,
        slope_scale: -2.0,
        clamp: 0.25,
    };
    assert!(initial.is_finite());
    assert!(
        !DynamicDepthBias {
            slope_scale: f32::NAN,
            ..initial
        }
        .is_finite()
    );
    let id = DeviceId::allocate().unwrap();
    // SAFETY: this test only checks wrapper metadata and never uses the fake native handle.
    let pipeline = unsafe {
        RasterPipeline::<NativeShared>::from_native_with_dynamic_depth_bias(
            id,
            Arc::new(()),
            Some(initial),
        )
    };
    let clone = pipeline.clone();
    assert_eq!(clone.dynamic_depth_bias, Some(initial));
}
