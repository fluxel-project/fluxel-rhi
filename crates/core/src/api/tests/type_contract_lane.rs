//! Lane-admission regressions for the Core test backend.

use super::*;

#[test]
fn deferred_encoder_rejects_an_undeclared_lane_before_recording() {
    let id = DeviceId::allocate().unwrap();
    let declared = LaneId {
        device: id,
        ordinal: 0,
    };
    let undeclared = LaneId {
        device: id,
        ordinal: 1,
    };
    let caps = DeviceCapabilities {
        lanes: Arc::from([LaneInfo {
            id: declared,
            class: LaneClass::Graphics,
            routes: Vec::new(),
        }]),
        ..DeviceCapabilities::default()
    };
    // SAFETY: the fixture pairs the mock native device with its identity and lane facts.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(matches!(device.encoder(undeclared), Err(Error::WrongLane)));
}
