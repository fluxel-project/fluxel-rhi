use super::*;

#[test]
fn orbit_uses_previous_angles_and_alt_pan_stays_horizontal() {
    let mut camera = Camera::default();
    let initial = camera.advance();
    camera.press(Drag::Orbit);
    camera.pointer(80., 40.);
    let delayed = camera.advance();
    assert_eq!(initial.forward, delayed.forward);
    let turned = camera.advance();
    assert_ne!(initial.forward, turned.forward);
    camera.release(Drag::Orbit);
    camera.alt(true);
    camera.press(Drag::Pan);
    camera.pointer(80., 100.);
    let panned = camera.advance();
    assert!((turned.position[1] - panned.position[1]).abs() < 1e-6);
    assert!((turned.position[0] - panned.position[0]).abs() > 0.01);
    assert!((turned.position[2] - panned.position[2]).abs() > 0.01);
}
