//! Third-person camera input and view state for the interactive example.
use std::f32::consts::{FRAC_PI_2, PI};

#[derive(Clone, Copy)]
pub struct View {
    pub position: [f32; 3],
    pub forward: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
}
#[derive(Clone, Copy, PartialEq)]
pub enum Drag {
    Orbit,
    Pan,
}
pub struct Camera {
    target: [f32; 3],
    distance: f32,
    yaw: f32,
    pitch: f32,
    position: (f32, f32),
    previous: (f32, f32),
    drag: Option<Drag>,
    alt: bool,
}
impl Default for Camera {
    fn default() -> Self {
        Self {
            target: [-2., 2., -2.],
            distance: 6.,
            yaw: 225. * PI / 180.,
            pitch: 20. * PI / 180.,
            position: (0., 0.),
            previous: (0., 0.),
            drag: None,
            alt: false,
        }
    }
}
impl Camera {
    pub fn pointer(&mut self, x: f32, y: f32) {
        self.position = (x, y);
    }
    pub fn press(&mut self, drag: Drag) {
        if self.drag.is_none() {
            self.drag = Some(drag);
            self.previous = self.position;
        }
    }
    pub fn release(&mut self, drag: Drag) {
        if self.drag == Some(drag) {
            self.drag = None;
        }
    }
    pub fn reset_input(&mut self) {
        self.drag = None;
        self.alt = false;
    }
    pub fn alt(&mut self, pressed: bool) {
        self.alt = pressed;
    }
    pub fn scroll(&mut self, y: f32) {
        self.distance = (self.distance * if y < 0. { 1.15 } else { 1. / 1.15 }).clamp(0., f32::MAX);
    }
    pub fn advance(&mut self) -> View {
        // Donut computes targetRotation before applying orbit input, so the
        // updated angles are visible in the following frame.
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let forward = [sy * cp, -sp, cy * cp];
        let right = [-cy, 0., sy];
        let up = [-sy * sp, cp, -cy * sp];
        let (dx, dy) = (
            self.position.0 - self.previous.0,
            self.position.1 - self.previous.1,
        );
        match self.drag {
            Some(Drag::Orbit) => {
                self.yaw -= 0.005 * dx;
                self.pitch = (self.pitch + 0.005 * dy).clamp(-FRAC_PI_2, FRAC_PI_2);
            }
            Some(Drag::Pan) => {
                let scale = 2. * self.distance * (PI / 8.).tan() / 709.;
                let horizontal = [sy, 0., cy];
                for i in 0..3 {
                    self.target[i] += -dx * scale * right[i]
                        + dy * scale * if self.alt { 1.5 * horizontal[i] } else { up[i] };
                }
            }
            None => {}
        }
        self.previous = self.position;
        View {
            position: std::array::from_fn(|i| self.target[i] - self.distance * forward[i]),
            forward,
            right,
            up,
        }
    }
}

#[cfg(test)]
mod tests {
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
}
