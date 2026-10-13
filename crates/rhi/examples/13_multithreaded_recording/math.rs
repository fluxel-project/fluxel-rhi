//! Deterministic transform generation for the UFO scene.
//!
//! Worker scheduling never affects these CPU values, so all secondary-recording
//! paths produce the same geometry placement and colors.

use super::{UFO_COUNT, UNIFORM};

pub(super) fn transforms(extent: [u32; 2], frame: u32) -> Vec<u8> {
    let mut out = vec![0; UFO_COUNT as usize * UNIFORM as usize];
    let p = perspective(
        60f32.to_radians(),
        extent[0].max(1) as f32 / extent[1].max(1) as f32,
        0.1,
        256.,
    );
    let view = translate(0., 0., 32.5);
    let time = frame as f32 * 0.012;
    for i in 0..UFO_COUNT as usize {
        // Same spherical placement and per-object animation distribution as
        // `multithreading.cpp`, with a stable seed so worker scheduling never
        // changes the picture. Native clipping rejects the far hemisphere;
        // depth testing orders all remaining packets independently of threads.
        let n = i as f32 + 0.5;
        let theta = core::f32::consts::TAU * hash01(n * 12.9898);
        let phi = (1.0 - 2.0 * hash01(n * 78.233)).acos();
        let x = phi.sin() * theta.cos() * 35.0;
        let z = phi.cos() * 35.0;
        let cycle = hash01(n * 37.719) + time * 0.15;
        let direction = if hash01(n * 19.19) < 0.5 { -1.0 } else { 1.0 };
        let rotation_speed_degrees = (2.0 + 4.0 * hash01(n * 41.31)) * direction;
        let y = (core::f32::consts::TAU * cycle).sin() * 2.5;
        let yaw = core::f32::consts::TAU * hash01(n * 91.17)
            + (time * 2.5 * rotation_speed_degrees).to_radians();
        let spin = core::f32::consts::TAU * cycle * direction;
        let size = 0.75 + hash01(n * 53.11) * 0.5;
        let m = multiply(
            translate(x, y, z),
            multiply(
                rotate_x(-(core::f32::consts::TAU * cycle).sin() * 0.25 * direction),
                multiply(
                    // This is Rx(tilt) × Ry(yaw) × Ry(deltaT spin), matching
                    // the native sample's GLM call order.
                    rotate_y(yaw + spin),
                    scale(size),
                ),
            ),
        );
        let record = &mut out[i * UNIFORM as usize..];
        matrix(record, multiply(p, multiply(view, m)));
        let tint = instance_tint(i as u32);
        for (component, value) in tint.into_iter().enumerate() {
            record[64 + component * 4..68 + component * 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    out
}
pub(super) fn star_matrix(extent: [u32; 2]) -> [f32; 16] {
    let projection = perspective(
        60f32.to_radians(),
        extent[0].max(1) as f32 / extent[1].max(1) as f32,
        0.1,
        256.0,
    );
    // C++ uses projection × view with the camera translation cleared, then
    // scales its unit sphere by two. This camera has no rotation.
    multiply(projection, scale(2.0))
}
pub(super) fn matrix_bytes(value: [f32; 16]) -> [u8; 64] {
    let mut bytes = [0; 64];
    matrix(&mut bytes, value);
    bytes
}
pub(crate) fn instance_tint(index: u32) -> [f32; 4] {
    // C++ samples three independent uniform components for every object.
    // Deriving all three from stable salts preserves that distribution without
    // coupling colours to worker execution order.
    let n = index as f32 + 0.5;
    [
        hash01(n * 17.131),
        hash01(n * 43.723),
        hash01(n * 97.409),
        1.0,
    ]
}
pub(crate) fn hash01(value: f32) -> f32 {
    (value.sin() * 43_758.547).fract().abs()
}
pub(crate) fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1. / (fov * 0.5).tan();
    [
        f / aspect,
        0.,
        0.,
        0.,
        0.,
        f,
        0.,
        0.,
        0.,
        0.,
        far / (far - near),
        1.,
        0.,
        0.,
        -near * far / (far - near),
        0.,
    ]
}
pub(crate) fn translate(x: f32, y: f32, z: f32) -> [f32; 16] {
    [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., x, y, z, 1.]
}
pub(crate) fn scale(s: f32) -> [f32; 16] {
    [s, 0., 0., 0., 0., s, 0., 0., 0., 0., s, 0., 0., 0., 0., 1.]
}
pub(crate) fn rotate_y(a: f32) -> [f32; 16] {
    let (s, c) = a.sin_cos();
    [c, 0., -s, 0., 0., 1., 0., 0., s, 0., c, 0., 0., 0., 0., 1.]
}
pub(crate) fn rotate_x(a: f32) -> [f32; 16] {
    let (s, c) = a.sin_cos();
    [1., 0., 0., 0., 0., c, s, 0., 0., -s, c, 0., 0., 0., 0., 1.]
}
pub(crate) fn multiply(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut r = [0.; 16];
    for c in 0..4 {
        for row in 0..4 {
            r[c * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[c * 4 + k]).sum();
        }
    }
    r
}
pub(crate) fn matrix(bytes: &mut [u8], m: [f32; 16]) {
    for (i, v) in m.into_iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
}
