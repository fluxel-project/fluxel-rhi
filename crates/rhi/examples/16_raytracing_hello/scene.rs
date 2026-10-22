//! Scene data and ray generation constants from Microsoft's DXR Hello World.
//!
//! Coordinates are in the original screen-space convention; the shader maps
//! dispatch indices into this viewport without a half-pixel offset.

pub const VERTICES: [f32; 9] = [0.0, -0.7, 1.0, -0.7, 0.7, 1.0, 0.7, 0.7, 1.0];
pub const INDICES: [u16; 3] = [0, 1, 2];

/// Eight 32-bit constants in the official local ray-generation root signature.
pub fn raygen_constants(width: u32, height: u32) -> [f32; 8] {
    let aspect = width as f32 / height as f32;
    let (x_border, y_border) = if width <= height {
        (0.1, 0.1 * aspect)
    } else {
        (0.1 / aspect, 0.1)
    };
    [
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0 + x_border,
        -1.0 + y_border,
        1.0 - x_border,
        1.0 - y_border,
    ]
}
