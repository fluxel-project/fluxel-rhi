//! Scene data and deterministic transforms for the dynamic-uniform cube grid.

pub(super) fn model(n: u32, time: f32) -> [u8; 64] {
    // The C++ sample lays out a 5³ lattice at five world-unit intervals and
    // rotates every cube independently. Keeping the transform generation here
    // lets the host path focus on resource setup, recording, and presentation.
    let x = n / 25;
    let y = (n / 5) % 5;
    let z = n % 5;
    let position = [
        -10.0 + x as f32 * 5.0,
        -10.0 + y as f32 * 5.0,
        -10.0 + z as f32 * 5.0,
    ];
    // The source initializes each cube with independent rotations in
    // [-2π, 2π] and then advances it with independent angular velocities.
    // Its time seed is intentionally not reproducible, so use a local stable
    // generator while preserving the same distribution and transform order.
    let mut random = CubeRng::for_cube(n);
    let initial: [f32; 3] = std::array::from_fn(|_| random.signed() * 2.0 * std::f32::consts::PI);
    let speed: [f32; 3] = std::array::from_fn(|_| random.signed());
    let angles: [f32; 3] = std::array::from_fn(|index| initial[index] + time * speed[index]);
    encode_matrix(multiply(
        translation(position),
        multiply(
            rotation_z(angles[2]),
            multiply(rotation_y(angles[1]), rotation_axis_xy(angles[0])),
        ),
    ))
}

struct CubeRng(u32);

impl CubeRng {
    fn for_cube(index: u32) -> Self {
        // Avoid correlated adjacent LCG streams so neighboring z layers do
        // not share the same apparent orientation.
        let mut seed = index.wrapping_add(0x9e37_79b9);
        seed ^= seed >> 16;
        seed = seed.wrapping_mul(0x85eb_ca6b);
        seed ^= seed >> 13;
        seed = seed.wrapping_mul(0xc2b2_ae35);
        Self(seed ^ (seed >> 16))
    }

    fn signed(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
    }
}

pub(super) fn camera_matrix(extent: [u32; 2]) -> [u8; 64] {
    let aspect = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    let f = 1.0 / 30f32.to_radians().tan();
    // Preserve the source's right-handed camera and WebGPU-compatible depth.
    let projection = [
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
        256. / -255.9,
        -1.,
        0.,
        0.,
        -25.6 / 255.9,
        0.,
    ];
    encode_matrix(multiply(projection, translation([0., 0., -30.])))
}

fn translation(position: [f32; 3]) -> [f32; 16] {
    [
        1.,
        0.,
        0.,
        0.,
        0.,
        1.,
        0.,
        0.,
        0.,
        0.,
        1.,
        0.,
        position[0],
        position[1],
        position[2],
        1.,
    ]
}

fn rotation_axis_xy(angle: f32) -> [f32; 16] {
    let (s, c) = angle.sin_cos();
    let half = 0.5 * (1.0 - c);
    let diagonal = c + half;
    let xy = half;
    let z = s * std::f32::consts::FRAC_1_SQRT_2;
    [
        diagonal, xy, -z, 0., xy, diagonal, z, 0., z, -z, c, 0., 0., 0., 0., 1.,
    ]
}

fn rotation_y(angle: f32) -> [f32; 16] {
    let (s, c) = angle.sin_cos();
    [c, 0., -s, 0., 0., 1., 0., 0., s, 0., c, 0., 0., 0., 0., 1.]
}

fn rotation_z(angle: f32) -> [f32; 16] {
    let (s, c) = angle.sin_cos();
    [c, s, 0., 0., -s, c, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]
}

fn multiply(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut output = [0.; 16];
    for column in 0..4 {
        for row in 0..4 {
            output[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
        }
    }
    output
}

fn encode_matrix(matrix: [f32; 16]) -> [u8; 64] {
    let mut bytes = [0; 64];
    for (index, value) in matrix.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub(super) fn cube_vertices() -> [u8; 192] {
    // Exact `generateCube` vertex order and colour values from the C++ demo.
    let p: [f32; 48] = [
        -1., -1., 1., 1., 0., 0., 1., -1., 1., 0., 1., 0., 1., 1., 1., 0., 0., 1., -1., 1., 1., 0.,
        0., 0., -1., -1., -1., 1., 0., 0., 1., -1., -1., 0., 1., 0., 1., 1., -1., 0., 0., 1., -1.,
        1., -1., 0., 0., 0.,
    ];
    let mut bytes = [0; 192];
    for (index, value) in p.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}

pub(super) fn cube_indices() -> [u8; 144] {
    let indices: [u32; 36] = [
        0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7, 4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4,
        3, 2, 6, 6, 7, 3,
    ];
    let mut bytes = [0; 144];
    for (index, value) in indices.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}
