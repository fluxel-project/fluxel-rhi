//! Static first-frame constants produced by the reference DirectXMath camera.

/// Initial camera (0, 5, 35), yaw pi, 60-degree vertical field of view,
/// 1280/720 aspect, near 1 and far 10000. The matrix bits preserve the
/// reference's DirectXMath single-precision evaluation order.
pub const CAMERA_BITS: [u32; 32] = [
    0x3f800000, 0x00000000, 0xb3bbbd2e, 0x364d56ea, 0x80000000, 0x3f800000, 0x00000000, 0xc0a00000,
    0x33bbbd2e, 0x00000000, 0x3f800000, 0xc20c0000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000,
    0x3f796a52, 0x00000000, 0xb3b6e8fa, 0x36480ed1, 0x00000000, 0x3fddb3d7, 0x00000000, 0xc10a9066,
    0xb3bbc1fd, 0x00000000, 0xbf800347, 0x4208037c, 0xb3bbbd2e, 0x00000000, 0xbf800000, 0x420c0000,
];

pub fn globals() -> Vec<u8> {
    let mut bytes: Vec<u8> = CAMERA_BITS.iter().flat_map(|v| v.to_le_bytes()).collect();
    bytes.extend_from_slice(&1u32.to_le_bytes()); // DrawMeshlets
    bytes.resize(256, 0);
    bytes
}

pub fn identity_instance() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(128);
    for _ in 0..2 {
        for row in 0..4 {
            for col in 0..4 {
                bytes.extend_from_slice(&if row == col { 1.0f32 } else { 0.0 }.to_le_bytes());
            }
        }
    }
    bytes
}
