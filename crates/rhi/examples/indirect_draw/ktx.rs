use fluxel_rhi::api::*;

pub(crate) struct Ktx {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) layers: u32,
    pub(crate) mips: u32,
    pub(crate) data: Vec<Vec<Vec<u8>>>,
}
impl Ktx {
    pub(crate) fn parse(b: &[u8]) -> Result<Self> {
        const ID: [u8; 12] = [
            0xAB, b'K', b'T', b'X', b' ', b'1', b'1', 0xBB, 13, 10, 26, 10,
        ];
        if b.len() < 64 || b[..12] != ID {
            return Err(Error::Unsupported("10 expects KTX1"));
        }
        let w = |o| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        if w(12) != 0x04030201 || w(16) != 0x1401 || w(24) != 0x1908 || w(28) != 0x8058 {
            return Err(Error::Unsupported("10 expects RGBA8 KTX1"));
        }
        let (width, height, layers, mips) = (w(36), w(40), w(48).max(1), w(56));
        let mut at = 64 + w(60) as usize;
        let mut data = vec![];
        for mip in 0..mips {
            if at + 4 > b.len() {
                return Err(Error::InvalidRange);
            }
            let image_size = w(at) as usize;
            at += 4;
            let expected = (width >> mip).max(1) as usize * (height >> mip).max(1) as usize * 4;
            // KTX1 writers in our assets use both legal conventions: imageSize
            // can describe one array image or the complete array level.
            let size = if image_size == expected {
                expected
            } else if image_size == expected * layers as usize {
                expected
            } else {
                return Err(Error::Unsupported("10 KTX mip layout"));
            };
            let mut level = vec![];
            for _ in 0..layers {
                let end = at + size;
                if end > b.len() {
                    return Err(Error::InvalidRange);
                }
                level.push(b[at..end].to_vec());
                at = (end + 3) & !3;
            }
            data.push(level);
        }
        Ok(Self {
            width,
            height,
            layers,
            mips,
            data,
        })
    }
}
pub(crate) fn frame_uniform(e: [u32; 2], t: f32) -> [u8; 256] {
    let f = 1.0 / 30f32.to_radians().tan();
    let a = e[0].max(1) as f32 / e[1].max(1) as f32;
    let projection = [
        f / a,
        0.,
        0.,
        0.,
        0.,
        f,
        0.,
        0.,
        0.,
        0.,
        256. / -255.,
        -1.,
        0.,
        0.,
        256. / -255.,
        0.,
    ];
    let (s, c) = (-0.25 * t).sin_cos();
    let view = [
        c, 0., s, 0., 0., 1., 0., 0., -s, 0., c, 0., 0., -2.5, -35., 1.,
    ];
    let mut output = [0; 256];
    for (index, value) in projection.into_iter().chain(view).enumerate() {
        output[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    output
}
