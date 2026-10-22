//! Builds the smoke mip chain with linear-light RGB and linear alpha.
pub fn smoke_mips(mut pixels: Vec<u8>) -> Vec<u8> {
    let mut width = 512usize;
    let mut all = pixels.clone();
    while width > 1 {
        let next_width = width / 2;
        let mut next = vec![0; next_width * next_width * 4];
        for y in 0..next_width {
            for x in 0..next_width {
                for channel in 0..4 {
                    let mut sum = 0.0f32;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let value = pixels[((2 * y + dy) * width + 2 * x + dx) * 4 + channel]
                                as f32
                                / 255.0;
                            sum += if channel == 3 {
                                value
                            } else if value <= 0.04045 {
                                value / 12.92
                            } else {
                                ((value + 0.055) / 1.055).powf(2.4)
                            };
                        }
                    }
                    let value = sum * 0.25;
                    let encoded = if channel == 3 {
                        value
                    } else if value <= 0.0031308 {
                        value * 12.92
                    } else {
                        1.055 * value.powf(1.0 / 2.4) - 0.055
                    };
                    next[(y * next_width + x) * 4 + channel] =
                        (encoded.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        all.extend_from_slice(&next);
        pixels = next;
        width = next_width;
    }
    all
}
