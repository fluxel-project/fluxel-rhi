//! Builds the smoke mip chain with linear-light RGB and linear alpha.
fn mips(mut pixels: Vec<u8>, mut width: usize) -> Vec<u8> {
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

/// The shader addresses the smoke and NVIDIA logo mip chains in one buffer.
pub fn particle_mips(smoke: Vec<u8>) -> Result<Vec<u8>, String> {
    let mut all = mips(smoke, 512);
    let decoder = png::Decoder::new(std::io::Cursor::new(include_bytes!(
        "../assets/19_ray_query_particles/nvidia-logo.png"
    )));
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let mut raw = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut raw).map_err(|e| e.to_string())?;
    if (info.width, info.height, info.color_type, info.bit_depth)
        != (768, 768, png::ColorType::Rgb, png::BitDepth::Eight)
    {
        return Err("invalid logo texture".into());
    }
    let mut rgba = Vec::with_capacity(768 * 768 * 4);
    for rgb in raw[..info.buffer_size()].chunks_exact(3) {
        rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    }
    all.extend(mips(rgba, 768));
    Ok(all)
}
