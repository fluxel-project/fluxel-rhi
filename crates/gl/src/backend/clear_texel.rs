//! Encoding of uncompressed color clear values for texture upload fallbacks.
//!
//! Each result is a single native texel, not a command or a cached resource.
use super::api::GlFormat as F;
use fluxel_rhi_core::api::{ClearColor, Error, Result};

// Convert an f32 to a binary float with five exponent bits and `mantissa`
// fraction bits, rounding ties to even. The signed half variant keeps sign;
// packed unsigned variants clamp negative inputs to zero.
fn small_float(value: f32, mantissa: u32, signed: bool) -> u32 {
    let bits = value.to_bits();
    let sign = if signed {
        (bits >> 31) << (5 + mantissa)
    } else {
        0
    };
    if !signed && bits >> 31 != 0 {
        return 0;
    }
    let exponent = ((bits >> 23) & 255) as i32;
    let fraction = bits & 0x7fffff;
    if exponent == 255 {
        return sign
            | (31 << mantissa)
            | if fraction == 0 {
                0
            } else {
                1 << (mantissa - 1)
            };
    }
    let adjusted = exponent - 127 + 15;
    if adjusted >= 31 {
        return sign | (31 << mantissa);
    }
    if exponent == 0 || adjusted < -(mantissa as i32) {
        return sign;
    }
    let (base, shift) = if adjusted <= 0 {
        (fraction | 0x800000, 23 - mantissa + (1 - adjusted) as u32)
    } else {
        (fraction, 23 - mantissa)
    };
    let mut rounded = base >> shift;
    let remainder = base & ((1 << shift) - 1);
    let midpoint = 1 << (shift - 1);
    if remainder > midpoint || (remainder == midpoint && rounded & 1 != 0) {
        rounded += 1;
    }
    if adjusted <= 0 {
        sign | rounded
    } else {
        sign | (((adjusted as u32) << mantissa) + rounded)
    }
}

fn rgb9e5(c: [f32; 4]) -> u32 {
    let values = c.map(|v| {
        if v.is_nan() {
            0.0
        } else {
            v.clamp(0.0, 65408.0)
        }
    });
    let max = values[0].max(values[1]).max(values[2]);
    if max == 0.0 {
        return 0;
    }
    let mut exp = ((max.log2().floor() as i32) + 16).clamp(0, 31);
    let mut scale = 2_f32.powi(exp - 24);
    if (max / scale).round() >= 512.0 && exp < 31 {
        exp += 1;
        scale *= 2.0;
    }
    let component = |v: f32| ((v / scale).round() as u32).min(511);
    component(values[0])
        | (component(values[1]) << 9)
        | (component(values[2]) << 18)
        | ((exp as u32) << 27)
}

pub(crate) fn encode(format: F, color: ClearColor) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let integer = match format {
        F::R8Uint => Some((1, 1, false)),
        F::Rg8Uint => Some((2, 1, false)),
        F::Rgba8Uint => Some((4, 1, false)),
        F::R8Sint => Some((1, 1, true)),
        F::Rg8Sint => Some((2, 1, true)),
        F::Rgba8Sint => Some((4, 1, true)),
        F::R16Uint => Some((1, 2, false)),
        F::Rg16Uint => Some((2, 2, false)),
        F::Rgba16Uint => Some((4, 2, false)),
        F::R16Sint => Some((1, 2, true)),
        F::Rg16Sint => Some((2, 2, true)),
        F::Rgba16Sint => Some((4, 2, true)),
        F::R32Uint => Some((1, 4, false)),
        F::Rg32Uint => Some((2, 4, false)),
        F::Rgba32Uint => Some((4, 4, false)),
        F::R32Sint => Some((1, 4, true)),
        F::Rg32Sint => Some((2, 4, true)),
        F::Rgba32Sint => Some((4, 4, true)),
        _ => None,
    };
    if let Some((channels, width, signed)) = integer {
        for index in 0..channels {
            let word = match (color, signed, width) {
                (ClearColor::Uint(c), false, 1) => c[index].min(u8::MAX as u32),
                (ClearColor::Uint(c), false, 2) => c[index].min(u16::MAX as u32),
                (ClearColor::Uint(c), false, _) => c[index],
                (ClearColor::Sint(c), true, 1) => {
                    c[index].clamp(i8::MIN as i32, i8::MAX as i32) as u32
                }
                (ClearColor::Sint(c), true, 2) => {
                    c[index].clamp(i16::MIN as i32, i16::MAX as i32) as u32
                }
                (ClearColor::Sint(c), true, _) => c[index] as u32,
                _ => return Err(Error::Unsupported("GL texture clear upload value")),
            };
            out.extend_from_slice(&word.to_le_bytes()[..width]);
        }
        return Ok(out);
    }
    if format == F::Rgb10a2Uint {
        let ClearColor::Uint(c) = color else {
            return Err(Error::Unsupported("GL texture clear upload value"));
        };
        let word = c[0].min(1023) | c[1].min(1023) << 10 | c[2].min(1023) << 20 | c[3].min(3) << 30;
        return Ok(word.to_le_bytes().to_vec());
    }
    let ClearColor::Float(mut c) = color else {
        return Err(Error::Unsupported("GL texture clear upload value"));
    };
    if format == F::Rgba8Srgb {
        for v in &mut c[..3] {
            let x = v.clamp(0., 1.);
            *v = if x <= 0.0031308 {
                x * 12.92
            } else {
                1.055 * x.powf(1. / 2.4) - 0.055
            };
        }
    }
    if format == F::Rgb10a2Unorm {
        let cv = |v: f32, max: f32| (v.clamp(0., 1.) * max).round() as u32;
        let word =
            cv(c[0], 1023.) | cv(c[1], 1023.) << 10 | cv(c[2], 1023.) << 20 | cv(c[3], 3.) << 30;
        return Ok(word.to_le_bytes().to_vec());
    }
    let (channels, kind) = match format {
        F::R8Unorm => (1, 5),
        F::Rg8Unorm => (2, 5),
        F::Rgba8Unorm | F::Rgba8Srgb => (4, 5),
        F::R8Snorm => (1, 0),
        F::Rg8Snorm => (2, 0),
        F::Rgba8Snorm => (4, 0),
        F::R16Snorm => (1, 1),
        F::Rg16Snorm => (2, 1),
        F::Rgba16Snorm => (4, 1),
        F::R16Unorm => (1, 2),
        F::Rg16Unorm => (2, 2),
        F::Rgba16Unorm => (4, 2),
        F::R16Float => (1, 3),
        F::Rg16Float => (2, 3),
        F::Rgba16Float => (4, 3),
        F::R32Float => (1, 4),
        F::Rg32Float => (2, 4),
        F::Rgba32Float => (4, 4),
        F::Rgb9e5Ufloat => {
            out.extend_from_slice(&rgb9e5(c).to_le_bytes());
            return Ok(out);
        }
        F::Rg11b10Ufloat => {
            let word = small_float(c[0], 6, false)
                | (small_float(c[1], 6, false) << 11)
                | (small_float(c[2], 5, false) << 22);
            out.extend_from_slice(&word.to_le_bytes());
            return Ok(out);
        }
        _ => return Err(Error::Unsupported("GL texture clear upload format")),
    };
    for &v in &c[..channels] {
        match kind {
            0 => out.push(((v.clamp(-1., 1.) * 127.).round() as i8) as u8),
            1 => out.extend_from_slice(&((v.clamp(-1., 1.) * 32767.).round() as i16).to_le_bytes()),
            2 => out.extend_from_slice(&((v.clamp(0., 1.) * 65535.).round() as u16).to_le_bytes()),
            3 => out.extend_from_slice(&(small_float(v, 10, true) as u16).to_le_bytes()),
            4 => out.extend_from_slice(&v.to_le_bytes()),
            _ => out.push((v.clamp(0., 1.) * 255.).round() as u8),
        }
    }
    Ok(out)
}

#[cfg(test)]
#[path = "../tests/backend/clear_texel__tests.rs"]
mod tests;
