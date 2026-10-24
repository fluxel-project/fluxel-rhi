//! WebGL constant vertex attributes for Core bindings with zero stride.
//!
//! WebGL's pointer stride zero means tightly packed. Generic attributes instead
//! represent one fixed value; refresh on every draw because the GPU can have
//! modified the source buffer after binding. The synchronous read is necessary
//! to preserve this contract on WebGL2, which lacks vertex binding stride zero.

use super::owner::{BrowserBuffer, BrowserEncoder};
use crate::backend::gl::api::{GlVertexAttribute, GlVertexFormat as F};
use fluxel_rhi_core::api::{Error, Result};

fn half(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = (bits >> 10) & 31;
    let frac = u32::from(bits & 1023);
    let word = match exp {
        0 if frac == 0 => sign,
        0 => {
            let shift = frac.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((frac << shift & 1023) << 13)
        }
        31 => sign | 0x7f800000 | (frac << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (frac << 13),
    };
    f32::from_bits(word)
}

impl BrowserEncoder {
    pub(crate) fn refresh_constant_vertex_attribute(
        &mut self,
        buffer: &BrowserBuffer,
        attribute: &GlVertexAttribute,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        self.same_owner(&buffer.owner)?;
        self.require_raster()?;
        let (lanes, width, kind) = match attribute.format {
            F::Uint8x2 => (2, 1, 0),
            F::Uint8x4 => (4, 1, 0),
            F::Sint8x2 => (2, 1, 1),
            F::Sint8x4 => (4, 1, 1),
            F::Unorm8x2 => (2, 1, 2),
            F::Unorm8x4 => (4, 1, 2),
            F::Snorm8x2 => (2, 1, 3),
            F::Snorm8x4 => (4, 1, 3),
            F::Uint16x2 => (2, 2, 0),
            F::Uint16x4 => (4, 2, 0),
            F::Sint16x2 => (2, 2, 1),
            F::Sint16x4 => (4, 2, 1),
            F::Unorm16x2 => (2, 2, 2),
            F::Unorm16x4 => (4, 2, 2),
            F::Snorm16x2 => (2, 2, 3),
            F::Snorm16x4 => (4, 2, 3),
            F::Float16x2 => (2, 2, 4),
            F::Float16x4 => (4, 2, 4),
            F::Float64 | F::Float64x2 | F::Float64x3 | F::Float64x4 => {
                return Err(Error::Unsupported("WebGL2 Float64 vertex attributes"));
            }
            F::Float32 => (1, 4, 5),
            F::Float32x2 => (2, 4, 5),
            F::Float32x3 => (3, 4, 5),
            F::Float32x4 => (4, 4, 5),
            F::Uint32 => (1, 4, 0),
            F::Uint32x2 => (2, 4, 0),
            F::Uint32x3 => (3, 4, 0),
            F::Uint32x4 => (4, 4, 0),
            F::Sint32 => (1, 4, 1),
            F::Sint32x2 => (2, 4, 1),
            F::Sint32x3 => (3, 4, 1),
            F::Sint32x4 => (4, 4, 1),
        };
        let start = u64::from(attribute.offset);
        let bytes = (lanes * width) as u64;
        if start.checked_add(bytes).is_none_or(|end| end > size) {
            return Err(Error::InvalidRange);
        }
        let start = offset.checked_add(start).ok_or(Error::InvalidRange)?;
        let data = self.owner.read_buffer(buffer, start, bytes)?;
        let mut u = [0_u32, 0, 0, 1];
        let mut i = [0_i32, 0, 0, 1];
        let mut f = [0_f32, 0., 0., 1.];
        for n in 0..lanes {
            let at = n * width;
            let v = match width {
                1 => u32::from(data[at]),
                2 => u32::from(u16::from_le_bytes([data[at], data[at + 1]])),
                _ => u32::from_le_bytes(data[at..at + 4].try_into().unwrap()),
            };
            let signed = match width {
                1 => v as u8 as i8 as i32,
                2 => v as u16 as i16 as i32,
                _ => v as i32,
            };
            u[n] = v;
            i[n] = signed;
            f[n] = match kind {
                2 => v as f32 / if width == 1 { 255. } else { 65535. },
                3 => (signed as f32 / if width == 1 { 127. } else { 32767. }).max(-1.),
                4 => half(v as u16),
                5 => f32::from_bits(v),
                _ => 0.,
            };
        }
        self.with_context("refresh_constant_vertex_attribute", |raw| {
            raw.disable_vertex_attrib_array(attribute.location);
            match kind {
                0 => raw.vertex_attrib_i4ui(attribute.location, u[0], u[1], u[2], u[3]),
                1 => raw.vertex_attrib_i4i(attribute.location, i[0], i[1], i[2], i[3]),
                _ => raw.vertex_attrib4f(attribute.location, f[0], f[1], f[2], f[3]),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::half;

    #[test]
    fn half_float_boundaries() {
        assert_eq!(half(0x0001), 2_f32.powi(-24));
        assert_eq!(half(0x03ff), 1023. * 2_f32.powi(-24));
        assert_eq!(half(0x0400), 2_f32.powi(-14));
        assert_eq!(half(0x3c00), 1.);
        assert_eq!(half(0xfc00), f32::NEG_INFINITY);
        assert!(half(0x7e00).is_nan());
    }
}
