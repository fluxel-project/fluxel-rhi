//! Convert Core specialization values to Naga pipeline constants.

use super::*;

pub(super) fn constant_value(value: api::SpecializationValue) -> Result<f64> {
    use api::SpecializationValue as V;
    Ok(match value {
        V::Bool(v) => {
            if v {
                1.0
            } else {
                0.0
            }
        }
        V::U8(v) => v as f64,
        V::I8(v) => v as f64,
        V::U16(v) => v as f64,
        V::I16(v) => v as f64,
        V::U32(v) => v as f64,
        V::I32(v) => v as f64,
        V::U64(v) => {
            let f = v as f64;
            if f as u128 != v as u128 {
                return Err(Error::Shader(
                    "override integer cannot be represented".into(),
                ));
            }
            f
        }
        V::I64(v) => {
            let f = v as f64;
            if f as i128 != v as i128 {
                return Err(Error::Shader(
                    "override integer cannot be represented".into(),
                ));
            }
            f
        }
        V::F32(v) => v as f64,
        V::F64(v) => v,
        V::F16Bits(v) => {
            let sign = if v & 0x8000 == 0 { 1.0 } else { -1.0 };
            let exponent = (v >> 10) & 31;
            let fraction = (v & 1023) as f64;
            sign * match exponent {
                0 => fraction * 2f64.powi(-24),
                31 if fraction == 0.0 => f64::INFINITY,
                31 => f64::NAN,
                _ => (1.0 + fraction / 1024.0) * 2f64.powi(exponent as i32 - 15),
            }
        }
    })
}
