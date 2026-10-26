//! Immediate WebGL uniform writes for lowered Core push constants.
//!
//! Core passes unpacked scalar words in column-major order; structure padding
//! and member offsets are resolved before entering this native boundary.
use super::owner::{BrowserEncoder, BrowserRasterProgram};
use fluxel_rhi_core::api::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub(crate) enum BrowserPushUniformKind {
    Float { lanes: u8 },
    Sint { lanes: u8 },
    Uint { lanes: u8 },
    Bool { lanes: u8 },
    Matrix { columns: u8, rows: u8 },
}

impl BrowserEncoder {
    pub(crate) fn upload_push_uniform(
        &mut self,
        program: &BrowserRasterProgram,
        name: &str,
        kind: BrowserPushUniformKind,
        words: &[u32],
    ) -> Result<()> {
        self.same_owner(&program.owner)?;
        self.require_raster()?;
        let expected = match kind {
            BrowserPushUniformKind::Float { lanes }
            | BrowserPushUniformKind::Sint { lanes }
            | BrowserPushUniformKind::Uint { lanes }
            | BrowserPushUniformKind::Bool { lanes }
                if (1..=4).contains(&lanes) =>
            {
                usize::from(lanes)
            }
            BrowserPushUniformKind::Matrix { columns, rows }
                if (2..=4).contains(&columns) && (2..=4).contains(&rows) =>
            {
                usize::from(columns) * usize::from(rows)
            }
            _ => return Err(Error::InvalidRange),
        };
        if words.len() != expected {
            return Err(Error::InvalidRange);
        }
        let mut floats = [0f32; 16];
        let mut integers = [0i32; 4];
        for (i, word) in words.iter().enumerate() {
            floats[i] = f32::from_bits(*word);
            if i < 4 {
                integers[i] = *word as i32;
            }
        }
        let location = program.uniform_location(name)?;
        self.with_context("upload_push_uniform", |raw| {
            let Some(location) = location.as_ref() else {
                return;
            };
            let location = Some(location);
            match kind {
                BrowserPushUniformKind::Float { lanes } => match lanes {
                    1 => raw.uniform1fv_with_f32_array(location, &floats[..1]),
                    2 => raw.uniform2fv_with_f32_array(location, &floats[..2]),
                    3 => raw.uniform3fv_with_f32_array(location, &floats[..3]),
                    4 => raw.uniform4fv_with_f32_array(location, &floats[..4]),
                    _ => unreachable!(),
                },
                BrowserPushUniformKind::Sint { lanes } | BrowserPushUniformKind::Bool { lanes } => {
                    match lanes {
                        1 => raw.uniform1iv_with_i32_array(location, &integers[..1]),
                        2 => raw.uniform2iv_with_i32_array(location, &integers[..2]),
                        3 => raw.uniform3iv_with_i32_array(location, &integers[..3]),
                        4 => raw.uniform4iv_with_i32_array(location, &integers[..4]),
                        _ => unreachable!(),
                    }
                }
                BrowserPushUniformKind::Uint { lanes } => match lanes {
                    1 => raw.uniform1uiv_with_u32_array(location, &words[..1]),
                    2 => raw.uniform2uiv_with_u32_array(location, &words[..2]),
                    3 => raw.uniform3uiv_with_u32_array(location, &words[..3]),
                    4 => raw.uniform4uiv_with_u32_array(location, &words[..4]),
                    _ => unreachable!(),
                },
                BrowserPushUniformKind::Matrix { columns, rows } => match (columns, rows) {
                    (2, 2) => {
                        raw.uniform_matrix2fv_with_f32_array(location, false, &floats[..expected])
                    }
                    (2, 3) => {
                        raw.uniform_matrix2x3fv_with_f32_array(location, false, &floats[..expected])
                    }
                    (2, 4) => {
                        raw.uniform_matrix2x4fv_with_f32_array(location, false, &floats[..expected])
                    }
                    (3, 2) => {
                        raw.uniform_matrix3x2fv_with_f32_array(location, false, &floats[..expected])
                    }
                    (3, 3) => {
                        raw.uniform_matrix3fv_with_f32_array(location, false, &floats[..expected])
                    }
                    (3, 4) => {
                        raw.uniform_matrix3x4fv_with_f32_array(location, false, &floats[..expected])
                    }
                    (4, 2) => {
                        raw.uniform_matrix4x2fv_with_f32_array(location, false, &floats[..expected])
                    }
                    (4, 3) => {
                        raw.uniform_matrix4x3fv_with_f32_array(location, false, &floats[..expected])
                    }
                    (4, 4) => {
                        raw.uniform_matrix4fv_with_f32_array(location, false, &floats[..expected])
                    }
                    _ => unreachable!(),
                },
            }
        })
    }
}
