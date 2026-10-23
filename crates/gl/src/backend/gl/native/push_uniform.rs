//! Immediate uploads of reflected push constant leaves to a linked program.
//!
//! Matrices arrive tightly packed in column-major order. Optimized-away leaves
//! have no location and need no upload; no commands or locations are cached.

use super::provider::NativeGlProvider;
use crate::backend::gl::api::{GlError, GlFamilyApi as _, ProgramId};
use glow::HasContext as _;

pub(crate) enum GlPushUniformValue {
    Float(Vec<f32>),
    Sint(Vec<i32>),
    Uint(Vec<u32>),
    Matrix {
        columns: u8,
        rows: u8,
        values: Vec<f32>,
    },
}

impl NativeGlProvider {
    /// Uploads the mip count associated with a texture-size lowering uniform.
    /// An optimized-away uniform has no location and requires no upload.
    pub(crate) fn set_texture_mip_count(
        &mut self,
        program: ProgramId,
        name: &str,
        count: u32,
    ) -> Result<(), GlError> {
        const OP: &str = "set-texture-mip-count";
        self.assert_ready(OP)?;
        if count == 0 || name.is_empty() || name.contains('\0') {
            return Err(Self::validation(OP, "invalid mip count or uniform name"));
        }
        let raw = self.program(OP, program)?.raw;
        // SAFETY: this is a live linked program owned by the current context.
        let location = unsafe { self.gl.get_uniform_location(raw, name) };
        self.driver_error(OP)?;
        let Some(location) = location else {
            return Ok(());
        };
        self.ensure_program(OP, program)?;
        // SAFETY: the program is selected and this uniform is generated as uint.
        unsafe {
            self.gl.uniform_1_u32(Some(&location), count);
        }
        self.driver_error(OP)
    }

    /// Reports whether the linked program retained a reflected uniform leaf.
    pub(crate) fn has_uniform(&self, program: ProgramId, name: &str) -> Result<bool, GlError> {
        const OP: &str = "has-push-uniform";
        self.assert_ready(OP)?;
        let raw = self.program(OP, program)?.raw;
        // SAFETY: the provider owns the current context and the program is live.
        let found = unsafe { self.gl.get_uniform_location(raw, name).is_some() };
        // SAFETY: the provider owns the current context.
        let error = unsafe { self.gl.get_error() };
        match error {
            glow::NO_ERROR => Ok(found),
            glow::CONTEXT_LOST => Err(GlError::ContextLost { operation: OP }),
            error => Err(GlError::Driver {
                operation: OP,
                message: format!("GL error 0x{error:04x}"),
            }),
        }
    }

    pub(crate) fn set_push_uniforms(
        &mut self,
        program: ProgramId,
        values: &[(String, GlPushUniformValue)],
    ) -> Result<(), GlError> {
        const OP: &str = "set-push-uniforms";
        self.assert_ready(OP)?;
        let raw = self.program(OP, program)?.raw;
        for (_, value) in values {
            let valid = match value {
                GlPushUniformValue::Float(v) => (1..=4).contains(&v.len()),
                GlPushUniformValue::Sint(v) => (1..=4).contains(&v.len()),
                GlPushUniformValue::Uint(v) => (1..=4).contains(&v.len()),
                GlPushUniformValue::Matrix {
                    columns,
                    rows,
                    values,
                } => {
                    (2..=4).contains(columns)
                        && (2..=4).contains(rows)
                        && values.len() == usize::from(*columns) * usize::from(*rows)
                }
            };
            if !valid {
                return Err(Self::validation(OP, "invalid push uniform shape"));
            }
        }
        self.ensure_program(OP, program)?;
        // SAFETY: provider owns the current context; the live program and the
        // lengths of all input slices were checked before issuing any calls.
        unsafe {
            for (name, value) in values {
                let location = self.gl.get_uniform_location(raw, name);
                let Some(location) = location.as_ref() else {
                    continue;
                };
                match value {
                    GlPushUniformValue::Float(v) => match v.len() {
                        1 => self.gl.uniform_1_f32_slice(Some(location), v),
                        2 => self.gl.uniform_2_f32_slice(Some(location), v),
                        3 => self.gl.uniform_3_f32_slice(Some(location), v),
                        4 => self.gl.uniform_4_f32_slice(Some(location), v),
                        _ => unreachable!(),
                    },
                    GlPushUniformValue::Sint(v) => match v.len() {
                        1 => self.gl.uniform_1_i32_slice(Some(location), v),
                        2 => self.gl.uniform_2_i32_slice(Some(location), v),
                        3 => self.gl.uniform_3_i32_slice(Some(location), v),
                        4 => self.gl.uniform_4_i32_slice(Some(location), v),
                        _ => unreachable!(),
                    },
                    GlPushUniformValue::Uint(v) => match v.len() {
                        1 => self.gl.uniform_1_u32_slice(Some(location), v),
                        2 => self.gl.uniform_2_u32_slice(Some(location), v),
                        3 => self.gl.uniform_3_u32_slice(Some(location), v),
                        4 => self.gl.uniform_4_u32_slice(Some(location), v),
                        _ => unreachable!(),
                    },
                    GlPushUniformValue::Matrix {
                        columns,
                        rows,
                        values: v,
                    } => match (*columns, *rows) {
                        (2, 2) => self.gl.uniform_matrix_2_f32_slice(Some(location), false, v),
                        (2, 3) => self
                            .gl
                            .uniform_matrix_2x3_f32_slice(Some(location), false, v),
                        (2, 4) => self
                            .gl
                            .uniform_matrix_2x4_f32_slice(Some(location), false, v),
                        (3, 2) => self
                            .gl
                            .uniform_matrix_3x2_f32_slice(Some(location), false, v),
                        (3, 3) => self.gl.uniform_matrix_3_f32_slice(Some(location), false, v),
                        (3, 4) => self
                            .gl
                            .uniform_matrix_3x4_f32_slice(Some(location), false, v),
                        (4, 2) => self
                            .gl
                            .uniform_matrix_4x2_f32_slice(Some(location), false, v),
                        (4, 3) => self
                            .gl
                            .uniform_matrix_4x3_f32_slice(Some(location), false, v),
                        (4, 4) => self.gl.uniform_matrix_4_f32_slice(Some(location), false, v),
                        _ => unreachable!(),
                    },
                }
            }
        }
        // Distinguish a lost context from ordinary driver errors even when a
        // location lookup returned None during loss.
        // SAFETY: the provider owns the current context.
        let error = unsafe { self.gl.get_error() };
        match error {
            glow::NO_ERROR => Ok(()),
            glow::CONTEXT_LOST => Err(GlError::ContextLost { operation: OP }),
            error => Err(GlError::Driver {
                operation: OP,
                message: format!("GL error 0x{error:04x}"),
            }),
        }
    }
}
