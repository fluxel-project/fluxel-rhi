//! Immediate uploads of reflected push constant leaves to a linked program.
//!
//! Matrices arrive tightly packed in column-major order. Optimized-away leaves
//! have no location and need no upload. Locations (including optimized-away
//! `None`) are cached by linked-program generation; uploads remain immediate
//! command emission and defer driver diagnostics to publication.

use super::provider::NativeGlProvider;
use crate::backend::api::{GlError, GlFamilyApi as _, ProgramId};
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
    fn cached_uniform_location(
        &mut self,
        operation: &'static str,
        program: ProgramId,
        name: &str,
    ) -> Result<Option<glow::NativeUniformLocation>, GlError> {
        if let Some(location) = self
            .program(operation, program)?
            .uniform_locations
            .get(name)
            .cloned()
        {
            return Ok(location);
        }
        let raw = self.program(operation, program)?.raw;
        // SAFETY: program is live and locations remain valid through its link
        // generation, which owns this cache entry.
        let location = unsafe { self.gl.get_uniform_location(raw, name) };
        self.programs
            .get_mut(&program)
            .expect("live program resolved before uniform location cache")
            .uniform_locations
            .insert(name.to_owned(), location.clone());
        self.defer_driver_error(operation)?;
        Ok(location)
    }
    /// Updates Naga's internal first-instance uniform when its value changes.
    ///
    /// Unlike application push uniforms, this has one fixed scalar type and a
    /// link-time-known name.  Keeping its location and last submitted value in
    /// the program record removes allocation, lookup, upload, and error-query
    /// work from the common `first_instance == 0` draw path.  Changed values
    /// still defer diagnostics to the frame boundary, avoiding a `glGetError`
    /// in the draw path.
    pub(crate) fn set_first_instance_uniform(
        &mut self,
        program: ProgramId,
        first_instance: u32,
    ) -> Result<(), GlError> {
        const OP: &str = "set-first-instance-uniform";
        self.assert_ready(OP)?;
        let (location, previous) = {
            let record = self.program(OP, program)?;
            (record.first_instance_uniform, record.first_instance_value)
        };
        let Some(location) = location else {
            return Ok(());
        };
        if previous == Some(first_instance) {
            return Ok(());
        }
        self.ensure_program(OP, program)?;
        // SAFETY: `location` was resolved from this live program at link time,
        // that program is current, and this generated uniform is a scalar uint.
        unsafe { self.gl.uniform_1_u32(Some(&location), first_instance) };
        self.defer_driver_error(OP)?;
        self.programs
            .get_mut(&program)
            .expect("live program resolved before uniform upload")
            .first_instance_value = Some(first_instance);
        Ok(())
    }

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
        let location = self.cached_uniform_location(OP, program, name)?;
        let Some(location) = location else {
            return Ok(());
        };
        self.ensure_program(OP, program)?;
        // SAFETY: the program is selected and this uniform is generated as uint.
        unsafe {
            self.gl.uniform_1_u32(Some(&location), count);
        }
        self.defer_driver_error(OP)
    }

    /// Reports whether the linked program retained a reflected uniform leaf.
    pub(crate) fn has_uniform(&mut self, program: ProgramId, name: &str) -> Result<bool, GlError> {
        const OP: &str = "has-push-uniform";
        self.assert_ready(OP)?;
        Ok(self.cached_uniform_location(OP, program, name)?.is_some())
    }

    pub(crate) fn set_push_uniforms(
        &mut self,
        program: ProgramId,
        values: &[(String, GlPushUniformValue)],
    ) -> Result<(), GlError> {
        const OP: &str = "set-push-uniforms";
        self.assert_ready(OP)?;
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
                let location = self.cached_uniform_location(OP, program, name)?;
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
        self.defer_driver_error(OP)
    }
}
