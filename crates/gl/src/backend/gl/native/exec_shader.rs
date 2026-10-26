//! Native shader, program, reflection, and binding execution.
//!
//! Sources arrive already lowered and profile-validated: ESSL 300/310/320 for
//! the embedded family, the exact `4xx` desktop GLSL for the desktop family.
//! This module compiles them, keeps every driver log inside a structured
//! error, reflects the actually linked resources, and rejects any layout or
//! program mismatch before a program object is published.

use super::provider::{NativeGlProvider, NativeProgram};
use crate::backend::gl::api::{
    BufferId, GlBindingApi, GlBindingLimits, GlBufferUsage, GlError, GlExecutableBindingAssignment,
    GlExecutableBindingLocation, GlFamilyApi as _, GlProgramDescriptor, GlProgramKind,
    GlProgramReflection, GlShaderApi, GlShaderResourceKind, GlShaderSource, GlShaderStage,
    GlTextureTarget, ProgramId, SamplerId, ShaderId, TextureId, validate_texture_unit,
    validate_uniform_buffer_binding, validate_uniform_range,
};
use glow::HasContext as _;

/// The sampler uniform types this backend can bind and reflect. The closed
/// list matches the shared contract; integer and cube-shadow sampler types
/// stay outside the common binding vocabulary and are skipped like plain
/// uniforms, exactly as the browser backend records them.
const SAMPLER_TYPES: &[u32] = &[
    0x8B5E, // SAMPLER_2D
    0x8B5F, // SAMPLER_3D
    0x8B60, // SAMPLER_CUBE
    0x8B62, // SAMPLER_2D_SHADOW
    0x8DC1, // SAMPLER_2D_ARRAY (GLSL 1.30 / ES 3.00 enum)
    0x8DC5, // SAMPLER_CUBE_SHADOW
];

/// Column count of an active vertex-input type, if it maps to the common
/// vertex-format vocabulary. Matrices expand to several locations and are
/// rejected instead of being silently collapsed.
fn vertex_input_columns(type_: u32) -> Option<u8> {
    Some(match type_ {
        0x1404..=0x1406 => 1,          // INT, UNSIGNED_INT, FLOAT
        0x8B50 | 0x8B53 | 0x8DC6 => 2, // FLOAT_VEC2, INT_VEC2, UNSIGNED_INT_VEC2
        0x8B51 | 0x8B54 | 0x8DC7 => 3, // FLOAT_VEC3, INT_VEC3, UNSIGNED_INT_VEC3
        0x8B52 | 0x8B55 | 0x8DC8 => 4, // FLOAT_VEC4, INT_VEC4, UNSIGNED_INT_VEC4
        _ => return None,
    })
}

fn is_sampler_type(type_: u32) -> bool {
    SAMPLER_TYPES.contains(&type_)
}

/// Strips the `[0]` suffix GL reports for array uniforms.
fn base_uniform_name(name: &str) -> &str {
    name.strip_suffix("[0]").unwrap_or(name)
}

impl GlShaderApi for NativeGlProvider {
    fn create_shader(&mut self, source: &GlShaderSource) -> Result<ShaderId, GlError> {
        const OP: &str = "create-shader";
        self.assert_ready(OP)?;
        source
            .validate_for(self.profile())
            .map_err(|_| Self::validation(OP, "shader source does not target this profile"))?;
        if source.stage == GlShaderStage::Compute
            && !self
                .discovery
                .capabilities()
                .supports(crate::backend::gl::api::GlCapability::Compute)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the compute capability",
            });
        }
        let raw = self.compile_shader_source(OP, source)?;
        let slot = self.slot(OP)?;
        let id = ShaderId::new(self.context_stamp(), slot, 0);
        self.shaders.insert(id, raw);
        Ok(id)
    }

    fn destroy_shader(&mut self, shader: ShaderId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "destroy-shader";
        self.assert_ready(OP)?;
        let raw = self.shader(OP, shader)?;
        // SAFETY: current-context contract; liveness was checked first.
        unsafe { self.gl.delete_shader(raw) };
        self.driver_error(OP)?;
        self.shaders.remove(&shader);
        Ok(())
    }

    fn create_program(
        &mut self,
        descriptor: &GlProgramDescriptor,
    ) -> Result<(ProgramId, GlProgramReflection), GlError> {
        const OP: &str = "create-program";
        self.assert_ready(OP)?;
        descriptor
            .validate_for(self.profile())
            .map_err(|_| Self::validation(OP, "invalid program descriptor"))?;
        // Compute programs route on the proved capability; without the probe
        // evidence the kind is rejected structurally before any driver call.
        if matches!(descriptor.kind, GlProgramKind::Compute { .. })
            && !self
                .discovery
                .capabilities()
                .supports(crate::backend::gl::api::GlCapability::Compute)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the compute capability",
            });
        }
        let stages: Vec<&GlShaderSource> = match &descriptor.kind {
            GlProgramKind::Raster { vertex, fragment } => vec![vertex, fragment],
            GlProgramKind::Compute { shader } => vec![shader],
        };
        let mut compiled = Vec::new();
        for source in stages {
            let mut source = source.clone();
            if matches!(
                self.profile(),
                crate::backend::gl::api::GlFamilyProfile::Embedded { .. }
            ) {
                for (point, binding) in descriptor.layout.bindings.iter().enumerate() {
                    if !matches!(binding.kind, GlShaderResourceKind::StorageBuffer(_)) {
                        continue;
                    }
                    for name in &binding.names {
                        let needle = format!("buffer {name} ");
                        if let Some(pos) = source.text.find(&needle) {
                            let line_start = source.text[..pos].rfind('\n').map_or(0, |i| i + 1);
                            let prefix = &source.text[line_start..pos];
                            let Some(layout) = prefix.find("layout(") else {
                                return Err(Self::validation(
                                    OP,
                                    "storage block has no layout qualifier",
                                ));
                            };
                            let Some(close) = prefix[layout..].find(')') else {
                                return Err(Self::validation(
                                    OP,
                                    "storage block has invalid layout qualifier",
                                ));
                            };
                            if prefix[layout..layout + close].contains("binding") {
                                // Direct ESSL may already choose its native binding.
                                // Linked reflection validates that point and the
                                // immediate binder uses the reflected assignment.
                                continue;
                            }
                            source.text.insert_str(
                                line_start + layout + close,
                                &format!(", binding = {point}"),
                            );
                        }
                    }
                }
            }
            match self.compile_shader_source(OP, &source) {
                Ok(raw) => compiled.push(raw),
                Err(error) => {
                    self.delete_shaders(&compiled);
                    return Err(error);
                }
            }
        }
        let linked = self.link_and_reflect(OP, descriptor, &compiled);
        // Shaders may be deleted once attached; the program owns its binaries.
        self.delete_shaders(&compiled);
        let (raw, reflection) = linked?;
        // Resolve the generated instance-base uniform while linking is already
        // a cold operation.  This name is owned by the Naga GL lowering, not
        // by application push-constant reflection.
        // SAFETY: `raw` is the successfully linked program for this current
        // context and get_uniform_location does not require it to be current.
        let first_instance_uniform = unsafe {
            self.gl
                .get_uniform_location(raw, naga::back::glsl::FIRST_INSTANCE_BINDING)
        };
        if let Err(error) = self.driver_error(OP) {
            // SAFETY: `raw` belongs to this current context and has not been
            // placed in the executable table on this failure path.
            unsafe { self.gl.delete_program(raw) };
            return Err(error);
        }
        let slot = self.slot(OP)?;
        let id = ProgramId::new(self.context_stamp(), slot, 0);
        self.programs.insert(
            id,
            NativeProgram {
                generation: id.generation,
                raw,
                descriptor: descriptor.clone(),
                reflection: reflection.clone(),
                first_instance_uniform,
                first_instance_value: None,
            },
        );
        Ok((id, reflection))
    }

    fn destroy_program(&mut self, program: ProgramId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "destroy-program";
        self.assert_ready(OP)?;
        self.program(OP, program)?;
        let entry = self
            .programs
            .remove(&program)
            .ok_or_else(|| Self::validation(OP, "program disappeared"))?;
        // SAFETY: current-context contract; liveness was checked first.
        unsafe { self.gl.delete_program(entry.raw) };
        self.driver_error(OP)
    }
}
