//! GPU-only conversion from tightly packed stencil bytes to GL packed texels.
use super::super::provider::NativeGlProvider;
use crate::backend::gl::api::{GlCapability, GlError, GlFamilyApi as _, GlFamilyProfile};
use glow::HasContext as _;

impl NativeGlProvider {
    pub(super) fn stencil_pack_preflight(
        &self,
        op: &'static str,
        count: usize,
        output_bytes: usize,
    ) -> Result<(), GlError> {
        if !matches!(self.discovery.context().profile(), GlFamilyProfile::Desktop { major, minor } if major > 4 || (major == 4 && minor >= 3))
        {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "stencil PBO packing requires desktop GLSL 4.30",
            });
        }
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::Compute)
            || !self
                .discovery
                .capabilities()
                .supports(GlCapability::StorageBuffer)
        {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "stencil PBO packing requires compute and storage buffers",
            });
        }
        let count = u32::try_from(count)
            .map_err(|_| Self::validation(op, "stencil pixel count exceeds shader address"))?;
        if count.div_ceil(64) > self.discovery.limits().max_compute_work_group_count[0]
            || output_bytes as u64 > self.discovery.limits().max_storage_block_size
        {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "stencil pack exceeds compute limits",
            });
        }
        Ok(())
    }
    /// Source is padded to four bytes and contains one tightly packed byte per texel.
    pub(super) fn pack_stencil_gpu(
        &mut self,
        op: &'static str,
        input: glow::NativeBuffer,
        output: glow::NativeBuffer,
        count: usize,
        float_depth: bool,
    ) -> Result<(), GlError> {
        let count = u32::try_from(count).map_err(|_| Self::validation(op, "pixel count"))?;
        let body = if float_depth {
            "dst[i*2u]=0u; dst[i*2u+1u]=v;"
        } else {
            "dst[i]=v;"
        };
        let code = format!(
            "#version 430 core\nlayout(local_size_x=64) in; layout(std430,binding=0) readonly buffer S {{ uint src[]; }}; layout(std430,binding=1) writeonly buffer D {{ uint dst[]; }}; void main() {{ uint i=gl_GlobalInvocationID.x; if(i>={count}u)return; uint v=(src[i/4u] >> ((i%4u)*8u)) & 255u; {body} }}"
        );
        // SAFETY: temporary shader/program and SSBOs belong to current context;
        // prior indexed and generic bindings and program are restored.
        unsafe {
            let shader = self
                .gl
                .create_shader(glow::COMPUTE_SHADER)
                .map_err(|_| GlError::OutOfMemory { operation: op })?;
            self.gl.shader_source(shader, &code);
            self.gl.compile_shader(shader);
            if !self.gl.get_shader_compile_status(shader) {
                let message = self.gl.get_shader_info_log(shader);
                self.gl.delete_shader(shader);
                return Err(GlError::Driver {
                    operation: op,
                    message,
                });
            }
            let program = match self.gl.create_program() {
                Ok(v) => v,
                Err(_) => {
                    self.gl.delete_shader(shader);
                    return Err(GlError::OutOfMemory { operation: op });
                }
            };
            self.gl.attach_shader(program, shader);
            self.gl.link_program(program);
            self.gl.delete_shader(shader);
            if !self.gl.get_program_link_status(program) {
                let message = self.gl.get_program_info_log(program);
                self.gl.delete_program(program);
                return Err(GlError::Driver {
                    operation: op,
                    message,
                });
            }
            let handle = |raw: i32| std::num::NonZeroU32::new(raw as u32).map(glow::NativeBuffer);
            let generic = handle(
                self.gl
                    .get_parameter_i32(glow::SHADER_STORAGE_BUFFER_BINDING),
            );
            let old = [0, 1].map(|i| {
                (
                    handle(
                        self.gl
                            .get_parameter_indexed_i32(glow::SHADER_STORAGE_BUFFER_BINDING, i),
                    ),
                    self.gl
                        .get_parameter_indexed_i64(glow::SHADER_STORAGE_BUFFER_START, i),
                    self.gl
                        .get_parameter_indexed_i64(glow::SHADER_STORAGE_BUFFER_SIZE, i),
                )
            });
            if old.iter().any(|(buffer, offset, size)| {
                buffer.is_some()
                    && (*offset < 0
                        || *size < 0
                        || *offset > i32::MAX as i64
                        || *size > i32::MAX as i64)
            }) {
                self.gl.delete_program(program);
                return Err(GlError::Unsupported {
                    operation: op,
                    reason: "existing SSBO binding cannot be restored by glow",
                });
            }
            let previous =
                std::num::NonZeroU32::new(self.gl.get_parameter_i32(glow::CURRENT_PROGRAM) as u32)
                    .map(glow::NativeProgram);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(input));
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 1, Some(output));
            self.gl.use_program(Some(program));
            self.gl.dispatch_compute(count.div_ceil(64), 1, 1);
            self.gl
                .memory_barrier(glow::PIXEL_BUFFER_BARRIER_BIT | glow::SHADER_STORAGE_BARRIER_BIT);
            for (i, (buffer, offset, size)) in old.into_iter().enumerate() {
                if buffer.is_some() && size > 0 {
                    self.gl.bind_buffer_range(
                        glow::SHADER_STORAGE_BUFFER,
                        i as u32,
                        buffer,
                        offset as i32,
                        size as i32,
                    );
                } else {
                    self.gl
                        .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, i as u32, buffer);
                }
            }
            self.gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, generic);
            self.gl.use_program(previous);
            self.gl.delete_program(program);
            self.driver_error(op)
        }
    }
}
