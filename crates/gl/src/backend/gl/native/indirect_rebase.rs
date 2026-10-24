//! GPU-only rewriting of indexed indirect records for an index binding offset.
use super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    GlCapability, GlError, GlFamilyApi as _, GlFamilyProfile, GlIndexFormat, GlIndirectCommandRange,
};
use glow::HasContext as _;

impl NativeGlProvider {
    pub(super) fn rebase_indirect_indices(
        &mut self,
        source: glow::NativeBuffer,
        command: GlIndirectCommandRange,
        index: crate::backend::gl::api::GlIndexBinding,
    ) -> Result<glow::NativeBuffer, GlError> {
        const OP: &str = "rebase-indexed-indirect";
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
                operation: OP,
                reason: "nonzero index offset requires GPU compute",
            });
        }
        let element_size = match index.format {
            GlIndexFormat::Uint16 => 2,
            GlIndexFormat::Uint32 => 4,
        };
        if index.offset % element_size != 0 {
            return Err(Self::validation(OP, "index offset is not element aligned"));
        }
        let delta = u32::try_from(index.offset / element_size)
            .map_err(|_| Self::validation(OP, "index offset exceeds indirect ABI"))?;
        let words = u64::from(command.draw_count)
            .checked_mul(5)
            .and_then(|v| v.checked_mul(4))
            .and_then(|v| i32::try_from(v).ok())
            .ok_or_else(|| Self::validation(OP, "rebased arguments exceed buffer limits"))?;
        if self.buffer(OP, command.range.buffer)?.1.size
            > self.discovery.limits().max_storage_block_size
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "indirect arguments exceed storage block limit",
            });
        }
        let base = command
            .range
            .offset
            .checked_add(command.command_offset)
            .and_then(|v| u32::try_from(v / 4).ok())
            .ok_or_else(|| Self::validation(OP, "argument offset exceeds shader address range"))?;
        let stride = if command.stride == 0 {
            5
        } else {
            command.stride / 4
        };
        let groups = command.draw_count.div_ceil(64);
        if groups > self.discovery.limits().max_compute_work_group_count[0] {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "rebased draw count exceeds compute dispatch limit",
            });
        }
        let version = match self.discovery.context().profile() {
            GlFamilyProfile::Desktop { .. } => "#version 430 core\n",
            GlFamilyProfile::Embedded { .. } => "#version 310 es\nprecision highp int;\n",
            _ => {
                return Err(GlError::Unsupported {
                    operation: OP,
                    reason: "compute profile unavailable",
                });
            }
        };
        let source_code = format!(
            "{version}layout(local_size_x=64) in; layout(std430,binding=0) readonly buffer S {{ uint src[]; }}; layout(std430,binding=1) writeonly buffer D {{ uint dst[]; }}; void main() {{ uint i=gl_GlobalInvocationID.x; if(i >= {}u) return; uint s={}u+i*{}u; uint d=i*5u; dst[d]=src[s]; dst[d+1u]=src[s+1u]; dst[d+2u]=src[s+2u]+{}u; dst[d+3u]=src[s+3u]; dst[d+4u]=src[s+4u]; }}",
            command.draw_count, base, stride, delta
        );
        // SAFETY: current context is exclusively held; temporary objects and all bindings are restored on every path.
        unsafe {
            let shader = self
                .gl
                .create_shader(glow::COMPUTE_SHADER)
                .map_err(|_| GlError::OutOfMemory { operation: OP })?;
            self.gl.shader_source(shader, &source_code);
            self.gl.compile_shader(shader);
            if !self.gl.get_shader_compile_status(shader) {
                let message = self.gl.get_shader_info_log(shader);
                self.gl.delete_shader(shader);
                return Err(GlError::Driver {
                    operation: OP,
                    message,
                });
            }
            let program = match self.gl.create_program() {
                Ok(v) => v,
                Err(_) => {
                    self.gl.delete_shader(shader);
                    return Err(GlError::OutOfMemory { operation: OP });
                }
            };
            self.gl.attach_shader(program, shader);
            self.gl.link_program(program);
            self.gl.delete_shader(shader);
            if !self.gl.get_program_link_status(program) {
                let message = self.gl.get_program_info_log(program);
                self.gl.delete_program(program);
                return Err(GlError::Driver {
                    operation: OP,
                    message,
                });
            }
            let scratch = match self.gl.create_buffer() {
                Ok(v) => v,
                Err(_) => {
                    self.gl.delete_program(program);
                    return Err(GlError::OutOfMemory { operation: OP });
                }
            };
            let handle = |raw: i32| core::num::NonZeroU32::new(raw as u32).map(glow::NativeBuffer);
            let old_generic = handle(
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
            let old_program =
                core::num::NonZeroU32::new(self.gl.get_parameter_i32(glow::CURRENT_PROGRAM) as u32)
                    .map(glow::NativeProgram);
            self.gl
                .bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(scratch));
            self.gl
                .buffer_data_size(glow::SHADER_STORAGE_BUFFER, words, glow::STREAM_DRAW);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(source));
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 1, Some(scratch));
            self.gl.use_program(Some(program));
            self.gl.dispatch_compute(groups, 1, 1);
            self.gl
                .memory_barrier(glow::COMMAND_BARRIER_BIT | glow::SHADER_STORAGE_BARRIER_BIT);
            for (i, (buffer, offset, size)) in old.into_iter().enumerate() {
                if buffer.is_some() && size > 0 {
                    self.bind_buffer_range_wide(
                        glow::SHADER_STORAGE_BUFFER,
                        i as u32,
                        buffer,
                        u64::try_from(offset)
                            .map_err(|_| Self::validation(OP, "saved SSBO offset is negative"))?,
                        u64::try_from(size)
                            .map_err(|_| Self::validation(OP, "saved SSBO size is negative"))?,
                        OP,
                    )?;
                } else {
                    self.gl
                        .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, i as u32, buffer);
                }
            }
            self.gl
                .bind_buffer(glow::SHADER_STORAGE_BUFFER, old_generic);
            self.gl.use_program(old_program);
            self.gl.delete_program(program);
            if let Err(e) = self.driver_error(OP) {
                self.gl.delete_buffer(scratch);
                return Err(e);
            }
            Ok(scratch)
        }
    }
}
