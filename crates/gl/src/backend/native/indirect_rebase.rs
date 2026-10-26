//! GPU-only rewriting of indexed indirect records for an index binding offset.
//!
//! One fixed compute utility and its uniform locations are cached per native
//! context. Command-specific values are uniforms, so ordinary indirect draws
//! never compile/link or reflect the helper program; normal teardown deletes
//! it while context loss only discards the now-invalid handle.
use super::provider::NativeGlProvider;
use crate::backend::api::{
    GlCapability, GlError, GlFamilyApi as _, GlFamilyProfile, GlIndexFormat, GlIndirectCommandRange,
};
use glow::HasContext as _;

pub(super) struct IndirectRebaseUtility {
    pub(super) program: glow::NativeProgram,
    draw_count: glow::NativeUniformLocation,
    base: glow::NativeUniformLocation,
    stride: glow::NativeUniformLocation,
    delta: glow::NativeUniformLocation,
}

impl NativeGlProvider {
    /// # Safety
    /// The provider's exact context is current during normal device teardown.
    pub(crate) unsafe fn clear_indirect_rebase_utility(&mut self) {
        if let Some(utility) = self.indirect_rebase_utility.take() {
            // SAFETY: caller holds the owning context current.
            unsafe { self.gl.delete_program(utility.program) };
        }
    }

    pub(super) fn rebase_indirect_indices(
        &mut self,
        source: glow::NativeBuffer,
        command: GlIndirectCommandRange,
        index: crate::backend::api::GlIndexBinding,
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
        let utility = self.ensure_indirect_rebase_utility(OP, version)?;
        let (program, draw_count, base_uniform, stride_uniform, delta_uniform) = (
            utility.program,
            utility.draw_count.clone(),
            utility.base.clone(),
            utility.stride.clone(),
            utility.delta.clone(),
        );
        // SAFETY: current context is exclusively held; temporary objects and all bindings are restored on every path.
        unsafe {
            let scratch = match self.gl.create_buffer() {
                Ok(v) => v,
                Err(_) => return Err(GlError::OutOfMemory { operation: OP }),
            };
            // The provider owns this context and records every public SSBO and
            // program bind. Snapshot those CPU facts; querying GL here would
            // synchronize the stream immediately before the utility dispatch.
            let old_generic = self.generic_storage_buffer;
            let old = [0, 1].map(|i| self.storage_bindings.get(&i).copied());
            let old_program = self.current_native_program;
            self.gl
                .bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(scratch));
            self.gl
                .buffer_data_size(glow::SHADER_STORAGE_BUFFER, words, glow::STREAM_DRAW);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(source));
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 1, Some(scratch));
            self.gl.use_program(Some(program));
            self.gl.uniform_1_u32(Some(&draw_count), command.draw_count);
            self.gl.uniform_1_u32(Some(&base_uniform), base);
            self.gl.uniform_1_u32(Some(&stride_uniform), stride);
            self.gl.uniform_1_u32(Some(&delta_uniform), delta);
            self.gl.dispatch_compute(groups, 1, 1);
            self.gl
                .memory_barrier(glow::COMMAND_BARRIER_BIT | glow::SHADER_STORAGE_BARRIER_BIT);
            for (i, binding) in old.into_iter().enumerate() {
                if let Some(binding) =
                    binding.filter(|binding| binding.buffer.is_some() && binding.size > 0)
                {
                    self.bind_buffer_range_wide(
                        glow::SHADER_STORAGE_BUFFER,
                        i as u32,
                        binding.buffer,
                        binding.offset,
                        binding.size,
                        OP,
                    )?;
                } else {
                    self.gl.bind_buffer_base(
                        glow::SHADER_STORAGE_BUFFER,
                        i as u32,
                        binding.and_then(|binding| binding.buffer),
                    );
                }
            }
            self.gl
                .bind_buffer(glow::SHADER_STORAGE_BUFFER, old_generic);
            self.gl.use_program(old_program);
            self.defer_driver_error(OP)?;
            Ok(scratch)
        }
    }

    fn ensure_indirect_rebase_utility(
        &mut self,
        operation: &'static str,
        version: &str,
    ) -> Result<&IndirectRebaseUtility, GlError> {
        if self.indirect_rebase_utility.is_none() {
            let source = format!(
                "{version}layout(local_size_x=64) in; layout(std430,binding=0) readonly buffer S {{ uint src[]; }}; layout(std430,binding=1) writeonly buffer D {{ uint dst[]; }}; uniform uint u_draw_count,u_base,u_stride,u_delta; void main() {{ uint i=gl_GlobalInvocationID.x; if(i >= u_draw_count) return; uint s=u_base+i*u_stride; uint d=i*5u; dst[d]=src[s]; dst[d+1u]=src[s+1u]; dst[d+2u]=src[s+2u]+u_delta; dst[d+3u]=src[s+3u]; dst[d+4u]=src[s+4u]; }}"
            );
            // SAFETY: this provider owns the current context and retains the
            // linked utility for this context generation only.
            unsafe {
                let shader = self
                    .gl
                    .create_shader(glow::COMPUTE_SHADER)
                    .map_err(|_| GlError::OutOfMemory { operation })?;
                self.gl.shader_source(shader, &source);
                self.gl.compile_shader(shader);
                if !self.gl.get_shader_compile_status(shader) {
                    let message = self.gl.get_shader_info_log(shader);
                    self.gl.delete_shader(shader);
                    return Err(GlError::Driver { operation, message });
                }
                let program = self
                    .gl
                    .create_program()
                    .map_err(|_| GlError::OutOfMemory { operation })?;
                self.gl.attach_shader(program, shader);
                self.gl.link_program(program);
                self.gl.delete_shader(shader);
                if !self.gl.get_program_link_status(program) {
                    let message = self.gl.get_program_info_log(program);
                    self.gl.delete_program(program);
                    return Err(GlError::Driver { operation, message });
                }
                let loc = |name| {
                    self.gl
                        .get_uniform_location(program, name)
                        .ok_or_else(|| GlError::Driver {
                            operation,
                            message: "indirect rebase utility uniform optimized out".into(),
                        })
                };
                let utility = IndirectRebaseUtility {
                    program,
                    draw_count: loc("u_draw_count")?,
                    base: loc("u_base")?,
                    stride: loc("u_stride")?,
                    delta: loc("u_delta")?,
                };
                self.indirect_rebase_utility = Some(utility);
            }
        }
        Ok(self
            .indirect_rebase_utility
            .as_ref()
            .expect("utility inserted"))
    }
}
