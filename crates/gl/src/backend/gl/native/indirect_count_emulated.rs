//! GLES GPU-count indirect draws through a bounded native compute utility.
//!
//! This executes at the raster command position. It snapshots arguments and
//! the count on the GPU, masks excess commands, then immediately issues native
//! indirect draws. It owns no command stream or render-graph state.
use super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    GlError, GlFamilyApi as _, GlIndexFormat, GlIndirectAbi, GlIndirectCommandRange,
    GlIndirectCountRange,
};
use glow::HasContext as _;

#[cfg(test)]
thread_local! { static TEST_CHUNK_CAPACITY: std::cell::Cell<u32> = const { std::cell::Cell::new(u32::MAX) }; }

/// Restricts a test on the current context thread to exercise native chunking.
#[cfg(test)]
pub(crate) fn set_test_chunk_capacity(capacity: u32) {
    TEST_CHUNK_CAPACITY.with(|value| value.set(capacity.max(1)));
}

impl NativeGlProvider {
    pub(super) fn emulate_indirect_count(
        &mut self,
        command: GlIndirectCommandRange,
        count: GlIndirectCountRange,
    ) -> Result<(), GlError> {
        const OP: &str = "GLES multi-draw-indirect-count";
        command.validate(OP)?;
        count.validate(OP)?;
        self.validate_indirect_vertex_state(OP)?;
        if command.draw_count != count.max_draw_count {
            return Err(Self::validation(OP, "command and count bounds differ"));
        }
        if count.max_draw_count == 0 {
            return Ok(());
        }
        let args = self.indirect_command_buffer(OP, command.range)?;
        let counts = self.indirect_command_buffer(OP, count.range)?;
        let (program, topology) = {
            let raster = self
                .raster
                .as_ref()
                .ok_or_else(|| Self::validation(OP, "no raster pipeline is installed"))?;
            (raster.program, raster.topology)
        };
        self.validate_indirect_driver(OP, program)?;
        let bound = self
            .bound_vertex_array
            .ok_or_else(|| Self::validation(OP, "no vertex array is bound"))?;
        let index = if command.abi == GlIndirectAbi::Indexed {
            let index = self.vertex_array(OP, bound)?.index.ok_or_else(|| {
                Self::validation(OP, "indexed indirect draw requires a bound index buffer")
            })?;
            self.buffer(OP, index.buffer)?;
            Some(index)
        } else {
            None
        };
        let delta = if let Some(index) = index {
            let size = match index.format {
                GlIndexFormat::Uint16 => 2,
                GlIndexFormat::Uint32 => 4,
            };
            if index.offset % size != 0 {
                return Err(Self::validation(OP, "index offset is not element aligned"));
            }
            u32::try_from(index.offset / size)
                .map_err(|_| Self::validation(OP, "index offset exceeds indirect ABI"))?
        } else {
            0
        };
        let words = if index.is_some() { 5u32 } else { 4u32 };
        let record = u64::from(words) * 4;
        let stride = if command.stride == 0 {
            record
        } else {
            u64::from(command.stride)
        };
        let src_offset = command
            .range
            .offset
            .checked_add(command.command_offset)
            .ok_or_else(|| Self::validation(OP, "argument offset overflow"))?;
        let count_offset = count
            .range
            .offset
            .checked_add(count.count_offset)
            .ok_or_else(|| Self::validation(OP, "count offset overflow"))?;
        let end = src_offset
            .checked_add(u64::from(count.max_draw_count - 1) * stride)
            .and_then(|v| v.checked_add(record))
            .ok_or_else(|| Self::validation(OP, "argument offset overflow"))?;
        if end > i32::MAX as u64
            || count_offset
                .checked_add(4)
                .is_none_or(|v| v > i32::MAX as u64)
        {
            return Err(Self::validation(OP, "buffer copy offset exceeds GLint"));
        }
        let limits = self.discovery.limits();
        let capacity = ((limits
            .max_storage_block_size
            .min(i32::MAX as u64)
            .saturating_sub(4))
            / record)
            .min(u64::from(limits.max_compute_work_group_count[0]) * 64)
            .min(u64::from(u32::MAX / words));
        if capacity == 0 {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "indirect count exceeds compute resource limits",
            });
        }
        #[cfg(test)]
        let capacity = capacity.min(u64::from(TEST_CHUNK_CAPACITY.with(|value| value.get())));
        let chunk_capacity = u32::try_from(capacity.min(u64::from(count.max_draw_count)))
            .map_err(|_| Self::validation(OP, "chunk size"))?;
        let buffer_bytes = i32::try_from(u64::from(chunk_capacity) * record + 4)
            .map_err(|_| Self::validation(OP, "scratch size"))?;
        // GL's buffer allocation/copy interface uses signed offsets. Snapshot
        // all records before any draw so later chunks cannot observe changes.
        let snapshot_data = u64::from(count.max_draw_count)
            .checked_mul(record)
            .ok_or_else(|| Self::validation(OP, "snapshot size overflow"))?;
        let snapshot_size = snapshot_data
            .checked_add(4)
            .and_then(|v| i32::try_from(v).ok())
            .ok_or(GlError::Unsupported {
                operation: OP,
                reason: "indirect snapshot exceeds GL buffer limit",
            })?;
        self.set_push_uniforms(
            program,
            &[(
                naga::back::glsl::FIRST_INSTANCE_BINDING.to_owned(),
                super::push_uniform::GlPushUniformValue::Uint(vec![0]),
            )],
        )?;
        self.ensure_program(OP, program)?;
        let source = format!(
            "#version 310 es\nprecision highp int;\nlayout(local_size_x=64) in; layout(std430,binding=0) readonly buffer S {{ uint src[]; }}; layout(std430,binding=1) writeonly buffer D {{ uint dst[]; }}; uniform uint u_base; uniform uint u_draws; uniform uint u_count_pos; void main() {{ uint i=gl_GlobalInvocationID.x; if(i>=u_draws)return; uint p=i*{words}u; bool enabled=u_base+i<src[u_count_pos]; for(uint j=0u;j<{words}u;j++) dst[p+j]=enabled ? src[p+j] : 0u; {} }}",
            if index.is_some() {
                format!("if(enabled) dst[p+2u]=src[p+2u]+{delta}u;")
            } else {
                String::new()
            }
        );
        // SAFETY: all offsets and sizes are validated against live GL allocations;
        // transient objects and every affected binding/program are restored below.
        unsafe {
            let shader = self
                .gl
                .create_shader(glow::COMPUTE_SHADER)
                .map_err(|_| GlError::OutOfMemory { operation: OP })?;
            self.gl.shader_source(shader, &source);
            self.gl.compile_shader(shader);
            if !self.gl.get_shader_compile_status(shader) {
                let message = self.gl.get_shader_info_log(shader);
                self.gl.delete_shader(shader);
                return Err(GlError::Driver {
                    operation: OP,
                    message,
                });
            }
            let utility = match self.gl.create_program() {
                Ok(v) => v,
                Err(_) => {
                    self.gl.delete_shader(shader);
                    return Err(GlError::OutOfMemory { operation: OP });
                }
            };
            self.gl.attach_shader(utility, shader);
            self.gl.link_program(utility);
            self.gl.delete_shader(shader);
            if !self.gl.get_program_link_status(utility) {
                let message = self.gl.get_program_info_log(utility);
                self.gl.delete_program(utility);
                return Err(GlError::Driver {
                    operation: OP,
                    message,
                });
            }
            let input = match self.gl.create_buffer() {
                Ok(v) => v,
                Err(_) => {
                    self.gl.delete_program(utility);
                    return Err(GlError::OutOfMemory { operation: OP });
                }
            };
            let output = match self.gl.create_buffer() {
                Ok(v) => v,
                Err(_) => {
                    self.gl.delete_buffer(input);
                    self.gl.delete_program(utility);
                    return Err(GlError::OutOfMemory { operation: OP });
                }
            };
            let snapshot = match self.gl.create_buffer() {
                Ok(v) => v,
                Err(_) => {
                    self.gl.delete_buffer(input);
                    self.gl.delete_buffer(output);
                    self.gl.delete_program(utility);
                    return Err(GlError::OutOfMemory { operation: OP });
                }
            };
            let handle = |raw: i32| core::num::NonZeroU32::new(raw as u32).map(glow::NativeBuffer);
            let generic = handle(
                self.gl
                    .get_parameter_i32(glow::SHADER_STORAGE_BUFFER_BINDING),
            );
            let read = handle(self.gl.get_parameter_i32(glow::COPY_READ_BUFFER_BINDING));
            let write = handle(self.gl.get_parameter_i32(glow::COPY_WRITE_BUFFER_BINDING));
            let indirect = handle(
                self.gl
                    .get_parameter_i32(glow::DRAW_INDIRECT_BUFFER_BINDING),
            );
            let old_program =
                core::num::NonZeroU32::new(self.gl.get_parameter_i32(glow::CURRENT_PROGRAM) as u32)
                    .map(glow::NativeProgram);
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
            if old.iter().any(|(b, o, n)| {
                b.is_some() && (*o < 0 || *n < 0 || *o > i32::MAX as i64 || *n > i32::MAX as i64)
            }) {
                self.gl.delete_buffer(input);
                self.gl.delete_buffer(output);
                self.gl.delete_buffer(snapshot);
                self.gl.delete_program(utility);
                return Err(GlError::Unsupported {
                    operation: OP,
                    reason: "existing SSBO binding cannot be restored",
                });
            }
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(input));
            self.gl
                .buffer_data_size(glow::COPY_WRITE_BUFFER, buffer_bytes, glow::STREAM_DRAW);
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(output));
            self.gl
                .buffer_data_size(glow::COPY_WRITE_BUFFER, buffer_bytes - 4, glow::STREAM_DRAW);
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(snapshot));
            self.gl
                .buffer_data_size(glow::COPY_WRITE_BUFFER, snapshot_size, glow::STREAM_DRAW);
            let mut result = self.driver_error(OP);
            if result.is_ok() {
                self.gl.memory_barrier(glow::BUFFER_UPDATE_BARRIER_BIT);
                self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(counts));
                self.gl.copy_buffer_sub_data(
                    glow::COPY_READ_BUFFER,
                    glow::COPY_WRITE_BUFFER,
                    count_offset as i32,
                    snapshot_data as i32,
                    4,
                );
                self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(args));
                if stride == record {
                    self.gl.copy_buffer_sub_data(
                        glow::COPY_READ_BUFFER,
                        glow::COPY_WRITE_BUFFER,
                        src_offset as i32,
                        0,
                        snapshot_data as i32,
                    );
                } else {
                    for i in 0..count.max_draw_count {
                        self.gl.copy_buffer_sub_data(
                            glow::COPY_READ_BUFFER,
                            glow::COPY_WRITE_BUFFER,
                            (src_offset + u64::from(i) * stride) as i32,
                            (u64::from(i) * record) as i32,
                            record as i32,
                        );
                    }
                }
                result = self.driver_error(OP);
            }
            let base_uniform = self.gl.get_uniform_location(utility, "u_base");
            let draws_uniform = self.gl.get_uniform_location(utility, "u_draws");
            let count_uniform = self.gl.get_uniform_location(utility, "u_count_pos");
            let mut base = 0u32;
            while base < count.max_draw_count && result.is_ok() {
                let draws = (count.max_draw_count - base).min(chunk_capacity);
                let count_pos = draws * words;
                self.gl.memory_barrier(
                    glow::BUFFER_UPDATE_BARRIER_BIT | glow::SHADER_STORAGE_BARRIER_BIT,
                );
                self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(input));
                self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(snapshot));
                self.gl.copy_buffer_sub_data(
                    glow::COPY_READ_BUFFER,
                    glow::COPY_WRITE_BUFFER,
                    snapshot_data as i32,
                    (count_pos * 4) as i32,
                    4,
                );
                self.gl.copy_buffer_sub_data(
                    glow::COPY_READ_BUFFER,
                    glow::COPY_WRITE_BUFFER,
                    (u64::from(base) * record) as i32,
                    0,
                    (u64::from(draws) * record) as i32,
                );
                self.gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT);
                self.gl
                    .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(input));
                self.gl
                    .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 1, Some(output));
                self.gl.use_program(Some(utility));
                self.gl.uniform_1_u32(base_uniform.as_ref(), base);
                self.gl.uniform_1_u32(draws_uniform.as_ref(), draws);
                self.gl.uniform_1_u32(count_uniform.as_ref(), count_pos);
                self.gl.dispatch_compute(draws.div_ceil(64), 1, 1);
                self.gl.memory_barrier(glow::COMMAND_BARRIER_BIT);
                if let Err(error) = self.driver_error(OP) {
                    result = Err(error);
                    break;
                }
                // Draw shaders must see their original SSBO bindings, not the
                // private compute utility's input and output buffers.
                for (i, (buffer, offset, size)) in old.iter().copied().enumerate() {
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
                self.gl.use_program(old_program);
                self.gl
                    .bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(output));
                let mode = super::exec_raster::topology_mode(topology);
                for i in 0..draws {
                    let offset = (u64::from(i) * record) as i32;
                    if let Some(index) = index {
                        self.gl.draw_elements_indirect_offset(
                            mode,
                            super::exec_vertex::indexed_draw_type(index),
                            offset,
                        );
                    } else {
                        self.gl.draw_arrays_indirect_offset(mode, offset);
                    }
                }
                result = self.driver_error(OP);
                base += draws;
            }
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
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, read);
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, write);
            self.gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, indirect);
            self.gl.use_program(old_program);
            self.gl.delete_buffer(input);
            self.gl.delete_buffer(output);
            self.gl.delete_buffer(snapshot);
            self.gl.delete_program(utility);
            result
        }
    }
}
