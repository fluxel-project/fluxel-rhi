//! GLES GPU-count indirect draws through a bounded native compute utility.
//!
//! This executes at the raster command position. It snapshots arguments and
//! the count on the GPU, masks excess commands, then immediately issues native
//! indirect draws. It owns no command stream or render-graph state.
use super::provider::NativeGlProvider;
use crate::backend::api::{
    GlError, GlFamilyApi as _, GlIndexFormat, GlIndirectAbi, GlIndirectCommandRange,
    GlIndirectCountRange,
};
use glow::HasContext as _;

pub(super) mod scratch;
mod transaction;
use transaction::IndirectCountTransaction;

/// Context-owned fixed compute lowering. The caller supplies every varying
/// record shape through uniforms, so this native program never encodes a draw
/// count, chunk base, index delta, or operation-specific allocation.
pub(crate) struct IndirectCountUtility {
    program: glow::NativeProgram,
    base: Option<glow::NativeUniformLocation>,
    draws: Option<glow::NativeUniformLocation>,
    count_pos: Option<glow::NativeUniformLocation>,
    words: Option<glow::NativeUniformLocation>,
    delta: Option<glow::NativeUniformLocation>,
    indexed: Option<glow::NativeUniformLocation>,
}

impl IndirectCountUtility {
    pub(crate) unsafe fn create(gl: &glow::Context) -> Result<Self, GlError> {
        const OP: &str = "GLES multi-draw-indirect-count utility";
        const SOURCE: &str = "#version 310 es\nprecision highp int;\nlayout(local_size_x=64) in;\nlayout(std430,binding=0) readonly buffer S { uint src[]; };\nlayout(std430,binding=1) writeonly buffer D { uint dst[]; };\nuniform uint u_base; uniform uint u_draws; uniform uint u_count_pos; uniform uint u_words; uniform uint u_delta; uniform uint u_indexed;\nvoid main(){ uint i=gl_GlobalInvocationID.x; if(i>=u_draws)return; uint p=i*u_words; bool enabled=u_base+i<src[u_count_pos]; for(uint j=0u;j<u_words;j++)dst[p+j]=enabled?src[p+j]:0u; if(enabled&&u_indexed!=0u)dst[p+2u]=src[p+2u]+u_delta; }";
        // SAFETY: current-context contract belongs to NativeGlProvider.
        let shader = unsafe { gl.create_shader(glow::COMPUTE_SHADER) }
            .map_err(|_| GlError::OutOfMemory { operation: OP })?;
        unsafe {
            gl.shader_source(shader, SOURCE);
            gl.compile_shader(shader);
        }
        if !unsafe { gl.get_shader_compile_status(shader) } {
            let message = unsafe { gl.get_shader_info_log(shader) };
            unsafe { gl.delete_shader(shader) };
            return Err(GlError::Driver {
                operation: OP,
                message,
            });
        }
        let program = match unsafe { gl.create_program() } {
            Ok(program) => program,
            Err(_) => {
                unsafe { gl.delete_shader(shader) };
                return Err(GlError::OutOfMemory { operation: OP });
            }
        };
        unsafe {
            gl.attach_shader(program, shader);
            gl.link_program(program);
            gl.delete_shader(shader);
        }
        if !unsafe { gl.get_program_link_status(program) } {
            let message = unsafe { gl.get_program_info_log(program) };
            unsafe { gl.delete_program(program) };
            return Err(GlError::Driver {
                operation: OP,
                message,
            });
        }
        Ok(Self {
            base: unsafe { gl.get_uniform_location(program, "u_base") },
            draws: unsafe { gl.get_uniform_location(program, "u_draws") },
            count_pos: unsafe { gl.get_uniform_location(program, "u_count_pos") },
            words: unsafe { gl.get_uniform_location(program, "u_words") },
            delta: unsafe { gl.get_uniform_location(program, "u_delta") },
            indexed: unsafe { gl.get_uniform_location(program, "u_indexed") },
            program,
        })
    }
    pub(crate) unsafe fn destroy(self, gl: &glow::Context) {
        unsafe { gl.delete_program(self.program) }
    }
}

impl NativeGlProvider {
    /// Called by the context-current device teardown path before the GL
    /// dispatch table is released.
    pub(crate) unsafe fn clear_indirect_count_utility(&mut self) {
        if let Some(utility) = self.indirect_count_utility.take() {
            // SAFETY: caller established that this provider's context is current.
            unsafe { utility.destroy(&self.gl) };
        }
    }

    /// Called by the context-current device teardown path after outstanding
    /// transactions have restored their bindings.
    pub(crate) unsafe fn clear_indirect_count_scratch(&mut self) {
        // SAFETY: caller established that this provider's exact context is current.
        unsafe { self.indirect_count_scratch.clear(&self.gl) };
    }
}

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
        let max_buffer_bytes =
            super::buffer_abi::advertised_native_max_buffer_bytes(self.buffer_functions.is_some());
        if end > max_buffer_bytes
            || count_offset
                .checked_add(4)
                .is_none_or(|v| v > max_buffer_bytes)
        {
            return Err(Self::validation(
                OP,
                "buffer copy offset exceeds native GLintptr",
            ));
        }
        let limits = self.discovery.limits();
        let capacity = ((limits
            .max_storage_block_size
            .min(max_buffer_bytes)
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
        let buffer_bytes = u64::from(chunk_capacity)
            .checked_mul(record)
            .and_then(|value| value.checked_add(4))
            .filter(|value| *value <= max_buffer_bytes)
            .ok_or_else(|| Self::validation(OP, "scratch size exceeds native GLsizeiptr"))?;
        // GL's buffer allocation/copy interface uses signed offsets. Snapshot
        // all records before any draw so later chunks cannot observe changes.
        let snapshot_data = u64::from(count.max_draw_count)
            .checked_mul(record)
            .ok_or_else(|| Self::validation(OP, "snapshot size overflow"))?;
        let snapshot_size = snapshot_data
            .checked_add(4)
            .filter(|value| *value <= max_buffer_bytes)
            .ok_or(GlError::Unsupported {
                operation: OP,
                reason: "indirect snapshot exceeds native GL buffer limit",
            })?;
        self.set_first_instance_uniform(program, 0)?;
        self.ensure_program(OP, program)?;
        // SAFETY: all offsets and sizes are validated against live GL allocations;
        // transient objects and every affected binding/program are restored below.
        unsafe {
            if self.indirect_count_utility.is_none() {
                self.indirect_count_utility = Some(IndirectCountUtility::create(&self.gl)?);
            }
            let (
                utility_program,
                base_uniform,
                draws_uniform,
                count_uniform,
                words_uniform,
                delta_uniform,
                indexed_uniform,
            ) = {
                let utility = self
                    .indirect_count_utility
                    .as_ref()
                    .expect("utility was installed");
                (
                    utility.program,
                    utility.base.clone(),
                    utility.draws.clone(),
                    utility.count_pos.clone(),
                    utility.words.clone(),
                    utility.delta.clone(),
                    utility.indexed.clone(),
                )
            };
            let mut transaction = IndirectCountTransaction::capture(
                self.gl.clone(),
                self.buffer_functions,
                [
                    self.storage_bindings.get(&0).copied(),
                    self.storage_bindings.get(&1).copied(),
                ],
                self.generic_storage_buffer,
                self.current_native_program,
                None,
                None,
                None,
                OP,
            )?;
            let scratch_bytes = buffer_bytes
                .checked_add(buffer_bytes - 4)
                .and_then(|value| value.checked_add(snapshot_size))
                .ok_or_else(|| Self::validation(OP, "scratch size overflow"))?;
            let scratch = self.indirect_count_scratch.lease(
                self.gl.clone(),
                [buffer_bytes, buffer_bytes - 4, snapshot_size],
                OP,
            )?;
            transaction.retain_scratch(scratch, scratch_bytes);
            let [input, output, snapshot] = transaction.scratch_buffers();
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(input));
            self.buffer_data_size_wide(
                glow::COPY_WRITE_BUFFER,
                buffer_bytes,
                glow::STREAM_DRAW,
                OP,
            )?;
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(output));
            self.buffer_data_size_wide(
                glow::COPY_WRITE_BUFFER,
                buffer_bytes - 4,
                glow::STREAM_DRAW,
                OP,
            )?;
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(snapshot));
            self.buffer_data_size_wide(
                glow::COPY_WRITE_BUFFER,
                snapshot_size,
                glow::STREAM_DRAW,
                OP,
            )?;
            self.gl.memory_barrier(glow::BUFFER_UPDATE_BARRIER_BIT);
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(counts));
            self.copy_buffer_sub_data_wide(
                glow::COPY_READ_BUFFER,
                glow::COPY_WRITE_BUFFER,
                count_offset,
                snapshot_data,
                4,
                OP,
            )?;
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(args));
            if stride == record {
                self.copy_buffer_sub_data_wide(
                    glow::COPY_READ_BUFFER,
                    glow::COPY_WRITE_BUFFER,
                    src_offset,
                    0,
                    snapshot_data,
                    OP,
                )?;
            } else {
                for i in 0..count.max_draw_count {
                    self.copy_buffer_sub_data_wide(
                        glow::COPY_READ_BUFFER,
                        glow::COPY_WRITE_BUFFER,
                        src_offset + u64::from(i) * stride,
                        u64::from(i) * record,
                        record,
                        OP,
                    )?;
                }
            }
            let mut base = 0u32;
            while base < count.max_draw_count {
                let draws = (count.max_draw_count - base).min(chunk_capacity);
                let count_pos = draws * words;
                self.gl.memory_barrier(
                    glow::BUFFER_UPDATE_BARRIER_BIT | glow::SHADER_STORAGE_BARRIER_BIT,
                );
                self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(input));
                self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(snapshot));
                self.copy_buffer_sub_data_wide(
                    glow::COPY_READ_BUFFER,
                    glow::COPY_WRITE_BUFFER,
                    snapshot_data,
                    u64::from(count_pos) * 4,
                    4,
                    OP,
                )?;
                self.copy_buffer_sub_data_wide(
                    glow::COPY_READ_BUFFER,
                    glow::COPY_WRITE_BUFFER,
                    u64::from(base) * record,
                    0,
                    u64::from(draws) * record,
                    OP,
                )?;
                self.gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT);
                self.gl
                    .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(input));
                self.gl
                    .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 1, Some(output));
                self.gl.use_program(Some(utility_program));
                self.gl.uniform_1_u32(base_uniform.as_ref(), base);
                self.gl.uniform_1_u32(draws_uniform.as_ref(), draws);
                self.gl.uniform_1_u32(count_uniform.as_ref(), count_pos);
                self.gl.uniform_1_u32(words_uniform.as_ref(), words);
                self.gl.uniform_1_u32(delta_uniform.as_ref(), delta);
                self.gl
                    .uniform_1_u32(indexed_uniform.as_ref(), u32::from(index.is_some()));
                self.gl.dispatch_compute(draws.div_ceil(64), 1, 1);
                self.gl.memory_barrier(glow::COMMAND_BARRIER_BIT);
                // Draw shaders must see their original SSBO bindings, not the
                // private compute utility's input and output buffers.
                transaction.restore_draw_state();
                self.gl
                    .bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(output));
                let mode = super::exec_raster::topology_mode(topology);
                for i in 0..draws {
                    let offset = u64::from(i)
                        .checked_mul(record)
                        .and_then(|value| isize::try_from(value).ok())
                        .expect("scratch allocation bounded every indirect record offset");
                    if let Some(index) = index {
                        if let Some(draw_elements_indirect) = self.draw_functions.elements_indirect
                        {
                            draw_elements_indirect(
                                mode,
                                super::exec_vertex::indexed_draw_type(index),
                                offset as usize as *const core::ffi::c_void,
                            );
                        } else {
                            self.gl.draw_elements_indirect_offset(
                                mode,
                                super::exec_vertex::indexed_draw_type(index),
                                i32::try_from(offset).map_err(|_| {
                                    Self::validation(
                                        OP,
                                        "wide DrawElementsIndirect entry point unavailable",
                                    )
                                })?,
                            );
                        }
                    } else {
                        if let Some(draw_arrays_indirect) = self.draw_functions.arrays_indirect {
                            draw_arrays_indirect(mode, offset as usize as *const core::ffi::c_void);
                        } else {
                            self.gl.draw_arrays_indirect_offset(
                                mode,
                                i32::try_from(offset).map_err(|_| {
                                    Self::validation(
                                        OP,
                                        "wide DrawArraysIndirect entry point unavailable",
                                    )
                                })?,
                            );
                        }
                    }
                }
                base += draws;
            }
            // Normal completion must also observe restoration/deletion errors.
            // Early `?` exits still use the guard's rollback during unwinding.
            drop(transaction);
            self.defer_driver_error(OP)
        }
    }
}
