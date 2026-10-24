//! Native compute, storage, and indirect execution.
//!
//! Every entry point preflights the proved capability, the installed program,
//! and the exact range or binding limits before touching the driver. The
//! compute program is installed through `set_compute_program`; indirect draws
//! reuse the installed raster topology and its recorded index binding.

use super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    GlBufferRange, GlBufferUsage, GlCapability, GlComputeDispatchApi, GlComputeLimits,
    GlDispatchGroups, GlDispatchIndirectApi, GlDispatchIndirectCommand, GlDrawIndirectApi, GlError,
    GlFamilyApi as _, GlIndirectAbi, GlIndirectCommandRange, GlMemoryBarrier, GlProgramKind,
    GlStorageBufferApi, GlStorageBufferLimits, GlStorageBufferRange, GlStorageImageAccess,
    GlStorageImageApi, GlStorageImageBinding, GlStorageImageLimits, GlTextureUsage,
    validate_indirect_allocation,
};

impl GlComputeDispatchApi for NativeGlProvider {
    /// Installs the compute program dispatch work executes.
    fn set_compute_program(
        &mut self,
        program: crate::backend::gl::api::ProgramId,
    ) -> Result<(), GlError> {
        const OP: &str = "set-compute-program";
        self.assert_ready(OP)?;
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::Compute)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the compute capability",
            });
        }
        let record = self.program(OP, program)?;
        if !matches!(&record.descriptor.kind, GlProgramKind::Compute { .. }) {
            return Err(Self::validation(
                OP,
                "compute dispatch requires a compute program",
            ));
        }
        // Compute and raster share GL_CURRENT_PROGRAM. ContextState is the
        // sole authority allowed to skip this selection; the provider's
        // `current_program` remains the observed driver fact after a call.
        let key = NativeGlProvider::state_program_key(program);
        if self.context_state.prepare_compute_program(key) {
            if let Err(error) = self.select_program(OP, program) {
                self.context_state.compute_failed();
                return Err(error);
            }
            self.context_state.commit_compute_program(key);
        }
        self.active_compute_program = Some(program);
        Ok(())
    }

    fn dispatch(&mut self, groups: GlDispatchGroups) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "dispatch";
        self.assert_ready(OP)?;
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::Compute)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the compute capability",
            });
        }
        let Some(program) = self.active_compute_program else {
            return Err(Self::validation(OP, "no compute program is installed"));
        };
        // A raster install or program link may have changed the shared current
        // slot. Reconcile through ContextState again; dispatch itself is never
        // cached or elided.
        let key = NativeGlProvider::state_program_key(program);
        if self.context_state.prepare_compute_program(key) {
            if let Err(error) = self.select_program(OP, program) {
                self.context_state.compute_failed();
                return Err(error);
            }
            self.context_state.commit_compute_program(key);
        }
        let limits = self.discovery.limits();
        if groups
            .validate(GlComputeLimits {
                max_group_count: limits.max_compute_work_group_count,
                max_group_size: limits.max_compute_work_group_size,
                max_group_invocations: limits.max_compute_work_group_invocations,
            })
            .is_err()
        {
            return Err(Self::validation(
                OP,
                "workgroup count is zero or exceeds a discovered axis limit",
            ));
        }
        // SAFETY: current-context contract; capability, program, and group
        // limits were validated before the dispatch.
        unsafe {
            self.gl
                .dispatch_compute(groups.0[0], groups.0[1], groups.0[2]);
        }
        self.driver_error(OP)
    }
}

impl GlStorageBufferApi for NativeGlProvider {
    fn bind_storage_buffer(
        &mut self,
        binding: u32,
        range: GlStorageBufferRange,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "bind-storage-buffer";
        self.assert_ready(OP)?;
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::StorageBuffer)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the storage-buffer capability",
            });
        }
        let limits = self.discovery.limits();
        range.validate(
            binding,
            GlStorageBufferLimits {
                max_bindings: limits.max_storage_buffer_bindings,
                max_block_size: limits.max_storage_block_size,
                offset_alignment: limits.storage_buffer_offset_alignment,
            },
        )?;
        let (name, desc) = self.buffer(OP, range.buffer)?;
        if !desc.usage.contains(GlBufferUsage::STORAGE) {
            return Err(Self::validation(OP, "buffer lacks storage usage"));
        }
        GlBufferRange {
            buffer: range.buffer,
            offset: range.offset,
            size: range.size,
        }
        .validate_for(desc)
        .map_err(|_| Self::validation(OP, "storage buffer range is outside the allocation"))?;
        // SAFETY: current-context contract; binding index, alignment, usage,
        // and range were validated against the live allocation.
        unsafe {
            self.bind_buffer_range_wide(
                glow::SHADER_STORAGE_BUFFER,
                binding,
                Some(name),
                range.offset,
                range.size,
                OP,
            )?;
        }
        self.driver_error(OP)
    }
}

impl GlStorageImageApi for NativeGlProvider {
    fn bind_storage_image(
        &mut self,
        binding: u32,
        image: GlStorageImageBinding,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "bind-storage-image";
        self.assert_ready(OP)?;
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::StorageImage)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the storage-image capability",
            });
        }
        image.validate(
            binding,
            GlStorageImageLimits {
                max_image_units: self.discovery.limits().max_image_units,
            },
            self.discovery.formats(),
        )?;
        let (name, desc) = self.texture(OP, image.texture)?;
        if !desc.usage.contains(GlTextureUsage::STORAGE_BINDING) {
            return Err(Self::validation(OP, "texture lacks storage-binding usage"));
        }
        if desc.format != image.format || desc.sample_count != image.sample_count {
            return Err(Self::validation(
                OP,
                "image format or sample count does not match the allocation",
            ));
        }
        let mip = desc
            .mip_extent(image.level)
            .ok_or_else(|| Self::validation(OP, "storage image mip level is invalid"))?;
        let layered = matches!(
            desc.dimension,
            crate::backend::gl::api::GlTextureDimension::D3
                | crate::backend::gl::api::GlTextureDimension::D2Array
                | crate::backend::gl::api::GlTextureDimension::Cube
        );
        let layer = match (image.layered, image.layer) {
            // GL ignores the layer argument for a layered binding; zero is
            // valid even on implementations that validate ignored arguments.
            (true, None) if layered => 0,
            (false, Some(layer)) if layer < mip.depth_or_layers => i32::try_from(layer)
                .map_err(|_| Self::validation(OP, "storage image layer exceeds GLint"))?,
            _ => {
                return Err(Self::validation(
                    OP,
                    "image layer selection is invalid for the texture shape",
                ));
            }
        };
        let format = native_image_format(image.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "format has no proved native image-unit mapping",
        })?;
        // SAFETY: current-context contract; capability, unit index, layer
        // selection, and the exact format access fact were all validated.
        unsafe {
            self.gl.bind_image_texture(
                binding,
                Some(name),
                i32::try_from(image.level)
                    .map_err(|_| Self::validation(OP, "storage image level exceeds GLint"))?,
                image.layered,
                layer,
                native_image_access(image.access),
                format,
            );
        }
        self.driver_error(OP)
    }
}

impl NativeGlProvider {
    pub(super) fn validate_indirect_driver(
        &self,
        operation: &'static str,
        program: crate::backend::gl::api::ProgramId,
    ) -> Result<(), GlError> {
        let context = self.discovery.context();
        let affected = context.version().contains("ANGLE 2.1.22762")
            && context
                .renderer()
                .contains("AMD Radeon 780M Graphics (0x00001900) Direct3D11")
            && context.renderer().contains("D3D11-32.0.21028.2002");
        if affected
            && matches!(&self.program(operation, program)?.descriptor.kind, GlProgramKind::Raster { vertex, .. } if vertex.text.contains("gl_VertexID"))
        {
            return Err(GlError::Unsupported {
                operation,
                reason: "ANGLE D3D11 indirect gl_VertexID driver defect",
            });
        }
        Ok(())
    }

    pub(super) fn validate_indirect_vertex_state(
        &self,
        operation: &'static str,
    ) -> Result<(), GlError> {
        use crate::backend::gl::api::GlVertexBufferMetadata;
        if self.pass.is_none() {
            return Err(Self::validation(operation, "no active render pass"));
        }
        let bound = self
            .bound_vertex_array
            .ok_or_else(|| Self::validation(operation, "no vertex array is bound"))?;
        let vertex_array = self.vertex_array(operation, bound)?;
        let mut metadata = Vec::with_capacity(
            vertex_array.bindings.len() + usize::from(vertex_array.index.is_some()),
        );
        for binding in &vertex_array.bindings {
            let (_, desc) = self.buffer(operation, binding.buffer)?;
            metadata.push(GlVertexBufferMetadata {
                buffer: binding.buffer,
                byte_length: desc.size,
                usage: desc.usage,
            });
        }
        if let Some(index) = vertex_array.index {
            let (_, desc) = self.buffer(operation, index.buffer)?;
            metadata.push(GlVertexBufferMetadata {
                buffer: index.buffer,
                byte_length: desc.size,
                usage: desc.usage,
            });
        }
        vertex_array
            .layout
            .validate_bindings(
                &vertex_array.bindings,
                vertex_array.index,
                &metadata,
                self.context_stamp(),
            )
            .map_err(|_| Self::validation(operation, "vertex bindings are incomplete or invalid"))
    }

    /// Resolves the live indirect command buffer one indirect range names.
    ///
    /// Two facts make an allocation usable as a command buffer, and both are
    /// proved here so neither indirect verb can forget one. The usage role is
    /// what lets the driver read the bytes as records instead of as caller data.
    /// The allocation bound is the one only this provider can supply: the range
    /// is a window the caller describes, so a window wider than its buffer makes
    /// the driver read an argument list that was never allocated, which no
    /// driver reports and no caller can observe. Both verbs resolve their buffer
    /// through this one function, so the rule and its error have exactly one
    /// spelling here rather than one per verb.
    pub(super) fn indirect_command_buffer(
        &self,
        operation: &'static str,
        range: GlBufferRange,
    ) -> Result<glow::NativeBuffer, GlError> {
        let (name, desc) = self.buffer(operation, range.buffer)?;
        if !desc.usage.contains(GlBufferUsage::INDIRECT) {
            return Err(Self::validation(
                operation,
                "command buffer lacks indirect usage",
            ));
        }
        validate_indirect_allocation(operation, range, desc)?;
        Ok(name)
    }
}

impl GlDrawIndirectApi for NativeGlProvider {
    fn draw_indirect(&mut self, command: GlIndirectCommandRange) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "draw-indirect";
        self.assert_ready(OP)?;
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::IndirectDraw)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the indirect-draw capability",
            });
        }
        command.validate(OP)?;
        self.validate_indirect_vertex_state(OP)?;
        let (program, topology) = {
            let raster = self
                .raster
                .as_ref()
                .ok_or_else(|| Self::validation(OP, "no raster pipeline is installed"))?;
            (raster.program, raster.topology)
        };
        self.validate_indirect_driver(OP, program)?;
        self.set_push_uniforms(
            program,
            &[(
                naga::back::glsl::FIRST_INSTANCE_BINDING.to_owned(),
                super::push_uniform::GlPushUniformValue::Uint(vec![0]),
            )],
        )?;
        // This is a raster verb that lives beside the compute ones, so the
        // program the compute path may have selected since the pipeline was
        // installed is re-asserted here exactly as `draw_raster` does it.
        self.ensure_program(OP, program)?;
        // The array comes from the binding record rather than from the pipeline
        // for the same reason it does in `draw_raster`: the geometry domain
        // replaces the installed array between the install and the draw whenever
        // the execution mode is uncached.
        let bound = self
            .bound_vertex_array
            .ok_or_else(|| Self::validation(OP, "no vertex array is bound"))?;
        let index = self.vertex_array(OP, bound)?.index;
        if command.abi == GlIndirectAbi::Indexed && index.is_none() {
            return Err(Self::validation(
                OP,
                "indexed indirect draw requires a bound index buffer",
            ));
        }
        let source = self.indirect_command_buffer(OP, command.range)?;
        let original_offset = draw_indirect_offset(command.range, command.command_offset)?;
        let scratch = if let Some(index) = index.filter(|_| command.abi == GlIndirectAbi::Indexed) {
            self.buffer(OP, index.buffer)?;
            if index.offset != 0 {
                Some(self.rebase_indirect_indices(source, command, index)?)
            } else {
                None
            }
        } else {
            None
        };
        let name = scratch.unwrap_or(source);
        let offset = if scratch.is_some() {
            0
        } else {
            original_offset
        };
        let mode = super::exec_raster::topology_mode(topology);
        // SAFETY: the argument allocation and index state were validated; a rebased scratch buffer is retained until after the call.
        unsafe {
            self.gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(name));
            if let Some(index) = index.filter(|_| command.abi == GlIndirectAbi::Indexed) {
                if let Some(draw_elements_indirect) = self.draw_functions.elements_indirect {
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
                            Self::validation(OP, "wide DrawArraysIndirect entry point unavailable")
                        })?,
                    );
                }
            }
            if let Some(scratch) = scratch {
                self.gl.delete_buffer(scratch);
            }
        }
        self.driver_error(OP)
    }
}

impl NativeGlProvider {
    pub(crate) fn supports_indirect_count(&self) -> bool {
        (self.indirect_count.is_some()
            || (matches!(self.discovery.context().profile(), crate::backend::gl::api::GlFamilyProfile::Embedded { major, minor } if major > 3 || major == 3 && minor >= 1)
                && self
                    .discovery
                    .capabilities()
                    .supports(GlCapability::Compute)
                && self
                    .discovery
                    .capabilities()
                    .supports(GlCapability::StorageBuffer)
                && self.discovery.limits().max_storage_buffer_bindings >= 2))
            && self
                .discovery
                .capabilities()
                .supports(GlCapability::IndirectDraw)
    }

    pub(crate) fn multi_draw_count(
        &mut self,
        command: GlIndirectCommandRange,
        count: crate::backend::gl::api::GlIndirectCountRange,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "multi-draw-indirect-count";
        self.assert_ready(OP)?;
        if self.indirect_count.is_none() && self.supports_indirect_count() {
            return self.emulate_indirect_count(command, count);
        }
        let functions = self.indirect_count.ok_or(GlError::Unsupported {
            operation: OP,
            reason: "indirect count entry points are unavailable",
        })?;
        if !self.supports_indirect_count() {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "indirect draw is unavailable",
            });
        }
        command.validate(OP)?;
        self.validate_indirect_vertex_state(OP)?;
        count.validate(OP)?;
        if command.draw_count != count.max_draw_count {
            return Err(Self::validation(OP, "command and count bounds differ"));
        }
        let max = i32::try_from(count.max_draw_count)
            .map_err(|_| Self::validation(OP, "draw count exceeds GLsizei"))?;
        let stride = i32::try_from(command.stride)
            .map_err(|_| Self::validation(OP, "stride exceeds GLsizei"))?;
        let count_offset = count
            .range
            .offset
            .checked_add(count.count_offset)
            .and_then(|v| isize::try_from(v).ok())
            .ok_or_else(|| Self::validation(OP, "count offset exceeds GLintptr"))?;
        let (program, topology) = {
            let raster = self
                .raster
                .as_ref()
                .ok_or_else(|| Self::validation(OP, "no raster pipeline is installed"))?;
            (raster.program, raster.topology)
        };
        self.validate_indirect_driver(OP, program)?;
        self.set_push_uniforms(
            program,
            &[(
                naga::back::glsl::FIRST_INSTANCE_BINDING.to_owned(),
                super::push_uniform::GlPushUniformValue::Uint(vec![0]),
            )],
        )?;
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
        let args = self.indirect_command_buffer(OP, command.range)?;
        let counts = self.indirect_command_buffer(OP, count.range)?;
        let original_offset = draw_indirect_offset(command.range, command.command_offset)?;
        let scratch = if let Some(index) = index.filter(|index| index.offset != 0) {
            Some(self.rebase_indirect_indices(args, command, index)?)
        } else {
            None
        };
        let offset = if scratch.is_some() {
            0
        } else {
            original_offset
        };
        let stride = if scratch.is_some() { 20 } else { stride };
        if let Err(error) = self.ensure_program(OP, program) {
            if let Some(scratch) = scratch {
                // SAFETY: the scratch name was allocated by this current context and has not escaped.
                unsafe {
                    self.gl.delete_buffer(scratch);
                }
            }
            return Err(error);
        }
        // SAFETY: the validated parameter and command buffers remain live through the draw.
        unsafe {
            self.gl
                .bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(scratch.unwrap_or(args)));
            self.gl.bind_buffer(0x80EE, Some(counts)); // GL_PARAMETER_BUFFER[_ARB]
            let ptr = offset as usize as *const core::ffi::c_void;
            let mode = super::exec_raster::topology_mode(topology);
            if let Some(index) = index {
                (functions.elements)(
                    mode,
                    super::exec_vertex::indexed_draw_type(index),
                    ptr,
                    count_offset,
                    max,
                    stride,
                );
            } else {
                (functions.arrays)(mode, ptr, count_offset, max, stride);
            }
            if let Some(scratch) = scratch {
                self.gl.delete_buffer(scratch);
            }
        }
        self.driver_error(OP)
    }
}

impl GlDispatchIndirectApi for NativeGlProvider {
    fn dispatch_indirect(&mut self, command: GlDispatchIndirectCommand) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "dispatch-indirect";
        self.assert_ready(OP)?;
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::IndirectDispatch)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the indirect-dispatch capability",
            });
        }
        let Some(program) = self.active_compute_program else {
            return Err(Self::validation(OP, "no compute program is installed"));
        };
        self.ensure_program(OP, program)?;
        // The record layout and its in-range position are the module's
        // contract, so the whole check happens before the binding changes.
        command.validate(OP)?;
        let name = self.indirect_command_buffer(OP, command.range)?;
        let offset = dispatch_indirect_offset(command.range, command.command_offset)?;
        // SAFETY: current-context contract; the range, usage, and program were
        // validated before the binding changed.
        unsafe {
            self.gl
                .bind_buffer(glow::DISPATCH_INDIRECT_BUFFER, Some(name));
            if let Some(dispatch_indirect) = self.draw_functions.dispatch_indirect {
                dispatch_indirect(offset);
            } else {
                self.gl
                    .dispatch_compute_indirect(i32::try_from(offset).map_err(|_| {
                        Self::validation(OP, "wide DispatchComputeIndirect entry point unavailable")
                    })?);
            }
        }
        self.driver_error(OP)
    }
}

/// Resolves the buffer offset GL reads one indirect record from.
///
/// The contract places `command_offset` relative to `range.size`, while GL
/// reads the record at an offset that is absolute into whatever is bound to the
/// matching `*_INDIRECT_BUFFER`. The record's buffer address is therefore the
/// range offset plus the record position, and both indirect verbs must produce
/// exactly that. They share this one expression because a verb that drops
/// `range.offset` still draws -- it just reads somebody else's arguments, which
/// no driver reports and no caller can see, so the divergence has to be made
/// impossible rather than found.
fn resolve_indirect_offset(
    operation: &'static str,
    range: GlBufferRange,
    command_offset: u64,
) -> Result<isize, GlError> {
    let absolute = range
        .offset
        .checked_add(command_offset)
        .ok_or(GlError::Validation {
            operation,
            message: "indirect record offset overflows the buffer address space".into(),
        })?;
    isize::try_from(absolute).map_err(|_| GlError::Validation {
        operation,
        message: "indirect record offset exceeds GLintptr".into(),
    })
}

/// The offset the raster indirect-draw verb hands to GL.
///
/// This exists as its own function, rather than as a call to
/// `resolve_indirect_offset` at the call site, so that a test can pin the
/// raster verb's convention without a live context. The two verbs are the pair
/// that drifted apart, and the drift was only visible by comparing them.
pub(super) fn draw_indirect_offset(
    range: GlBufferRange,
    command_offset: u64,
) -> Result<isize, GlError> {
    resolve_indirect_offset("draw-indirect", range, command_offset)
}

/// The offset the indirect-dispatch verb hands to GL.
pub(super) fn dispatch_indirect_offset(
    range: GlBufferRange,
    command_offset: u64,
) -> Result<isize, GlError> {
    resolve_indirect_offset("dispatch-indirect", range, command_offset)
}

/// Translates the contract's barrier classes into GL memory-barrier bits.
pub(super) const fn native_barrier_bits(barriers: u32) -> u32 {
    let mut bits = 0;
    if barriers & GlMemoryBarrier::VERTEX_ATTRIB_ARRAY.0 != 0 {
        bits |= glow::VERTEX_ATTRIB_ARRAY_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::ELEMENT_ARRAY.0 != 0 {
        bits |= glow::ELEMENT_ARRAY_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::UNIFORM.0 != 0 {
        bits |= glow::UNIFORM_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::TEXTURE_FETCH.0 != 0 {
        bits |= glow::TEXTURE_FETCH_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::SHADER_IMAGE_ACCESS.0 != 0 {
        bits |= glow::SHADER_IMAGE_ACCESS_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::COMMAND.0 != 0 {
        bits |= glow::COMMAND_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::PIXEL_BUFFER.0 != 0 {
        bits |= glow::PIXEL_BUFFER_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::TEXTURE_UPDATE.0 != 0 {
        bits |= glow::TEXTURE_UPDATE_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::BUFFER_UPDATE.0 != 0 {
        bits |= glow::BUFFER_UPDATE_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::FRAMEBUFFER.0 != 0 {
        bits |= glow::FRAMEBUFFER_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::TRANSFORM_FEEDBACK.0 != 0 {
        bits |= glow::TRANSFORM_FEEDBACK_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::ATOMIC_COUNTER.0 != 0 {
        bits |= glow::ATOMIC_COUNTER_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::SHADER_STORAGE.0 != 0 {
        bits |= glow::SHADER_STORAGE_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::CLIENT_MAPPED_BUFFER.0 != 0 {
        bits |= glow::CLIENT_MAPPED_BUFFER_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::QUERY_BUFFER.0 != 0 {
        bits |= glow::QUERY_BUFFER_BARRIER_BIT;
    }
    bits
}

/// The image-unit format constant for one proved storage format.
const fn native_image_format(format: crate::backend::gl::api::GlFormat) -> Option<u32> {
    super::resource::native_texture_format(format)
}

const fn native_image_access(access: GlStorageImageAccess) -> u32 {
    match access {
        GlStorageImageAccess::ReadOnly => glow::READ_ONLY,
        GlStorageImageAccess::WriteOnly => glow::WRITE_ONLY,
        GlStorageImageAccess::ReadWrite => glow::READ_WRITE,
    }
}
