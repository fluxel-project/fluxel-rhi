//! Native vertex-array execution.
//!
//! Like the browser backend, a VAO cannot pre-store a layout without buffers,
//! so `create_vertex_array` records the structural layout and
//! `bind_vertex_array` re-emits attribute pointers against the supplied
//! bindings; redundant-call elimination belongs to the Layer 2 cache, never
//! to this contract.

use super::provider::{NativeGlProvider, NativeVertexArray};
use crate::backend::gl::api::{
    GlError, GlFamilyApi as _, GlIndexBinding, GlIndexFormat, GlVertexApi, GlVertexBufferBinding,
    GlVertexBufferMetadata, GlVertexFormat, GlVertexLayout, GlVertexStepMode, VertexArrayId,
};

/// How one attribute reaches its shader input.
enum AttributePath {
    /// `vertexAttribPointer` with these `(size, type, normalized)` values.
    Float(i32, u32, bool),
    /// `vertexAttribIPointer` with these `(size, type)` values.
    Integer(i32, u32),
    /// `vertexAttribLPointer` for double-precision shader inputs.
    Double(i32),
}

const fn attribute_path(format: GlVertexFormat) -> AttributePath {
    use GlVertexFormat as F;
    match format {
        F::Uint8x2 => AttributePath::Integer(2, glow::UNSIGNED_BYTE),
        F::Uint8x4 => AttributePath::Integer(4, glow::UNSIGNED_BYTE),
        F::Sint8x2 => AttributePath::Integer(2, glow::BYTE),
        F::Sint8x4 => AttributePath::Integer(4, glow::BYTE),
        F::Unorm8x2 => AttributePath::Float(2, glow::UNSIGNED_BYTE, true),
        F::Unorm8x4 => AttributePath::Float(4, glow::UNSIGNED_BYTE, true),
        F::Snorm8x2 => AttributePath::Float(2, glow::BYTE, true),
        F::Snorm8x4 => AttributePath::Float(4, glow::BYTE, true),
        F::Uint16x2 => AttributePath::Integer(2, glow::UNSIGNED_SHORT),
        F::Uint16x4 => AttributePath::Integer(4, glow::UNSIGNED_SHORT),
        F::Sint16x2 => AttributePath::Integer(2, glow::SHORT),
        F::Sint16x4 => AttributePath::Integer(4, glow::SHORT),
        F::Unorm16x2 => AttributePath::Float(2, glow::UNSIGNED_SHORT, true),
        F::Unorm16x4 => AttributePath::Float(4, glow::UNSIGNED_SHORT, true),
        F::Snorm16x2 => AttributePath::Float(2, glow::SHORT, true),
        F::Snorm16x4 => AttributePath::Float(4, glow::SHORT, true),
        F::Float16x2 => AttributePath::Float(2, glow::HALF_FLOAT, false),
        F::Float16x4 => AttributePath::Float(4, glow::HALF_FLOAT, false),
        F::Float64 => AttributePath::Double(1),
        F::Float64x2 => AttributePath::Double(2),
        F::Float64x3 => AttributePath::Double(3),
        F::Float64x4 => AttributePath::Double(4),
        F::Float32 => AttributePath::Float(1, glow::FLOAT, false),
        F::Float32x2 => AttributePath::Float(2, glow::FLOAT, false),
        F::Float32x3 => AttributePath::Float(3, glow::FLOAT, false),
        F::Float32x4 => AttributePath::Float(4, glow::FLOAT, false),
        F::Uint32 => AttributePath::Integer(1, glow::UNSIGNED_INT),
        F::Uint32x2 => AttributePath::Integer(2, glow::UNSIGNED_INT),
        F::Uint32x3 => AttributePath::Integer(3, glow::UNSIGNED_INT),
        F::Uint32x4 => AttributePath::Integer(4, glow::UNSIGNED_INT),
        F::Sint32 => AttributePath::Integer(1, glow::INT),
        F::Sint32x2 => AttributePath::Integer(2, glow::INT),
        F::Sint32x3 => AttributePath::Integer(3, glow::INT),
        F::Sint32x4 => AttributePath::Integer(4, glow::INT),
    }
}

const fn index_type(format: GlIndexFormat) -> u32 {
    match format {
        GlIndexFormat::Uint16 => glow::UNSIGNED_SHORT,
        GlIndexFormat::Uint32 => glow::UNSIGNED_INT,
    }
}

const fn index_size(format: GlIndexFormat) -> u64 {
    match format {
        GlIndexFormat::Uint16 => 2,
        GlIndexFormat::Uint32 => 4,
    }
}

impl NativeGlProvider {
    /// `glVertexAttribLPointer` is core in desktop GL 4.1 and is exposed by
    /// `GL_ARB_vertex_attrib_64bit` on otherwise supported GL 4.0 contexts.
    /// GLES deliberately has no route: its vertex input API has no matching
    /// double-precision attribute representation.
    fn supports_f64_vertex_attributes(&self) -> bool {
        match self.discovery.context().profile() {
            crate::backend::gl::api::GlFamilyProfile::Desktop { major, minor } => {
                ((major, minor) >= (4, 1)
                    || self
                        .discovery
                        .extensions()
                        .raw_reported_names()
                        .any(|name| name == "GL_ARB_vertex_attrib_64bit"))
                    && self
                        .vertex_functions
                        .is_some_and(super::buffer_abi::NativeVertexFunctions::supports_f64)
            }
            crate::backend::gl::api::GlFamilyProfile::Embedded { .. }
            | crate::backend::gl::api::GlFamilyProfile::WebGl2 => false,
        }
    }

    /// Validates the mutable bindings of the VAO that the driver currently
    /// holds, without cloning its immutable layout/bindings or constructing a
    /// per-draw metadata vector.
    ///
    /// `create_vertex_array` validates the structural layout before publishing
    /// it. Bindings and allocation lifetimes remain mutable, so this keeps the
    /// draw-time context, liveness, role, offset, stride-zero and index checks
    /// that prevent a native draw from reading beyond an allocation.
    pub(super) fn validate_current_vertex_array_bindings(
        &self,
        operation: &'static str,
    ) -> Result<Option<GlIndexBinding>, GlError> {
        let bound = self
            .bound_vertex_array
            .ok_or_else(|| Self::validation(operation, "no vertex array is bound"))?;
        let vertex_array = self.vertex_array(operation, bound)?;
        let index = vertex_array.index;

        // Preserve the native provider's liveness/context checks before the
        // layout validator uses allocation facts. This has no allocation and
        // keeps stale resource rejection from becoming a generic missing-facts
        // validation error.
        for binding in &vertex_array.bindings {
            self.buffer(operation, binding.buffer)?;
        }
        if let Some(index) = index {
            self.buffer(operation, index.buffer)?;
        }

        vertex_array
            .layout
            .validate_bound_bindings_with(
                &vertex_array.bindings,
                index,
                self.context_stamp(),
                |buffer| {
                    self.buffers
                        .get(&buffer)
                        .map(|(_, desc)| GlVertexBufferMetadata {
                            buffer,
                            byte_length: desc.size,
                            usage: desc.usage,
                        })
                },
            )
            .map_err(|_| {
                Self::validation(operation, "vertex bindings are incomplete or invalid")
            })?;
        Ok(index)
    }
}

impl GlVertexApi for NativeGlProvider {
    fn create_vertex_array(&mut self, layout: &GlVertexLayout) -> Result<VertexArrayId, GlError> {
        use glow::HasContext as _;
        const OP: &str = "create-vertex-array";
        self.assert_ready(OP)?;
        layout
            .validate()
            .map_err(|_| Self::validation(OP, "invalid vertex layout"))?;
        let max_attributes = self.discovery.limits().max_vertex_attributes;
        if layout
            .attributes
            .iter()
            .any(|attribute| attribute.location >= max_attributes)
        {
            return Err(Self::validation(
                OP,
                "attribute location exceeds the discovered limit",
            ));
        }
        if layout.attributes.iter().any(|attribute| {
            matches!(
                attribute.format,
                GlVertexFormat::Float64
                    | GlVertexFormat::Float64x2
                    | GlVertexFormat::Float64x3
                    | GlVertexFormat::Float64x4
            )
        }) && !self.supports_f64_vertex_attributes()
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "Float64 vertex attributes require desktop GL 4.1 or GL_ARB_vertex_attrib_64bit",
            });
        }
        // SAFETY: current-context contract.
        let raw = unsafe { self.gl.create_vertex_array() }.map_err(|message| GlError::Driver {
            operation: OP,
            message,
        })?;
        let slot = self.slot(OP)?;
        let id = VertexArrayId::new(self.context_stamp(), slot, 0);
        self.vertex_arrays.insert(
            id,
            NativeVertexArray {
                generation: id.generation,
                raw,
                layout: layout.clone(),
                index: None,
                bindings: Vec::new(),
            },
        );
        Ok(id)
    }

    fn destroy_vertex_array(&mut self, vertex_array: VertexArrayId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "destroy-vertex-array";
        self.assert_ready(OP)?;
        self.vertex_array(OP, vertex_array)?;
        let entry = self
            .vertex_arrays
            .remove(&vertex_array)
            .ok_or_else(|| Self::validation(OP, "vertex array disappeared"))?;
        // SAFETY: current-context contract; liveness was checked first.
        unsafe { self.gl.delete_vertex_array(entry.raw) };
        self.driver_error(OP)
    }

    fn bind_vertex_array(
        &mut self,
        vertex_array: VertexArrayId,
        buffers: &[GlVertexBufferBinding],
        index: Option<GlIndexBinding>,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "bind-vertex-array";
        self.assert_ready(OP)?;
        self.vertex_array(OP, vertex_array)?;
        // Allocation facts come from this provider's own table, so callers
        // never forge byte lengths.
        let mut metadata: Vec<GlVertexBufferMetadata> = Vec::with_capacity(buffers.len() + 1);
        for binding in buffers {
            let entry = self.buffer(OP, binding.buffer)?;
            metadata.push(GlVertexBufferMetadata {
                buffer: binding.buffer,
                byte_length: entry.1.size,
                usage: entry.1.usage,
            });
        }
        if let Some(index) = index {
            let entry = self.buffer(OP, index.buffer)?;
            // The index buffer reaches validation through the same table as
            // the attribute buffers, so its size and roles are checked there
            // rather than re-derived here.
            metadata.push(GlVertexBufferMetadata {
                buffer: index.buffer,
                byte_length: entry.1.size,
                usage: entry.1.usage,
            });
        }
        let layout = {
            match self.vertex_arrays.get(&vertex_array) {
                Some(entry) if entry.generation == vertex_array.generation => entry.layout.clone(),
                _ => return Err(Self::validation(OP, "vertex array disappeared")),
            }
        };
        layout
            .validate_bindings(buffers, index, &metadata, self.context_stamp())
            .map_err(|_| Self::validation(OP, "invalid vertex bindings"))?;
        let raw = match self.vertex_arrays.get(&vertex_array) {
            Some(entry) => entry.raw,
            None => return Err(Self::validation(OP, "vertex array disappeared")),
        };
        // SAFETY: current-context contract; every binding was validated above
        // against the live allocation table before any GL state changed.
        unsafe {
            self.gl.bind_vertex_array(Some(raw));
            // Recorded the moment the driver takes the binding rather than after
            // the rest of this verb succeeds, because the record answers what the
            // driver holds and not what this call intended: an attribute emission
            // that fails below leaves the array bound all the same, and a draw
            // that trusted a stale record would re-bind an array the caller never
            // asked for.  This is one of the two writers of that field.
            self.bound_vertex_array = Some(vertex_array);
            self.emit_attributes(&layout, buffers)?;
            // ELEMENT_ARRAY_BUFFER state lives inside the VAO, so the index
            // binding is recorded for draw-time bounds checks and byte offsets.
            if let Some(index) = index {
                let raw = self
                    .buffers
                    .get(&index.buffer)
                    .map(|entry| entry.0)
                    .ok_or_else(|| Self::validation(OP, "index buffer disappeared"))?;
                self.gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(raw));
            } else {
                self.gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);
            }
        }
        let result = self.driver_error(OP);
        match result {
            Ok(()) => {
                if let Some(entry) = self.vertex_arrays.get_mut(&vertex_array) {
                    entry.index = index;
                    entry.bindings = buffers.to_vec();
                }
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    fn set_vertex_binding(
        &mut self,
        vertex_array: VertexArrayId,
        binding: GlVertexBufferBinding,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "set-vertex-binding";
        let (mut bindings, layout, raw) = {
            let entry = self.vertex_array("set-vertex-binding", vertex_array)?;
            (entry.bindings.clone(), entry.layout.clone(), entry.raw)
        };
        if !layout.buffers.iter().any(|slot| slot.slot == binding.slot) {
            return Err(Self::validation(
                OP,
                "binding slot is absent from the vertex layout",
            ));
        }
        self.buffer(OP, binding.buffer)?;
        // SAFETY: the VAO and buffer came from this provider's live tables;
        // the partial layout emits only this slot, so other required slots may
        // be installed later without turning a setter into deferred state.
        unsafe {
            self.gl.bind_vertex_array(Some(raw));
            self.bound_vertex_array = Some(vertex_array);
            let partial = GlVertexLayout {
                buffers: layout
                    .buffers
                    .into_iter()
                    .filter(|slot| slot.slot == binding.slot)
                    .collect(),
                attributes: layout.attributes,
            };
            self.emit_attributes(&partial, core::slice::from_ref(&binding))?;
        }
        if let Some(old) = bindings.iter_mut().find(|old| old.slot == binding.slot) {
            *old = binding;
        } else {
            bindings.push(binding);
        }
        self.vertex_arrays.get_mut(&vertex_array).unwrap().bindings = bindings;
        self.driver_error(OP)
    }

    fn set_index_binding(
        &mut self,
        vertex_array: VertexArrayId,
        index: Option<GlIndexBinding>,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "set-index-binding";
        let raw = self.vertex_array(OP, vertex_array)?.raw;
        if let Some(index) = index {
            self.buffer(OP, index.buffer)?;
        }
        // SAFETY: the VAO and optional index buffer are live provider objects;
        // ELEMENT_ARRAY_BUFFER is VAO state and needs no vertex-slot validation.
        unsafe {
            self.gl.bind_vertex_array(Some(raw));
            self.gl.bind_buffer(
                glow::ELEMENT_ARRAY_BUFFER,
                index.and_then(|i| self.buffers.get(&i.buffer).map(|e| e.0)),
            );
        }
        self.bound_vertex_array = Some(vertex_array);
        self.vertex_arrays.get_mut(&vertex_array).unwrap().index = index;
        self.driver_error(OP)
    }
}

impl NativeGlProvider {
    /// Emits one `vertexAttribPointer` group per layout slot.
    ///
    /// # Safety
    ///
    /// The provider context must be current on its owner thread and the caller
    /// must bind a live VAO from this provider first. `layout` and `buffers`
    /// must have passed the binding validation; this helper only emits their
    /// immediate VAO state and publishes no ownership.
    pub(super) unsafe fn emit_attributes(
        &self,
        layout: &GlVertexLayout,
        buffers: &[GlVertexBufferBinding],
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        unsafe {
            let vertex_functions = self.vertex_functions;
            if layout.buffers.iter().any(|slot| slot.stride == 0) {
                let profile = self.discovery.context().profile();
                let available = profile.has_core(
                    crate::backend::gl::api::GlVersion::new(4, 3),
                    crate::backend::gl::api::GlVersion::new(3, 1),
                ) || self
                    .discovery
                    .extensions()
                    .raw_reported_names()
                    .any(|name| name == "GL_ARB_vertex_attrib_binding");
                if !available || !self.vertex_binding_entries {
                    return Err(GlError::Unsupported {
                        operation: "bind-vertex-array",
                        reason: "zero stride requires vertex attribute binding",
                    });
                }
                let max = self.gl.get_parameter_i32(glow::MAX_VERTEX_ATTRIB_BINDINGS);
                if layout
                    .attributes
                    .iter()
                    .filter(|a| {
                        layout
                            .buffers
                            .iter()
                            .any(|slot| slot.stride == 0 && slot.slot == a.buffer_slot)
                    })
                    .any(|a| a.location >= max as u32)
                {
                    return Err(GlError::Unsupported {
                        operation: "bind-vertex-array",
                        reason: "zero stride vertex binding limit",
                    });
                }
            }
            for slot_layout in &layout.buffers {
                let Some(binding) = buffers
                    .iter()
                    .find(|binding| binding.slot == slot_layout.slot)
                else {
                    return Err(Self::validation(
                        "bind-vertex-array",
                        "layout slot lost its binding",
                    ));
                };
                let raw = self
                    .buffers
                    .get(&binding.buffer)
                    .map(|entry| entry.0)
                    .ok_or_else(|| {
                        Self::validation("bind-vertex-array", "bound buffer disappeared")
                    })?;
                self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(raw));
                for attribute in layout
                    .attributes
                    .iter()
                    .filter(|attribute| attribute.buffer_slot == slot_layout.slot)
                {
                    let stride = slot_layout.stride;
                    let offset = binding.offset.checked_add(u64::from(attribute.offset));
                    let offset = match offset.and_then(|value| isize::try_from(value).ok()) {
                        Some(offset) => offset,
                        None => {
                            return Err(Self::validation(
                                "bind-vertex-array",
                                "attribute offset exceeds native GLintptr",
                            ));
                        }
                    };
                    if stride == 0 {
                        // Separate attribute binding interprets zero as a fixed address,
                        // unlike the legacy pointer API where zero means tightly packed.
                        if matches!(attribute_path(attribute.format), AttributePath::Double(_))
                            && !vertex_functions.is_some_and(
                                super::buffer_abi::NativeVertexFunctions::supports_zero_stride_f64,
                            )
                        {
                            return Err(GlError::Unsupported {
                                operation: "bind-vertex-array",
                                reason: "zero-stride Float64 attributes require native BindVertexBuffer and VertexAttribLFormat",
                            });
                        }
                        if let Some(bind_vertex_buffer) =
                            vertex_functions.and_then(|functions| functions.bind_vertex_buffer)
                        {
                            bind_vertex_buffer(attribute.location, raw.0.get(), offset, 0);
                        } else {
                            self.gl.bind_vertex_buffer(
                                attribute.location,
                                Some(raw),
                                i32::try_from(offset).map_err(|_| {
                                    Self::validation(
                                        "bind-vertex-array",
                                        "wide BindVertexBuffer entry point unavailable",
                                    )
                                })?,
                                0,
                            );
                        }
                        match attribute_path(attribute.format) {
                            AttributePath::Float(size, type_, normalized) => {
                                self.gl.vertex_attrib_format_f32(
                                    attribute.location,
                                    size,
                                    type_,
                                    normalized,
                                    0,
                                )
                            }
                            AttributePath::Integer(size, type_) => self
                                .gl
                                .vertex_attrib_format_i32(attribute.location, size, type_, 0),
                            AttributePath::Double(size) => (vertex_functions
                                .and_then(|functions| functions.attrib_l_format)
                                .expect("zero-stride Float64 preflight loaded LFormat"))(
                                attribute.location,
                                size,
                                glow::DOUBLE,
                                0,
                            ),
                        }
                        self.gl
                            .vertex_attrib_binding(attribute.location, attribute.location);
                        self.gl.vertex_binding_divisor(
                            attribute.location,
                            match slot_layout.step_mode {
                                GlVertexStepMode::Vertex => 0,
                                GlVertexStepMode::Instance => 1,
                            },
                        );
                        self.gl.enable_vertex_attrib_array(attribute.location);
                        continue;
                    }
                    match attribute_path(attribute.format) {
                        AttributePath::Float(size, type_, normalized) => {
                            if let Some(functions) = vertex_functions {
                                (functions.attrib_pointer)(
                                    attribute.location,
                                    size,
                                    type_,
                                    u8::from(normalized),
                                    stride as i32,
                                    offset as usize as *const core::ffi::c_void,
                                );
                            } else {
                                self.gl.vertex_attrib_pointer_f32(
                                    attribute.location,
                                    size,
                                    type_,
                                    normalized,
                                    stride as i32,
                                    i32::try_from(offset).map_err(|_| {
                                        Self::validation(
                                            "bind-vertex-array",
                                            "wide vertex pointer entry point unavailable",
                                        )
                                    })?,
                                );
                            }
                        }
                        AttributePath::Integer(size, type_) => {
                            if let Some(functions) = vertex_functions {
                                (functions.attrib_i_pointer)(
                                    attribute.location,
                                    size,
                                    type_,
                                    stride as i32,
                                    offset as usize as *const core::ffi::c_void,
                                );
                            } else {
                                self.gl.vertex_attrib_pointer_i32(
                                    attribute.location,
                                    size,
                                    type_,
                                    stride as i32,
                                    i32::try_from(offset).map_err(|_| {
                                        Self::validation(
                                            "bind-vertex-array",
                                            "wide vertex pointer entry point unavailable",
                                        )
                                    })?,
                                );
                            }
                        }
                        AttributePath::Double(size) => {
                            (vertex_functions
                                .and_then(|functions| functions.attrib_l_pointer)
                                .expect("Float64 support preflight loaded LPointer"))(
                                attribute.location,
                                size,
                                glow::DOUBLE,
                                stride as i32,
                                offset as usize as *const core::ffi::c_void,
                            );
                        }
                    }
                    self.gl.enable_vertex_attrib_array(attribute.location);
                    self.gl.vertex_attrib_divisor(
                        attribute.location,
                        match slot_layout.step_mode {
                            GlVertexStepMode::Vertex => 0,
                            GlVertexStepMode::Instance => 1,
                        },
                    );
                }
            }
        }
        Ok(())
    }
}

pub(super) const fn attribute_byte_size(format: GlVertexFormat) -> u32 {
    // The same closed table `GlVertexFormat::byte_size` uses, kept local so
    // this module does not depend on private trait items.
    match format {
        GlVertexFormat::Uint8x2
        | GlVertexFormat::Sint8x2
        | GlVertexFormat::Unorm8x2
        | GlVertexFormat::Snorm8x2 => 2,
        GlVertexFormat::Uint8x4
        | GlVertexFormat::Sint8x4
        | GlVertexFormat::Unorm8x4
        | GlVertexFormat::Snorm8x4 => 4,
        GlVertexFormat::Uint16x2
        | GlVertexFormat::Sint16x2
        | GlVertexFormat::Unorm16x2
        | GlVertexFormat::Snorm16x2
        | GlVertexFormat::Float16x2 => 4,
        GlVertexFormat::Uint16x4
        | GlVertexFormat::Sint16x4
        | GlVertexFormat::Unorm16x4
        | GlVertexFormat::Snorm16x4
        | GlVertexFormat::Float16x4 => 8,
        GlVertexFormat::Float32 | GlVertexFormat::Uint32 | GlVertexFormat::Sint32 => 4,
        GlVertexFormat::Float32x2 | GlVertexFormat::Uint32x2 | GlVertexFormat::Sint32x2 => 8,
        GlVertexFormat::Float32x3 | GlVertexFormat::Uint32x3 | GlVertexFormat::Sint32x3 => 12,
        GlVertexFormat::Float32x4 | GlVertexFormat::Uint32x4 | GlVertexFormat::Sint32x4 => 16,
        GlVertexFormat::Float64 => 8,
        GlVertexFormat::Float64x2 => 16,
        GlVertexFormat::Float64x3 => 24,
        GlVertexFormat::Float64x4 => 32,
    }
}

/// Total bytes one indexed draw consumes, from its base binding offset.
pub(super) fn indexed_draw_span(
    index: GlIndexBinding,
    first_index: u32,
    index_count: u32,
) -> Option<u64> {
    let size = index_size(index.format);
    let start = u64::from(first_index).checked_mul(size)?;
    let count = u64::from(index_count).checked_mul(size)?;
    index.offset.checked_add(start)?.checked_add(count)
}

/// The index type constant for an index binding.
pub(super) const fn indexed_draw_type(index: GlIndexBinding) -> u32 {
    index_type(index.format)
}

/// The byte offset one indexed draw starts at.
pub(super) fn indexed_draw_offset(index: GlIndexBinding, first_index: u32) -> Option<u64> {
    let size = index_size(index.format);
    index
        .offset
        .checked_add(u64::from(first_index).checked_mul(size)?)
}
