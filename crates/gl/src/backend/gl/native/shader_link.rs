//! Native shader compilation, linking, and reflection internals.
//!
//! The public shader API owns object-table publication. This module performs
//! the transactional driver work beneath it: compile temporary stage objects,
//! link and reflect one program, then either return complete evidence or
//! delete every temporary/native name before an error escapes.

use super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    BufferId, GlBindingApi, GlBindingLimits, GlBufferUsage, GlError, GlExecutableBindingAssignment,
    GlExecutableBindingLocation, GlFamilyApi as _, GlProgramDescriptor, GlProgramReflection,
    GlShaderResourceKind, GlShaderSource, GlShaderStage, GlTextureTarget, SamplerId, TextureId,
    validate_texture_unit, validate_uniform_buffer_binding, validate_uniform_range,
};

/// Sampler uniform types supported by the shared desktop and ES binding ABI.
const SAMPLER_TYPES: &[u32] = &[
    0x8B5E, // SAMPLER_2D
    0x8B5F, // SAMPLER_3D
    0x8B60, // SAMPLER_CUBE
    0x8B62, // SAMPLER_2D_SHADOW
    0x8DC1, // SAMPLER_2D_ARRAY (GLSL 1.30 / ES 3.00 enum)
    0x8B5D, 0x8B61, 0x8B63, 0x8B64, 0x8DC0, 0x8DC2, 0x8DC3, 0x8DC9, 0x8DCD, 0x8DCE, 0x8DD0, 0x8DD1,
    0x8DD5, 0x8DD6, 0x8DD8, 0x900C, 0x900D, 0x900E, 0x900F, 0x9108, 0x9109, 0x910A, 0x910B, 0x910C,
    0x910D, 0x8DC5, // SAMPLER_CUBE_SHADOW
    0x8DC4, // SAMPLER_2D_ARRAY_SHADOW
    0x8DCA, 0x8DCB, 0x8DCC, 0x8DCF, // INT_SAMPLER_2D, 3D, CUBE, 2D_ARRAY
    0x8DD2, 0x8DD3, 0x8DD4, 0x8DD7, // UNSIGNED_INT_SAMPLER_2D, 3D, CUBE, 2D_ARRAY
];

/// Column count of an active vertex-input type, if it maps to the common
/// vertex-format vocabulary. Matrices expand to several locations and are
/// rejected instead of being silently collapsed.
fn vertex_input_columns(type_: u32) -> Option<u8> {
    Some(match type_ {
        0x1404..=0x1406 => 1,
        0x8B50 | 0x8B53 | 0x8DC6 => 2,
        0x8B51 | 0x8B54 | 0x8DC7 => 3,
        0x8B52 | 0x8B55 | 0x8DC8 => 4,
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
impl NativeGlProvider {
    pub(super) fn delete_shaders(&self, shaders: &[glow::NativeShader]) {
        use glow::HasContext as _;
        for shader in shaders {
            // SAFETY: current-context contract; these handles were created on
            // it during this transaction.
            unsafe { self.gl.delete_shader(*shader) };
        }
    }

    /// Compiles one already-lowered source, keeping the complete driver log.
    pub(super) fn compile_shader_source(
        &self,
        op: &'static str,
        source: &GlShaderSource,
    ) -> Result<glow::NativeShader, GlError> {
        use glow::HasContext as _;
        let kind = match source.stage {
            GlShaderStage::Vertex => glow::VERTEX_SHADER,
            GlShaderStage::Fragment => glow::FRAGMENT_SHADER,
            GlShaderStage::Compute => glow::COMPUTE_SHADER,
        };
        // SAFETY: current-context contract; a failed compile leaves no object.
        let raw = unsafe { self.gl.create_shader(kind) }.map_err(|message| GlError::Driver {
            operation: op,
            message,
        })?;
        // Naga names the EXT spelling on all ESSL versions. ES 3.2 has
        // cube arrays in the language core, while some ES 3.x drivers expose
        // only the equivalent OES spelling.
        let mut shader_text = source.text.clone();
        let directive = "#extension GL_EXT_texture_cube_map_array : require\n";
        if shader_text.contains(directive) {
            if let crate::backend::gl::api::GlFamilyProfile::Embedded { major, minor } =
                self.profile()
            {
                let names = self.discovery.extensions();
                let has_ext = names
                    .raw_reported_names()
                    .any(|name| name == "GL_EXT_texture_cube_map_array");
                let has_oes = names
                    .raw_reported_names()
                    .any(|name| name == "GL_OES_texture_cube_map_array");
                if !has_ext && (major > 3 || (major == 3 && minor >= 2)) {
                    shader_text = shader_text.replace(directive, "");
                    shader_text = shader_text
                        .replacen("#version 300 es", "#version 320 es", 1)
                        .replacen("#version 310 es", "#version 320 es", 1);
                } else if !has_ext && has_oes {
                    shader_text = shader_text.replace(
                        directive,
                        "#extension GL_OES_texture_cube_map_array : require\n",
                    );
                } else if !has_ext {
                    // SAFETY: this unpublished shader belongs to the current context.
                    unsafe { self.gl.delete_shader(raw) };
                    return Err(GlError::Unsupported {
                        operation: op,
                        reason: "cube array shader support unavailable",
                    });
                }
            }
        }
        // SAFETY: see above.
        let compiled = unsafe {
            self.gl.shader_source(raw, &shader_text);
            self.gl.compile_shader(raw);
            let compiled = self.gl.get_shader_compile_status(raw);
            if compiled {
                Ok(())
            } else {
                Err(GlError::Shader {
                    stage: shader_stage_name(source.stage),
                    log: self.gl.get_shader_info_log(raw),
                })
            }
        }
        .and_then(|()| self.driver_error(op));
        if let Err(error) = compiled {
            // SAFETY: current-context contract.
            unsafe { self.gl.delete_shader(raw) };
            return Err(error);
        }
        Ok(raw)
    }

    /// Attaches, links, reflects, and validates one program inside an
    /// explicit `useProgram` scope that leaves no program selected.
    pub(super) fn link_and_reflect(
        &mut self,
        op: &'static str,
        descriptor: &GlProgramDescriptor,
        shaders: &[glow::NativeShader],
    ) -> Result<(glow::NativeProgram, GlProgramReflection), GlError> {
        use glow::HasContext as _;
        // SAFETY: current-context contract; the program is either returned
        // into the caller's table or deleted on every error path below.
        unsafe {
            let program = self
                .gl
                .create_program()
                .map_err(|message| GlError::Driver {
                    operation: op,
                    message,
                })?;
            let linked = 'link: {
                if self.cache_retrievable_hint {
                    self.gl.program_binary_retrievable_hint(program, true);
                }
                for shader in shaders {
                    self.gl.attach_shader(program, *shader);
                }
                self.gl.link_program(program);
                if !self.gl.get_program_link_status(program) {
                    break 'link Err(GlError::Program {
                        operation: op,
                        log: self.gl.get_program_info_log(program),
                    });
                }
                break 'link self.driver_error(op);
            };
            if let Err(error) = linked {
                self.gl.delete_program(program);
                return Err(error);
            }
            // Sampler-unit assignment needs the program current: `uniform1i`
            // targets the bound program, so reflection runs inside an explicit
            // bind scope and leaves no program selected afterwards.  The
            // provider's record of the driver's current program has to follow it
            // out of the scope -- a pipeline installed before this link is no
            // longer the driver's current program, and a record that still named
            // it would skip the selection a draw needs.
            self.gl.use_program(Some(program));
            let reflection = self.reflect_and_assign(op, program, descriptor);
            self.gl.use_program(None);
            self.current_program = None;
            self.context_state.program_unbound();
            match reflection {
                Ok(reflection) => match self.driver_error(op) {
                    Ok(()) => Ok((program, reflection)),
                    Err(error) => {
                        self.gl.delete_program(program);
                        Err(error)
                    }
                },
                Err(error) => {
                    self.gl.delete_program(program);
                    Err(error)
                }
            }
        }
    }

    /// Reflects the linked program against the frozen logical layout and
    /// applies the deterministic sampler-unit assignment.
    ///
    /// Fragment-output facts stay empty across the shared contract; draw
    /// buffers select outputs at pass time.
    ///
    /// # Safety
    ///
    /// Current-context contract; `program` must be the current program.
    pub(super) unsafe fn reflect_and_assign(
        &self,
        op: &'static str,
        program: glow::NativeProgram,
        descriptor: &GlProgramDescriptor,
    ) -> Result<GlProgramReflection, GlError> {
        use glow::HasContext as _;
        // SAFETY: current-context contract; `program` is the current program
        // and every statement below is a reflection query or sampler-unit
        // assignment against it.
        unsafe {
            let attribute_count = self.gl.get_active_attributes(program);
            let mut vertex_inputs = Vec::new();
            for index in 0..attribute_count {
                let Some(info) = self.gl.get_active_attribute(program, index) else {
                    return Err(GlError::Driver {
                        operation: op,
                        message: "active attribute was not reported".into(),
                    });
                };
                // GLSL reflection includes built-ins such as `gl_VertexID`;
                // they are supplied by the draw command, not Core vertex input.
                if info.name.starts_with("gl_") {
                    continue;
                }
                let columns = vertex_input_columns(info.atype).ok_or(GlError::Unsupported {
                    operation: op,
                    reason: "vertex input type is outside the common attribute vocabulary",
                })?;
                let location =
                    self.gl
                        .get_attrib_location(program, &info.name)
                        .ok_or(GlError::Driver {
                            operation: op,
                            message: format!(
                                "active attribute {:?} has no assigned location",
                                info.name
                            ),
                        })?;
                vertex_inputs.push(crate::backend::gl::api::GlVertexInputReflection {
                    name: info.name,
                    location,
                    columns,
                });
            }

            let block_count = u32::try_from(
                self.gl
                    .get_program_parameter_i32(program, glow::ACTIVE_UNIFORM_BLOCKS),
            )
            .map_err(|_| GlError::Driver {
                operation: op,
                message: "uniform block count was negative".into(),
            })?;
            let mut blocks: Vec<(u32, String)> = Vec::new();
            for index in 0..block_count {
                blocks.push((index, self.gl.get_active_uniform_block_name(program, index)));
            }
            let uniform_count = self.gl.get_active_uniforms(program);
            let mut samplers: Vec<(String, u32, u32)> = Vec::new();
            let mut images: Vec<(String, u32, u32)> = Vec::new();
            for index in 0..uniform_count {
                let Some(info) = self.gl.get_active_uniform(program, index) else {
                    return Err(GlError::Driver {
                        operation: op,
                        message: "active uniform was not reported".into(),
                    });
                };
                if (0x904c..=0x906d).contains(&info.utype) {
                    let size = u32::try_from(info.size)
                        .ok()
                        .filter(|size| *size > 0)
                        .unwrap_or(1);
                    images.push((base_uniform_name(&info.name).to_owned(), size, info.utype));
                }
                if !is_sampler_type(info.utype) {
                    continue;
                }
                let size = u32::try_from(info.size)
                    .ok()
                    .filter(|size| *size > 0)
                    .unwrap_or(1);
                samplers.push((base_uniform_name(&info.name).to_owned(), size, info.utype));
            }

            // Deterministic assignment: uniform blocks take their GL block index;
            // sampler groups receive consecutive units in layout order.
            let mut assignments = Vec::new();
            let mut next_unit = 0u32;
            let max_units = self.discovery.limits().max_combined_texture_image_units;
            // Image units are their own namespace, so this counter starts at
            // zero independently of the sampler one above.
            let max_image_units = self.discovery.limits().max_image_units;
            for (uniform_name, active_size, uniform_type) in &samplers {
                let matches: Vec<_> = descriptor
                    .layout
                    .bindings
                    .iter()
                    .filter(|binding| {
                        matches!(
                            binding.kind,
                            GlShaderResourceKind::Sampler
                                | GlShaderResourceKind::Texture
                                | GlShaderResourceKind::CombinedTextureSampler
                        ) && binding.names.iter().any(|name| name == uniform_name)
                    })
                    .collect();
                if matches.is_empty()
                    || matches.iter().any(|binding| {
                        (binding.fixed_array
                            && binding.element.is_none()
                            && binding.array_count < *active_size)
                            || (binding.element.is_some() && *active_size != 1)
                            || binding
                                .sampled_type
                                .is_some_and(|expected| !expected.matches_uniform(*uniform_type))
                    })
                {
                    return Err(Self::validation(
                        op,
                        "active sampler does not match the pipeline layout",
                    ));
                }
                let texture_count = matches
                    .iter()
                    .filter(|binding| matches!(binding.kind, GlShaderResourceKind::Texture))
                    .count();
                let sampler_count = matches
                    .iter()
                    .filter(|binding| matches!(binding.kind, GlShaderResourceKind::Sampler))
                    .count();
                if texture_count > 1 || sampler_count > 1 {
                    return Err(Self::validation(
                        op,
                        "sampler uniform has ambiguous logical bindings",
                    ));
                }
                let unit = next_unit;
                next_unit = unit
                    .checked_add(*active_size)
                    .ok_or(GlError::OutOfMemory { operation: op })?;
                if next_unit > max_units {
                    return Err(Self::validation(
                        op,
                        "sampler assignments exceed the discovered unit count",
                    ));
                }
                for element in 0..*active_size {
                    let element_name = if *active_size == 1 {
                        uniform_name.clone()
                    } else {
                        format!("{uniform_name}[{element}]")
                    };
                    let Some(location) = self.gl.get_uniform_location(program, &element_name)
                    else {
                        return Err(GlError::Driver {
                            operation: op,
                            message: "sampler uniform has no location".into(),
                        });
                    };
                    self.gl
                        .uniform_1_i32(Some(&location), (unit + element) as i32);
                }
                for binding in matches {
                    for element in 0..*active_size {
                        assignments.push(GlExecutableBindingAssignment {
                            element: if binding.fixed_array && binding.element.is_none() {
                                Some(element)
                            } else {
                                binding.element
                            },
                            logical: binding.location,
                            executable: GlExecutableBindingLocation::TextureUnit(unit + element),
                        });
                    }
                }
            }
            samplers.clear();
            let mut next_uniform_point = 0u32;
            let mut next_storage_point = 0u32;
            // WGSL lowering assigns image units explicitly. Direct GLSL image
            // uniforms can instead be assigned independently after linking.
            let mut used_image_units: std::collections::BTreeSet<u32> = descriptor
                .layout
                .bindings
                .iter()
                .filter_map(|binding| binding.image_unit)
                .collect();
            let mut direct_image_units = std::collections::BTreeMap::new();
            for binding in &descriptor.layout.bindings {
                match binding.kind {
                    GlShaderResourceKind::UniformBuffer => {
                        let direct_array = binding.fixed_array && binding.element.is_none();
                        let mut elements: Vec<u32> = if direct_array {
                            blocks
                                .iter()
                                .filter_map(|(_, active)| {
                                    binding.names.iter().find_map(|base| {
                                        active
                                            .strip_prefix(base)?
                                            .strip_prefix('[')?
                                            .strip_suffix(']')?
                                            .parse::<u32>()
                                            .ok()
                                            .filter(|index| *index < binding.array_count)
                                    })
                                })
                                .collect()
                        } else {
                            vec![0]
                        };
                        elements.sort_unstable();
                        elements.dedup();
                        for element in elements {
                            let names: Vec<_> = binding
                                .names
                                .iter()
                                .map(|name| {
                                    if direct_array {
                                        format!("{name}[{element}]")
                                    } else {
                                        name.clone()
                                    }
                                })
                                .collect();
                            let indices: Vec<_> = names
                                .iter()
                                .filter_map(|name| {
                                    blocks
                                        .iter()
                                        .position(|(_, found)| found == name)
                                        .map(|position| blocks.remove(position).0)
                                })
                                .collect();
                            if indices.is_empty() {
                                if direct_array {
                                    continue;
                                }
                                return Err(Self::validation(
                                    op,
                                    "declared uniform block is missing from the linked program",
                                ));
                            }
                            let point = next_uniform_point;
                            next_uniform_point = next_uniform_point
                                .checked_add(1)
                                .ok_or(GlError::OutOfMemory { operation: op })?;
                            if point >= self.discovery.limits().max_uniform_buffer_bindings {
                                return Err(Self::validation(
                                    op,
                                    "uniform buffer assignments exceed device limit",
                                ));
                            }
                            for index in indices {
                                self.gl.uniform_block_binding(program, index, point);
                            }
                            assignments.push(GlExecutableBindingAssignment {
                                element: if direct_array {
                                    Some(element)
                                } else {
                                    binding.element
                                },
                                logical: binding.location,
                                executable: GlExecutableBindingLocation::UniformBlock(point),
                            });
                        }
                    }
                    GlShaderResourceKind::Sampler
                    | GlShaderResourceKind::Texture
                    | GlShaderResourceKind::CombinedTextureSampler => {}
                    GlShaderResourceKind::StorageBuffer(_) => {
                        let direct_array = binding.fixed_array && binding.element.is_none();
                        let mut elements: Vec<u32> = if direct_array {
                            self.active_storage_blocks(op, program)?
                                .iter()
                                .filter_map(|(_, active)| {
                                    binding.names.iter().find_map(|base| {
                                        active
                                            .strip_prefix(base)?
                                            .strip_prefix('[')?
                                            .strip_suffix(']')?
                                            .parse::<u32>()
                                            .ok()
                                            .filter(|index| *index < binding.array_count)
                                    })
                                })
                                .collect()
                        } else {
                            vec![0]
                        };
                        elements.sort_unstable();
                        elements.dedup();
                        for element in elements {
                            let mut indices = Vec::new();
                            for name in &binding.names {
                                let name = if direct_array {
                                    format!("{name}[{element}]")
                                } else {
                                    name.clone()
                                };
                                if let Some(index) =
                                    self.gl.get_shader_storage_block_index(program, &name)
                                {
                                    indices.push(index);
                                }
                            }
                            if indices.is_empty() {
                                if direct_array {
                                    continue;
                                }
                                return Err(Self::validation(
                                    op,
                                    "declared storage block is missing from the linked program",
                                ));
                            }
                            let embedded = matches!(
                                self.profile(),
                                crate::backend::gl::api::GlFamilyProfile::Embedded { .. }
                            );
                            let mut point = next_storage_point;
                            if embedded {
                                let query = self.program_resource.ok_or_else(|| {
                                    Self::validation(op, "glGetProgramResourceiv was not loaded")
                                })?;
                                let mut actual = None;
                                for &index in &indices {
                                    let mut value = -1;
                                    let mut length = 0;
                                    // SAFETY: each index is an active storage block of this live linked program.
                                    query(
                                        program.0.get(),
                                        glow::SHADER_STORAGE_BLOCK,
                                        index,
                                        1,
                                        &glow::BUFFER_BINDING,
                                        1,
                                        &mut length,
                                        &mut value,
                                    );
                                    if length != 1
                                        || value < 0
                                        || actual.is_some_and(|prior| prior != value)
                                    {
                                        return Err(Self::validation(
                                            op,
                                            "storage block bindings disagree",
                                        ));
                                    }
                                    actual = Some(value);
                                }
                                point = actual.ok_or_else(|| {
                                    Self::validation(op, "missing storage block binding")
                                })? as u32;
                            }
                            if point >= self.discovery.limits().max_storage_buffer_bindings {
                                return Err(Self::validation(
                                    op,
                                    "storage buffer assignments exceed device limit",
                                ));
                            }
                            next_storage_point = next_storage_point
                                .checked_add(1)
                                .ok_or(GlError::OutOfMemory { operation: op })?;
                            if !embedded {
                                for index in indices {
                                    self.gl.shader_storage_block_binding(program, index, point);
                                }
                            }
                            assignments.push(GlExecutableBindingAssignment {
                                element: if direct_array {
                                    Some(element)
                                } else {
                                    binding.element
                                },
                                logical: binding.location,
                                executable: GlExecutableBindingLocation::StorageBlock(point),
                            });
                        }
                    }
                    GlShaderResourceKind::StorageImage(_) => {
                        for name in &binding.names {
                            if let Some((_, _, ty)) =
                                images.iter().find(|(active, _, _)| active == name)
                            {
                                if binding
                                    .image_type
                                    .is_some_and(|expected| !image_uniform_matches(expected, *ty))
                                {
                                    return Err(Self::validation(
                                        op,
                                        "storage image uniform type does not match layout",
                                    ));
                                }
                            }
                        }
                        if let Some(unit) = binding.image_unit {
                            if unit >= max_image_units {
                                return Err(Self::validation(
                                    op,
                                    "storage image assignments exceed the discovered image unit count",
                                ));
                            }
                            assignments.push(GlExecutableBindingAssignment {
                                element: binding.element,
                                logical: binding.location,
                                executable: GlExecutableBindingLocation::ImageUnit(unit),
                            });
                        } else {
                            // ES image units are fixed by shader layout qualifiers;
                            // desktop permits assigning them after linking.
                            for name in &binding.names {
                                let Some((_, active_size, _)) =
                                    images.iter().find(|(active, _, _)| active == name)
                                else {
                                    continue;
                                };
                                if *active_size > binding.array_count {
                                    return Err(Self::validation(
                                        op,
                                        "active image array exceeds the layout",
                                    ));
                                }
                                for element in 0..*active_size {
                                    let name = if binding.array_count > 1 {
                                        format!("{name}[{element}]")
                                    } else {
                                        name.clone()
                                    };
                                    let location =
                                        self.gl.get_uniform_location(program, &name).ok_or_else(
                                            || Self::validation(op, "active image has no location"),
                                        )?;
                                    let logical_element =
                                        if binding.fixed_array && binding.element.is_none() {
                                            Some(element)
                                        } else {
                                            binding.element
                                        };
                                    let key = (binding.location, logical_element);
                                    let embedded = matches!(
                                        self.profile(),
                                        crate::backend::gl::api::GlFamilyProfile::Embedded { .. }
                                    );
                                    let unit = if embedded {
                                        let mut actual = [-1];
                                        self.gl.get_uniform_i32(program, &location, &mut actual);
                                        let unit = u32::try_from(actual[0]).map_err(|_| {
                                            Self::validation(op, "negative image unit")
                                        })?;
                                        if unit >= max_image_units {
                                            return Err(Self::validation(
                                                op,
                                                "storage image assignment exceeds the discovered image unit count",
                                            ));
                                        }
                                        unit
                                    } else if let Some(&unit) = direct_image_units.get(&key) {
                                        unit
                                    } else {
                                        let unit = (0..max_image_units).find(|unit| !used_image_units.contains(unit)).ok_or_else(|| Self::validation(op, "storage image assignments exceed the discovered image unit count"))?;
                                        used_image_units.insert(unit);
                                        direct_image_units.insert(key, unit);
                                        unit
                                    };
                                    if !embedded {
                                        self.gl.uniform_1_i32(Some(&location), unit as i32);
                                    }
                                    assignments.push(GlExecutableBindingAssignment {
                                        element: logical_element,
                                        logical: binding.location,
                                        executable: GlExecutableBindingLocation::ImageUnit(unit),
                                    });
                                }
                            }
                        }
                    }
                }
            }
            if !blocks.is_empty() {
                return Err(Self::validation(
                    op,
                    "linked program declares a uniform block absent from the layout",
                ));
            }
            if !samplers.is_empty() {
                return Err(Self::validation(
                    op,
                    "linked program declares a sampler absent from the layout",
                ));
            }
            let reflection = GlProgramReflection {
                vertex_inputs,
                fragment_outputs: Vec::new(),
                assignments,
            };
            reflection
                .validate_against(&descriptor.layout)
                .map_err(|_| {
                    Self::validation(op, "linked reflection does not satisfy the frozen layout")
                })?;
            Ok(reflection)
        }
    }
}

fn image_uniform_matches(
    expected: (
        crate::backend::gl::api::GlSampledDimension,
        crate::backend::gl::api::GlSampledKind,
    ),
    ty: u32,
) -> bool {
    use crate::backend::gl::api::{GlSampledDimension as D, GlSampledKind as K};
    matches!(
        (expected, ty),
        ((D::D2, K::Float), glow::IMAGE_2D)
            | ((D::D2Array, K::Float), glow::IMAGE_2D_ARRAY)
            | ((D::D3, K::Float), glow::IMAGE_3D)
            | ((D::D2, K::Sint), glow::INT_IMAGE_2D)
            | ((D::D2Array, K::Sint), glow::INT_IMAGE_2D_ARRAY)
            | ((D::D3, K::Sint), glow::INT_IMAGE_3D)
            | ((D::D2, K::Uint), glow::UNSIGNED_INT_IMAGE_2D)
            | ((D::D2Array, K::Uint), glow::UNSIGNED_INT_IMAGE_2D_ARRAY)
            | ((D::D3, K::Uint), glow::UNSIGNED_INT_IMAGE_3D)
    )
}

const fn shader_stage_name(stage: GlShaderStage) -> &'static str {
    match stage {
        GlShaderStage::Vertex => "vertex",
        GlShaderStage::Fragment => "fragment",
        GlShaderStage::Compute => "compute",
    }
}

impl GlBindingApi for NativeGlProvider {
    fn active_texture(&mut self, unit: u32) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "active-texture";
        self.assert_ready(OP)?;
        validate_texture_unit(unit, self.binding_limits())
            .map_err(|error| Self::validation(OP, error.message()))?;
        // SAFETY: current-context contract; the unit was bounds-checked.
        unsafe { self.gl.active_texture(glow::TEXTURE0 + unit) };
        self.driver_error(OP)
    }

    fn bind_texture(
        &mut self,
        unit: u32,
        target: GlTextureTarget,
        texture: Option<TextureId>,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "bind-texture";
        self.assert_ready(OP)?;
        validate_texture_unit(unit, self.binding_limits())
            .map_err(|error| Self::validation(OP, error.message()))?;
        let raw = texture
            .map(|texture| {
                let entry = self.texture(OP, texture)?;
                Ok::<_, GlError>(entry.0)
            })
            .transpose()?;
        // SAFETY: current-context contract; liveness and unit bounds checked.
        unsafe {
            self.gl.active_texture(glow::TEXTURE0 + unit);
            self.gl.bind_texture(texture_target(target), raw);
        }
        self.driver_error(OP)?;
        let class = match texture {
            Some(id) => super::sampler_border::numeric_class(self.texture(OP, id)?.1.format),
            None => super::sampler_border::NumericClass::Float,
        };
        self.texture_unit_classes.insert(unit, class);
        if let Some(sampler) = self.texture_unit_samplers.get(&unit).copied() {
            self.bind_sampler_variant(unit, Some(sampler), class)?;
        }
        Ok(())
    }

    fn bind_sampler(&mut self, unit: u32, sampler: Option<SamplerId>) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "bind-sampler";
        self.assert_ready(OP)?;
        validate_texture_unit(unit, self.binding_limits())
            .map_err(|error| Self::validation(OP, error.message()))?;
        let class = self
            .texture_unit_classes
            .get(&unit)
            .copied()
            .unwrap_or(super::sampler_border::NumericClass::Float);
        self.bind_sampler_variant(unit, sampler, class)?;
        if let Some(id) = sampler {
            self.texture_unit_samplers.insert(unit, id);
        } else {
            self.texture_unit_samplers.remove(&unit);
        }
        Ok(())
    }

    fn bind_uniform_buffer(
        &mut self,
        index: u32,
        buffer: Option<BufferId>,
        offset: u32,
        size: u32,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "bind-uniform-buffer";
        self.assert_ready(OP)?;
        let limits = self.binding_limits();
        validate_uniform_buffer_binding(index, buffer, offset, size, limits)
            .map_err(|error| Self::validation(OP, error.message()))?;
        let Some(buffer) = buffer else {
            // SAFETY: current-context contract; the unbind shape was checked.
            unsafe {
                self.gl.bind_buffer_base(glow::UNIFORM_BUFFER, index, None);
            }
            return self.driver_error(OP);
        };
        let (name, desc) = {
            let entry = self.buffer(OP, buffer)?;
            if !entry.1.usage.contains(GlBufferUsage::UNIFORM) {
                return Err(Self::validation(OP, "buffer lacks uniform usage"));
            }
            entry
        };
        validate_uniform_range(offset, size, desc.size)
            .map_err(|error| Self::validation(OP, error.message()))?;
        let resolved = if size == 0 {
            desc.size - u64::from(offset)
        } else {
            u64::from(size)
        };
        let (offset, resolved) = {
            let offset = i32::try_from(offset)
                .map_err(|_| Self::validation(OP, "offset exceeds GLintptr"))?;
            let resolved = i32::try_from(resolved)
                .map_err(|_| Self::validation(OP, "range exceeds GLintptr"))?;
            (offset, resolved)
        };
        // SAFETY: current-context contract; usage, alignment, and range were
        // validated against the live allocation.
        unsafe {
            self.gl
                .bind_buffer_range(glow::UNIFORM_BUFFER, index, Some(name), offset, resolved);
        }
        self.driver_error(OP)
    }
}

impl NativeGlProvider {
    /// The discovered limits one binding validation needs.
    pub(super) fn binding_limits(&self) -> GlBindingLimits {
        let limits = self.discovery.limits();
        GlBindingLimits {
            max_texture_units: limits.max_combined_texture_image_units,
            max_uniform_buffer_bindings: limits.max_uniform_buffer_bindings,
            uniform_buffer_offset_alignment: limits.uniform_buffer_offset_alignment,
        }
    }
}

const fn texture_target(target: GlTextureTarget) -> u32 {
    match target {
        GlTextureTarget::D1 => glow::TEXTURE_1D,
        GlTextureTarget::D2 => glow::TEXTURE_2D,
        GlTextureTarget::D2Multisample => glow::TEXTURE_2D_MULTISAMPLE,
        GlTextureTarget::Cube => glow::TEXTURE_CUBE_MAP,
        GlTextureTarget::CubeArray => glow::TEXTURE_CUBE_MAP_ARRAY,
        GlTextureTarget::D3 => glow::TEXTURE_3D,
        GlTextureTarget::D2Array => glow::TEXTURE_2D_ARRAY,
        GlTextureTarget::D2MultisampleArray => glow::TEXTURE_2D_MULTISAMPLE_ARRAY,
    }
}
