//! Native program linking, reflected resource assignment, and interface validation.
//!
//! Native GL owns compilation and execution. Direct SPIR-V carries numeric
//! resource identities because drivers may discard its diagnostic names.
//! Immutable shader metadata is not an application command stream.

use super::provider::NativeGlProvider;
use crate::backend::api::{
    GlError, GlExecutableBindingAssignment, GlExecutableBindingLocation, GlFamilyApi as _,
    GlProgramDescriptor, GlProgramReflection, GlShaderResourceKind, GlShaderSource,
};
fn base_uniform_name(name: &str) -> &str {
    name.strip_suffix("[0]").unwrap_or(name)
}
fn program_sources(descriptor: &GlProgramDescriptor) -> Vec<&GlShaderSource> {
    match &descriptor.kind {
        crate::backend::api::GlProgramKind::Raster { vertex, fragment } => {
            vec![vertex, fragment]
        }
        crate::backend::api::GlProgramKind::Compute { shader } => vec![shader],
    }
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

    /// Compiles one already-lowered source, retaining the complete driver log.
    pub(super) fn compile_shader_source(
        &self,
        op: &'static str,
        source: &GlShaderSource,
    ) -> Result<glow::NativeShader, GlError> {
        super::shader_compile::compile_shader_source(self, op, source)
    }

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
            let reflection = self
                .driver_error(op)
                .and_then(|()| self.reflect_and_assign(op, program, descriptor));
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

    /// # Safety
    /// `program` is a linked object owned by this current-context provider and
    /// remains current and alive for the complete reflection transaction.
    pub(super) unsafe fn reflect_and_assign(
        &self,
        op: &'static str,
        program: glow::NativeProgram,
        descriptor: &GlProgramDescriptor,
    ) -> Result<GlProgramReflection, GlError> {
        use glow::HasContext as _;
        let projected = if program_sources(descriptor)
            .iter()
            .any(|source| source.spirv.is_some())
        {
            let mut projected = descriptor.clone();
            projected.layout = super::shader_interfaces::spirv_layout(&descriptor.layout)?;
            Some(projected)
        } else {
            None
        };
        let descriptor = projected.as_ref().unwrap_or(descriptor);
        // SAFETY: the caller supplies the exclusive current-context transaction;
        // metadata checks precede each bounded native assignment.
        unsafe {
            let sources = program_sources(descriptor);
            let direct_spirv = sources.iter().any(|source| source.spirv.is_some());
            let opaque_names =
                super::shader_interfaces::opaque_uniforms(self, op, program, &sources)?;
            let opaque: Vec<_> = opaque_names.values().copied().collect();
            let vertex_inputs =
                super::shader_interfaces::vertex_inputs(self, op, program, &sources)?;

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
                let name = self.gl.get_active_uniform_block_name(program, index);
                if direct_spirv {
                    let query = self.program_resource.ok_or_else(|| {
                        Self::validation(
                            op,
                            "GL program-interface query is unavailable for unnamed SPIR-V block",
                        )
                    })?;
                    let (mut count, mut point) = (0, -1);
                    query(
                        program.0.get(),
                        glow::UNIFORM_BLOCK,
                        index,
                        1,
                        [glow::BUFFER_BINDING].as_ptr(),
                        1,
                        &mut count,
                        &mut point,
                    );
                    if count != 1 || point < 0 {
                        return Err(Self::validation(
                            op,
                            "unnamed SPIR-V block has no numeric binding",
                        ));
                    }
                    blocks.push((index, format!("@{point}")));
                } else {
                    blocks.push((index, name));
                }
            }
            let uniform_count = if direct_spirv {
                0
            } else {
                self.gl.get_active_uniforms(program)
            };
            let mut samplers: Vec<(u32, String, u32, u32)> = Vec::new();
            let mut images: Vec<(u32, String, u32, u32)> = Vec::new();
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
                    images.push((
                        index,
                        base_uniform_name(&info.name).to_owned(),
                        size,
                        info.utype,
                    ));
                }
                if !super::shader_interfaces::is_sampler_type(info.utype) {
                    continue;
                }
                let size = u32::try_from(info.size)
                    .ok()
                    .filter(|size| *size > 0)
                    .unwrap_or(1);
                samplers.push((
                    index,
                    base_uniform_name(&info.name).to_owned(),
                    size,
                    info.utype,
                ));
            }

            for (name, uniform) in &opaque_names {
                let entry = (0, name.clone(), uniform.array_count, uniform.uniform_type);
                if uniform.image {
                    images.push(entry);
                } else {
                    samplers.push(entry);
                }
            }
            let mut assignments = Vec::new();
            let mut next_unit = 0u32;
            let max_units = self.discovery.limits().max_combined_texture_image_units;
            let max_image_units = self.discovery.limits().max_image_units;
            let mut next_image_unit = 0_u32;
            for (_, uniform_name, active_size, uniform_type) in &samplers {
                let metadata = opaque_names.get(uniform_name);
                let uniform_location = if let Some(uniform) = metadata {
                    glow::NativeUniformLocation(uniform.location)
                } else {
                    self.gl
                        .get_uniform_location(program, uniform_name)
                        .ok_or_else(|| Self::validation(op, "active sampler has no location"))?
                };
                let direct_binding = metadata.map(|uniform| uniform.binding);
                let array_base = direct_binding
                    .and_then(|number| {
                        super::shader_blocks::at(
                            &descriptor.layout,
                            GlShaderResourceKind::Texture,
                            number,
                        )
                        .ok()
                        .flatten()
                        .map(|(_, offset)| offset)
                    })
                    .unwrap_or(0);
                let numeric_matches = direct_binding.and_then(|number| {
                    let (anchor, array_base) = super::shader_blocks::at(
                        &descriptor.layout,
                        GlShaderResourceKind::Texture,
                        number,
                    )
                    .ok()
                    .flatten()?;
                    if array_base.checked_add(*active_size)? > anchor.array_count {
                        return None;
                    }
                    let mut result = vec![anchor];
                    result.extend(descriptor.layout.bindings.iter().filter(|binding| {
                        binding.pair == Some(anchor.location)
                            || anchor.pair == Some(binding.location)
                    }));
                    Some(result)
                });
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
                let matches = numeric_matches
                    .filter(|matches| !matches.is_empty())
                    .unwrap_or(matches);
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
                    let location = if metadata.is_some() {
                        glow::NativeUniformLocation(uniform_location.0 + element)
                    } else if element == 0 {
                        uniform_location.clone()
                    } else if let Some(location) =
                        self.gl.get_uniform_location(program, &element_name)
                    {
                        location
                    } else {
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
                                Some(array_base + element)
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
            let mut storage_blocks = if self
                .discovery
                .capabilities()
                .supports(crate::backend::api::GlCapability::StorageBuffer)
            {
                self.active_storage_blocks(op, program, direct_spirv)?
            } else {
                Vec::new()
            };
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
                        let numeric_base = if direct_spirv {
                            super::shader_blocks::ordinals(&descriptor.layout)?
                                .into_iter()
                                .find(|(_, entry)| core::ptr::eq(*entry, binding))
                                .map(|(base, _)| base)
                        } else {
                            None
                        };
                        for (element, indices) in
                            super::shader_blocks::block_indices(binding, numeric_base, &mut blocks)?
                        {
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
                                element: if binding.fixed_array && binding.element.is_none() {
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
                        let embedded = !matches!(
                            self.profile(),
                            crate::backend::api::GlFamilyProfile::Desktop { .. }
                        );
                        let numeric_base = if direct_spirv {
                            super::shader_blocks::ordinals(&descriptor.layout)?
                                .into_iter()
                                .find(|(_, entry)| core::ptr::eq(*entry, binding))
                                .map(|(base, _)| base)
                        } else {
                            None
                        };
                        for (element, indices) in super::shader_blocks::block_indices(
                            binding,
                            numeric_base,
                            &mut storage_blocks,
                        )? {
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
                                element: if binding.fixed_array && binding.element.is_none() {
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
                        if direct_spirv {
                            let ordinal = super::shader_blocks::ordinals(&descriptor.layout)?
                                .into_iter()
                                .find(|(_, entry)| core::ptr::eq(*entry, binding))
                                .map(|(base, _)| base)
                                .ok_or_else(|| {
                                    Self::validation(
                                        op,
                                        "storage image is absent from the frozen layout",
                                    )
                                })?;
                            let Some(uniform) = opaque
                                .iter()
                                .find(|uniform| uniform.image && uniform.binding == ordinal)
                            else {
                                continue;
                            };
                            if uniform.array_count > binding.array_count
                                || binding.image_type.is_some_and(|expected| {
                                    !image_uniform_matches(expected, uniform.uniform_type)
                                })
                            {
                                return Err(Self::validation(
                                    op,
                                    "SPIR-V storage image type or array size does not match layout",
                                ));
                            }
                            let unit = next_image_unit;
                            next_image_unit = unit
                                .checked_add(uniform.array_count)
                                .ok_or(GlError::OutOfMemory { operation: op })?;
                            if next_image_unit > max_image_units {
                                return Err(Self::validation(
                                    op,
                                    "storage image assignment exceeds the discovered image unit count",
                                ));
                            }
                            for element in 0..uniform.array_count {
                                let location =
                                    glow::NativeUniformLocation(uniform.location + element);
                                self.gl
                                    .uniform_1_i32(Some(&location), (unit + element) as i32);
                                assignments.push(GlExecutableBindingAssignment {
                                    element: if binding.fixed_array && binding.element.is_none() {
                                        Some(element)
                                    } else {
                                        binding.element
                                    },
                                    logical: binding.location,
                                    executable: GlExecutableBindingLocation::ImageUnit(
                                        unit + element,
                                    ),
                                });
                            }
                            continue;
                        }
                        for name in &binding.names {
                            if let Some((_, _, _, ty)) =
                                images.iter().find(|(_, active, _, _)| active == name)
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
                            for name in &binding.names {
                                let Some((_, _, active_size, _)) =
                                    images.iter().find(|(_, active, _, _)| active == name)
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
                                        crate::backend::api::GlFamilyProfile::Embedded { .. }
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
            if !storage_blocks.is_empty() {
                return Err(Self::validation(
                    op,
                    "linked program declares a storage block absent from the layout",
                ));
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
        crate::backend::api::GlSampledDimension,
        crate::backend::api::GlSampledKind,
    ),
    ty: u32,
) -> bool {
    use crate::backend::api::{GlSampledDimension as D, GlSampledKind as K};
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
