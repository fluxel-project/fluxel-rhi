//! Complete private bindings for unpopulated sampled-resource array entries.
//!
//! Names are scoped to the provider's current context and reused until that
//! context is destroyed. No dummy content is observable by a valid shader:
//! partially bound entries must not be accessed by the application.

use super::provider::NativeGlProvider;
use crate::backend::api::{
    GlError, GlExtent3d, GlFamilyApi as _, GlFamilyProfile, GlFormat, GlResourceApi as _,
    GlStorageImageAccess, GlStorageImageApi as _, GlStorageImageBinding, GlTextureDesc,
    GlTextureDimension, GlTextureTarget, GlTextureUsage,
};
use crate::backend::state::{ScopedRawAccess, StateEvent};
use fluxel_rhi_core::api::TextureSampleType;
use glow::HasContext as _;

impl NativeGlProvider {
    pub(crate) fn bind_dummy_storage_buffer(
        &mut self,
        program: crate::backend::api::ProgramId,
        point: u32,
    ) -> Result<(), GlError> {
        use crate::backend::api::GlExecutableBindingLocation;
        const OP: &str = "bind dummy storage buffer";
        self.assert_ready(OP)?;
        if point >= self.discovery.limits().max_storage_buffer_bindings {
            return Err(Self::validation(
                OP,
                "storage binding exceeds the discovered count",
            ));
        }
        let cached = self
            .program(OP, program)?
            .dummy_storage_block_sizes
            .get(&point)
            .copied();
        let size = match cached {
            Some(Some(size)) => size,
            Some(None) => {
                return Err(Self::validation(
                    OP,
                    "no active storage block at binding point",
                ));
            }
            None => {
                // Reflection is immutable for a linked program. Do this cold
                // once, then retain both successful and absent results.
                let query = self.program_resource.ok_or(GlError::Unsupported {
                    operation: OP,
                    reason: "program resource query entry unavailable",
                })?;
                let (raw, names) = {
                    let program = self.program(OP, program)?;
                    let mut names = Vec::new();
                    for assignment in &program.reflection.assignments {
                        if assignment.executable != GlExecutableBindingLocation::StorageBlock(point)
                        {
                            continue;
                        }
                        for binding in &program.descriptor.layout.bindings {
                            if binding.location != assignment.logical {
                                continue;
                            }
                            if binding.element == assignment.element {
                                names.extend(binding.names.iter().cloned());
                            } else if binding.element.is_none() && binding.fixed_array {
                                if let Some(element) = assignment
                                    .element
                                    .filter(|element| *element < binding.array_count)
                                {
                                    names.extend(
                                        binding
                                            .names
                                            .iter()
                                            .map(|name| format!("{name}[{element}]")),
                                    );
                                }
                            }
                        }
                    }
                    (program.raw, names)
                };
                let size = if names.is_empty() {
                    None
                } else {
                    let mut size = 0i32;
                    for name in names {
                        // SAFETY: the program is linked in the current context and the name is owned here.
                        let index = unsafe { self.gl.get_shader_storage_block_index(raw, &name) }
                            .ok_or_else(|| {
                            Self::validation(OP, "linked storage block is missing")
                        })?;
                        let mut actual = -1i32;
                        let mut length = 0;
                        // SAFETY: the linked block belongs to this context; one property and output slot are valid.
                        unsafe {
                            query(
                                raw.0.get(),
                                glow::SHADER_STORAGE_BLOCK,
                                index,
                                1,
                                &glow::BUFFER_DATA_SIZE,
                                1,
                                &mut length,
                                &mut actual,
                            );
                        }
                        self.driver_error(OP)?;
                        if length != 1 || actual < 0 {
                            return Err(Self::validation(OP, "invalid storage block size"));
                        }
                        size = size.max(actual);
                    }
                    Some(size.max(4) as u32)
                };
                self.programs
                    .get_mut(&program)
                    .expect("validated live program")
                    .dummy_storage_block_sizes
                    .insert(point, size);
                size.ok_or_else(|| {
                    Self::validation(OP, "no active storage block at binding point")
                })?
            }
        };
        let buffer = match self.dummy_storage_buffers.get(&(point, size)).copied() {
            Some(buffer) => buffer,
            None => {
                // SAFETY: the temporary name is owned by the current context.
                let buffer =
                    unsafe { self.gl.create_buffer() }.map_err(|message| GlError::Driver {
                        operation: OP,
                        message,
                    })?;
                unsafe {
                    self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
                    self.gl.buffer_data_size(
                        glow::COPY_WRITE_BUFFER,
                        size as i32,
                        glow::DYNAMIC_DRAW,
                    );
                    self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
                }
                if let Err(error) = self.driver_error(OP) {
                    unsafe { self.gl.delete_buffer(buffer) };
                    return Err(error);
                }
                self.dummy_storage_buffers.insert((point, size), buffer);
                buffer
            }
        };
        // SAFETY: each point has a distinct buffer in this context, sized for all active blocks.
        unsafe {
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, point, Some(buffer));
        }
        self.defer_driver_error(OP)
    }

    pub(crate) fn bind_dummy_uniform_buffer(
        &mut self,
        program: crate::backend::api::ProgramId,
        point: u32,
    ) -> Result<(), GlError> {
        const OP: &str = "bind dummy uniform buffer";
        self.assert_ready(OP)?;
        if point >= self.discovery.limits().max_uniform_buffer_bindings {
            return Err(Self::validation(
                OP,
                "uniform binding exceeds the discovered count",
            ));
        }
        let cached = self
            .program(OP, program)?
            .dummy_uniform_block_sizes
            .get(&point)
            .copied();
        let size = match cached {
            Some(Some(size)) => size,
            Some(None) => {
                return Err(Self::validation(
                    OP,
                    "no active uniform block at binding point",
                ));
            }
            None => {
                let raw = self.program(OP, program)?.raw;
                // SAFETY: the program was checked against the current context and is linked.
                let count = unsafe {
                    self.gl
                        .get_program_parameter_i32(raw, glow::ACTIVE_UNIFORM_BLOCKS)
                };
                self.driver_error(OP)?;
                let mut size = 0;
                for index in 0..u32::try_from(count)
                    .map_err(|_| Self::validation(OP, "invalid active block count"))?
                {
                    // SAFETY: index is within the active block range of this linked program.
                    let binding = unsafe {
                        self.gl.get_active_uniform_block_parameter_i32(
                            raw,
                            index,
                            glow::UNIFORM_BLOCK_BINDING,
                        )
                    };
                    if binding == point as i32 {
                        let block_size = unsafe {
                            self.gl.get_active_uniform_block_parameter_i32(
                                raw,
                                index,
                                glow::UNIFORM_BLOCK_DATA_SIZE,
                            )
                        };
                        size = size.max(block_size);
                    }
                }
                self.driver_error(OP)?;
                let size = (size > 0).then_some(size as u32);
                self.programs
                    .get_mut(&program)
                    .expect("validated live program")
                    .dummy_uniform_block_sizes
                    .insert(point, size);
                size.ok_or_else(|| {
                    Self::validation(OP, "no active uniform block at binding point")
                })?
            }
        };
        let buffer = match self.dummy_uniform_buffers.get(&size).copied() {
            Some(buffer) => buffer,
            None => {
                // SAFETY: the temporary name is owned by the current context.
                let buffer =
                    unsafe { self.gl.create_buffer() }.map_err(|message| GlError::Driver {
                        operation: OP,
                        message,
                    })?;
                unsafe {
                    self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
                    self.gl.buffer_data_size(
                        glow::COPY_WRITE_BUFFER,
                        size as i32,
                        glow::STATIC_DRAW,
                    );
                    self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
                }
                if let Err(error) = self.driver_error(OP) {
                    unsafe { self.gl.delete_buffer(buffer) };
                    return Err(error);
                }
                self.dummy_uniform_buffers.insert(size, buffer);
                buffer
            }
        };
        // SAFETY: the buffer belongs to this context and meets every block's size at this point.
        unsafe {
            self.gl
                .bind_buffer_base(glow::UNIFORM_BUFFER, point, Some(buffer));
        }
        self.defer_driver_error(OP)
    }

    pub(crate) fn bind_dummy_storage_image(
        &mut self,
        unit: u32,
        dimension: GlTextureDimension,
        format: GlFormat,
        access: GlStorageImageAccess,
    ) -> Result<(), GlError> {
        const OP: &str = "bind dummy storage image";
        self.assert_ready(OP)?;
        if unit >= self.discovery.limits().max_image_units {
            return Err(Self::validation(
                OP,
                "image unit exceeds the discovered unit count",
            ));
        }
        let layered = match dimension {
            GlTextureDimension::D2 => false,
            GlTextureDimension::D2Array | GlTextureDimension::D3 => true,
            _ => {
                return Err(GlError::Unsupported {
                    operation: OP,
                    reason: "storage image dimension",
                });
            }
        };
        let id = match self
            .dummy_storage_images
            .iter()
            .find(|(u, d, f, _)| *u == unit && *d == dimension && *f == format)
            .map(|(_, _, _, id)| *id)
        {
            Some(id) => id,
            None => {
                let id = self.create_texture_resource(GlTextureDesc {
                    dimension,
                    extent: GlExtent3d {
                        width: 1,
                        height: 1,
                        depth_or_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    format,
                    usage: GlTextureUsage::STORAGE_BINDING,
                })?;
                self.dummy_storage_images
                    .push((unit, dimension, format, id));
                id
            }
        };
        self.bind_storage_image(
            unit,
            GlStorageImageBinding {
                texture: id,
                level: 0,
                sample_count: 1,
                layered,
                layer: if layered { None } else { Some(0) },
                format,
                access,
            },
        )
    }

    pub(crate) fn bind_dummy_sampled_texture(
        &mut self,
        unit: u32,
        target: GlTextureTarget,
        sample: TextureSampleType,
    ) -> Result<(), GlError> {
        const OP: &str = "bind dummy sampled texture";
        self.assert_ready(OP)?;
        if unit >= self.discovery.limits().max_combined_texture_image_units {
            return Err(Self::validation(
                OP,
                "texture unit exceeds the discovered unit count",
            ));
        }
        let (kind, internal, format, ty, class) = match sample {
            TextureSampleType::Float | TextureSampleType::UnfilterableFloat => (
                0,
                glow::RGBA8,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                super::sampler_border::NumericClass::Float,
            ),
            TextureSampleType::Sint => (
                1,
                glow::RGBA8I,
                glow::RGBA_INTEGER,
                glow::BYTE,
                super::sampler_border::NumericClass::Sint,
            ),
            TextureSampleType::Uint => (
                2,
                glow::RGBA8UI,
                glow::RGBA_INTEGER,
                glow::UNSIGNED_BYTE,
                super::sampler_border::NumericClass::Uint,
            ),
            TextureSampleType::Depth => (
                3,
                glow::DEPTH_COMPONENT24,
                glow::DEPTH_COMPONENT,
                glow::UNSIGNED_INT,
                super::sampler_border::NumericClass::Float,
            ),
        };
        let gl_target = match target {
            GlTextureTarget::D1 => glow::TEXTURE_1D,
            GlTextureTarget::D2 => glow::TEXTURE_2D,
            GlTextureTarget::D2Array => glow::TEXTURE_2D_ARRAY,
            GlTextureTarget::D3 => glow::TEXTURE_3D,
            GlTextureTarget::Cube => glow::TEXTURE_CUBE_MAP,
            GlTextureTarget::CubeArray => glow::TEXTURE_CUBE_MAP_ARRAY,
            GlTextureTarget::D2Multisample => glow::TEXTURE_2D_MULTISAMPLE,
            GlTextureTarget::D2MultisampleArray => glow::TEXTURE_2D_MULTISAMPLE_ARRAY,
        };
        let key = (target, kind);
        let existing = self.dummy_textures.get(&key).copied();
        let texture = match existing {
            Some(texture) => texture,
            None => {
                if target == GlTextureTarget::D2MultisampleArray && self.multisample_array.is_none()
                {
                    return Err(GlError::Unsupported {
                        operation: OP,
                        reason: "multisample array storage entry unavailable",
                    });
                }
                // SAFETY: provider owns the current context and the unpublished name.
                let texture =
                    unsafe { self.gl.create_texture() }.map_err(|message| GlError::Driver {
                        operation: OP,
                        message,
                    })?;
                unsafe {
                    self.gl.active_texture(glow::TEXTURE0 + unit);
                    self.gl.bind_texture(gl_target, Some(texture));
                    self.gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
                    match target {
                        GlTextureTarget::D1 => self.gl.tex_image_1d(
                            gl_target,
                            0,
                            internal as i32,
                            1,
                            0,
                            format,
                            ty,
                            glow::PixelUnpackData::Slice(None),
                        ),
                        GlTextureTarget::D2 => self.gl.tex_image_2d(
                            gl_target,
                            0,
                            internal as i32,
                            1,
                            1,
                            0,
                            format,
                            ty,
                            glow::PixelUnpackData::Slice(None),
                        ),
                        GlTextureTarget::Cube => {
                            for face in 0..6 {
                                self.gl.tex_image_2d(
                                    glow::TEXTURE_CUBE_MAP_POSITIVE_X + face,
                                    0,
                                    internal as i32,
                                    1,
                                    1,
                                    0,
                                    format,
                                    ty,
                                    glow::PixelUnpackData::Slice(None),
                                );
                            }
                        }
                        GlTextureTarget::D2Array | GlTextureTarget::D3 => self.gl.tex_image_3d(
                            gl_target,
                            0,
                            internal as i32,
                            1,
                            1,
                            1,
                            0,
                            format,
                            ty,
                            glow::PixelUnpackData::Slice(None),
                        ),
                        GlTextureTarget::CubeArray => self.gl.tex_image_3d(
                            gl_target,
                            0,
                            internal as i32,
                            1,
                            1,
                            6,
                            0,
                            format,
                            ty,
                            glow::PixelUnpackData::Slice(None),
                        ),
                        GlTextureTarget::D2Multisample => {
                            match self.discovery.context().profile() {
                                GlFamilyProfile::Embedded { .. } => self
                                    .gl
                                    .tex_storage_2d_multisample(gl_target, 1, internal, 1, 1, true),
                                _ => self.gl.tex_image_2d_multisample(
                                    gl_target,
                                    1,
                                    internal as i32,
                                    1,
                                    1,
                                    true,
                                ),
                            }
                        }
                        GlTextureTarget::D2MultisampleArray => {
                            (self.multisample_array.unwrap())(gl_target, 1, internal, 1, 1, 1, 1)
                        }
                    }
                    if !matches!(
                        target,
                        GlTextureTarget::D2Multisample | GlTextureTarget::D2MultisampleArray
                    ) {
                        self.gl.tex_parameter_i32(
                            gl_target,
                            glow::TEXTURE_MIN_FILTER,
                            glow::NEAREST as i32,
                        );
                        self.gl.tex_parameter_i32(
                            gl_target,
                            glow::TEXTURE_MAG_FILTER,
                            glow::NEAREST as i32,
                        );
                    }
                }
                self.context_state
                    .event(StateEvent::ScopedRawAccess(ScopedRawAccess::all()));
                if let Err(error) = self.driver_error(OP) {
                    unsafe { self.gl.delete_texture(texture) };
                    return Err(error);
                }
                self.dummy_textures.insert(key, texture);
                texture
            }
        };
        // SAFETY: cached name belongs to this context and unit was validated.
        unsafe {
            self.gl.active_texture(glow::TEXTURE0 + unit);
            self.gl.bind_texture(gl_target, Some(texture));
        }
        self.defer_driver_error(OP)?;
        self.texture_unit_classes.insert(unit, class);
        if let Some(sampler) = self.texture_unit_samplers.get(&unit).copied() {
            self.bind_sampler_variant(unit, Some(sampler), class)?;
        }
        Ok(())
    }

    pub(crate) fn bind_dummy_sampler(
        &mut self,
        unit: u32,
        comparison: bool,
    ) -> Result<(), GlError> {
        const OP: &str = "bind dummy sampler";
        self.assert_ready(OP)?;
        if unit >= self.discovery.limits().max_combined_texture_image_units {
            return Err(Self::validation(
                OP,
                "texture unit exceeds the discovered unit count",
            ));
        }
        let sampler = match self.dummy_samplers.get(&comparison).copied() {
            Some(sampler) => sampler,
            None => {
                // SAFETY: provider owns the current context and unpublished name.
                let sampler =
                    unsafe { self.gl.create_sampler() }.map_err(|message| GlError::Driver {
                        operation: OP,
                        message,
                    })?;
                unsafe {
                    self.gl.sampler_parameter_i32(
                        sampler,
                        glow::TEXTURE_MIN_FILTER,
                        glow::NEAREST as i32,
                    );
                    self.gl.sampler_parameter_i32(
                        sampler,
                        glow::TEXTURE_MAG_FILTER,
                        glow::NEAREST as i32,
                    );
                    if comparison {
                        self.gl.sampler_parameter_i32(
                            sampler,
                            glow::TEXTURE_COMPARE_MODE,
                            glow::COMPARE_REF_TO_TEXTURE as i32,
                        );
                        self.gl.sampler_parameter_i32(
                            sampler,
                            glow::TEXTURE_COMPARE_FUNC,
                            glow::LESS as i32,
                        );
                    }
                }
                if let Err(error) = self.driver_error(OP) {
                    unsafe { self.gl.delete_sampler(sampler) };
                    return Err(error);
                }
                self.dummy_samplers.insert(comparison, sampler);
                sampler
            }
        };
        // SAFETY: cached sampler belongs to this context and unit is valid.
        unsafe {
            self.gl.bind_sampler(unit, Some(sampler));
        }
        self.defer_driver_error(OP)?;
        self.texture_unit_samplers.remove(&unit);
        Ok(())
    }
}
