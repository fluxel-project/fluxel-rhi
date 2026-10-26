//! GLES depth/stencil buffer copies using a private compute shader.
//! Each texel is written as one scratch uint and copied to its byte-sized
//! destination without touching surrounding row padding.
use super::super::provider::NativeGlProvider;
use crate::backend::api::{
    GlBufferRange, GlCapability, GlError, GlFamilyApi as _, GlFamilyProfile, GlFormat,
    GlFormatResourceKind, GlPixelFormat, GlPixelLayout, GlTextureAspect, GlTextureDesc,
    GlTextureDimension, GlTextureRegion,
};
use crate::backend::state::{ScopedRawAccess, StateEvent};
use glow::HasContext as _;

/// The six fixed GLES plane-readback programs owned by one native context.
///
/// Transfer geometry belongs to uniforms: changing a source origin, extent,
/// mip, or array layer therefore never reflects, compiles, or links a shader.
/// Context loss simply drops these names; normal teardown deletes them while
/// the exact owner context is current.
pub(crate) struct GlesPlaneReadbackPrograms {
    stencil_2d: GlesPlaneReadbackProgram,
    stencil_2d_array: GlesPlaneReadbackProgram,
    depth16_2d: GlesPlaneReadbackProgram,
    depth16_2d_array: GlesPlaneReadbackProgram,
    float32_2d: GlesPlaneReadbackProgram,
    float32_2d_array: GlesPlaneReadbackProgram,
}

struct GlesPlaneReadbackProgram {
    raw: glow::NativeProgram,
    count: glow::NativeUniformLocation,
    width: glow::NativeUniformLocation,
    height: glow::NativeUniformLocation,
    origin_x: glow::NativeUniformLocation,
    origin_y: glow::NativeUniformLocation,
    base_layer: Option<glow::NativeUniformLocation>,
    mip: glow::NativeUniformLocation,
}

impl GlesPlaneReadbackPrograms {
    unsafe fn create(gl: &glow::Context) -> Result<Self, GlError> {
        const OP: &str = "create GLES plane readback programs";
        // SAFETY: caller owns the current context. Each partial cache is
        // destroyed before an error escapes.
        unsafe {
            let stencil_2d = GlesPlaneReadbackProgram::compile(gl, true, false, 1, OP)?;
            let stencil_2d_array = match GlesPlaneReadbackProgram::compile(gl, true, true, 1, OP) {
                Ok(value) => value,
                Err(error) => {
                    gl.delete_program(stencil_2d.raw);
                    return Err(error);
                }
            };
            let depth16_2d = match GlesPlaneReadbackProgram::compile(gl, false, false, 2, OP) {
                Ok(value) => value,
                Err(error) => {
                    gl.delete_program(stencil_2d.raw);
                    gl.delete_program(stencil_2d_array.raw);
                    return Err(error);
                }
            };
            let depth16_2d_array = match GlesPlaneReadbackProgram::compile(gl, false, true, 2, OP) {
                Ok(value) => value,
                Err(error) => {
                    gl.delete_program(stencil_2d.raw);
                    gl.delete_program(stencil_2d_array.raw);
                    gl.delete_program(depth16_2d.raw);
                    return Err(error);
                }
            };
            let float32_2d = match GlesPlaneReadbackProgram::compile(gl, false, false, 4, OP) {
                Ok(value) => value,
                Err(error) => {
                    gl.delete_program(stencil_2d.raw);
                    gl.delete_program(stencil_2d_array.raw);
                    gl.delete_program(depth16_2d.raw);
                    gl.delete_program(depth16_2d_array.raw);
                    return Err(error);
                }
            };
            let float32_2d_array = match GlesPlaneReadbackProgram::compile(gl, false, true, 4, OP) {
                Ok(value) => value,
                Err(error) => {
                    gl.delete_program(stencil_2d.raw);
                    gl.delete_program(stencil_2d_array.raw);
                    gl.delete_program(depth16_2d.raw);
                    gl.delete_program(depth16_2d_array.raw);
                    gl.delete_program(float32_2d.raw);
                    return Err(error);
                }
            };
            Ok(Self {
                stencil_2d,
                stencil_2d_array,
                depth16_2d,
                depth16_2d_array,
                float32_2d,
                float32_2d_array,
            })
        }
    }

    fn select(&self, stencil: bool, bpp: usize, array: bool) -> &GlesPlaneReadbackProgram {
        match (stencil, bpp, array) {
            (true, _, false) => &self.stencil_2d,
            (true, _, true) => &self.stencil_2d_array,
            (false, 2, false) => &self.depth16_2d,
            (false, 2, true) => &self.depth16_2d_array,
            (false, 4, false) => &self.float32_2d,
            (false, 4, true) => &self.float32_2d_array,
            _ => unreachable!("validated plane representation"),
        }
    }

    pub(crate) unsafe fn destroy(self, gl: &glow::Context) {
        // SAFETY: normal teardown has made the owner context current.
        unsafe {
            for program in [
                self.stencil_2d,
                self.stencil_2d_array,
                self.depth16_2d,
                self.depth16_2d_array,
                self.float32_2d,
                self.float32_2d_array,
            ] {
                gl.delete_program(program.raw);
            }
        }
    }
}

impl GlesPlaneReadbackProgram {
    unsafe fn compile(
        gl: &glow::Context,
        stencil: bool,
        array: bool,
        bpp: usize,
        operation: &'static str,
    ) -> Result<Self, GlError> {
        let sampler = if stencil { "usampler" } else { "sampler" };
        let ty = if array { "2DArray" } else { "2D" };
        let coords = if array {
            "ivec3(int(x+u_origin_x),int(y+u_origin_y),int(layer+u_base_layer))"
        } else {
            "ivec2(int(x+u_origin_x),int(y+u_origin_y))"
        };
        let sample = if stencil { "uint" } else { "float" };
        let convert = if stencil {
            "v"
        } else if bpp == 2 {
            "uint(round(clamp(v,0.0,1.0)*65535.0))"
        } else {
            "floatBitsToUint(v)"
        };
        let source = format!(
            "#version 310 es\nprecision highp int; precision highp float; precision highp {sampler}{ty};\nlayout(local_size_x=64) in; layout(binding=0) uniform {sampler}{ty} image; layout(std430,binding=0) writeonly buffer D {{ uint out_data[]; }}; uniform uint u_count,u_width,u_height,u_origin_x,u_origin_y; uniform int u_mip; {} void main() {{ uint i=gl_GlobalInvocationID.x; if(i>=u_count)return; uint x=i%u_width; uint y=(i/u_width)%u_height; uint layer=i/(u_width*u_height); {sample} v=texelFetch(image,{coords},u_mip).r; out_data[i]={convert}; }}",
            if array {
                "uniform uint u_base_layer;"
            } else {
                ""
            }
        );
        // SAFETY: the current context belongs to the caller; every failed
        // construction path deletes the newly created native names.
        unsafe {
            let shader = gl
                .create_shader(glow::COMPUTE_SHADER)
                .map_err(|_| GlError::OutOfMemory { operation })?;
            gl.shader_source(shader, &source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                let message = gl.get_shader_info_log(shader);
                gl.delete_shader(shader);
                return Err(GlError::Driver { operation, message });
            }
            let program = match gl.create_program() {
                Ok(value) => value,
                Err(_) => {
                    gl.delete_shader(shader);
                    return Err(GlError::OutOfMemory { operation });
                }
            };
            gl.attach_shader(program, shader);
            gl.link_program(program);
            gl.delete_shader(shader);
            if !gl.get_program_link_status(program) {
                let message = gl.get_program_info_log(program);
                gl.delete_program(program);
                return Err(GlError::Driver { operation, message });
            }
            let required = |name| {
                gl.get_uniform_location(program, name)
                    .ok_or_else(|| GlError::Driver {
                        operation,
                        message: format!("linked GLES plane readback program omitted {name}"),
                    })
            };
            let result = (|| {
                Ok(Self {
                    raw: program,
                    count: required("u_count")?,
                    width: required("u_width")?,
                    height: required("u_height")?,
                    origin_x: required("u_origin_x")?,
                    origin_y: required("u_origin_y")?,
                    base_layer: if array {
                        Some(required("u_base_layer")?)
                    } else {
                        None
                    },
                    mip: required("u_mip")?,
                })
            })();
            if result.is_err() {
                gl.delete_program(program);
            }
            result
        }
    }
}

impl NativeGlProvider {
    /// # Safety
    /// The provider's exact context is current during normal device teardown.
    pub(crate) unsafe fn clear_gles_plane_readback_programs(&mut self) {
        if let Some(programs) = self.gles_plane_readback_programs.take() {
            // SAFETY: caller holds the owning context current.
            unsafe { programs.destroy(&self.gl) };
        }
    }

    pub(super) fn gles_plane_readback(
        &mut self,
        op: &'static str,
        texture: glow::NativeTexture,
        desc: GlTextureDesc,
        source: GlTextureRegion,
        buffer: glow::NativeBuffer,
        dest: GlBufferRange,
        layout: GlPixelLayout,
    ) -> Result<(), GlError> {
        if !matches!(self.discovery.context().profile(), GlFamilyProfile::Embedded { major, minor } if major > 3 || (major == 3 && minor >= 1))
            || !self
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
                reason: "GLES plane readback requires ES 3.1 compute and storage buffers",
            });
        }
        let (stencil, bpp) = match (desc.format, source.subresource.aspect, layout.format) {
            (GlFormat::Depth16Unorm, GlTextureAspect::DepthOnly, GlPixelFormat::Depth16) => {
                (false, 2usize)
            }
            (
                GlFormat::Depth32Float | GlFormat::Depth32FloatStencil8,
                GlTextureAspect::DepthOnly,
                GlPixelFormat::Depth32Float,
            ) => (false, 4),
            (
                GlFormat::Stencil8 | GlFormat::Depth24PlusStencil8 | GlFormat::Depth32FloatStencil8,
                GlTextureAspect::StencilOnly,
                GlPixelFormat::Stencil8,
            ) => (true, 1),
            _ => {
                return Err(GlError::Unsupported {
                    operation: op,
                    reason: "GLES plane readback representation",
                });
            }
        };
        if !matches!(
            desc.dimension,
            GlTextureDimension::D2 | GlTextureDimension::D2Array
        ) || desc.sample_count != 1
        {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "GLES plane readback requires single-sample 2D textures",
            });
        }
        if !self
            .discovery
            .formats()
            .get_for(GlFormatResourceKind::Texture, desc.format, 1)
            .is_some_and(|f| f.sampled && f.copy_source)
        {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "GLES plane readback requires sampled copy-source format facts",
            });
        }
        let end = layout
            .required_bytes(source)
            .map_err(|_| Self::validation(op, "invalid readback layout"))?;
        if end > dest.size
            || dest
                .offset
                .checked_add(end)
                .is_none_or(|end| end > i32::MAX as u64)
        {
            return Err(Self::validation(
                op,
                "readback destination exceeds buffer or GLint",
            ));
        }
        let count = u64::from(source.extent.width)
            .checked_mul(u64::from(source.extent.height))
            .and_then(|n| n.checked_mul(u64::from(source.extent.depth_or_layers)))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| Self::validation(op, "plane readback size overflow"))?;
        let size = u64::from(count)
            .checked_mul(4)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| Self::validation(op, "plane scratch size overflow"))?;
        if u64::try_from(size).unwrap() > self.discovery.limits().max_storage_block_size
            || count.div_ceil(64) > self.discovery.limits().max_compute_work_group_count[0]
        {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "plane readback exceeds compute limits",
            });
        }
        let z = source
            .subresource
            .base_layer
            .checked_add(source.origin[2])
            .ok_or_else(|| Self::validation(op, "layer overflow"))?;
        let target = if desc.dimension == GlTextureDimension::D2 {
            glow::TEXTURE_2D
        } else {
            glow::TEXTURE_2D_ARRAY
        };
        if self.gles_plane_readback_programs.is_none() {
            // SAFETY: this command owns the current context and keeps the
            // cache only for its generation.
            self.gles_plane_readback_programs =
                Some(unsafe { GlesPlaneReadbackPrograms::create(&self.gl)? });
        }
        let (
            program,
            count_location,
            width_location,
            height_location,
            origin_x_location,
            origin_y_location,
            base_layer_location,
            mip_location,
        ) = {
            let variant = self
                .gles_plane_readback_programs
                .as_ref()
                .expect("plane readback cache was installed")
                .select(stencil, bpp, target == glow::TEXTURE_2D_ARRAY);
            (
                variant.raw,
                variant.count.clone(),
                variant.width.clone(),
                variant.height.clone(),
                variant.origin_x.clone(),
                variant.origin_y.clone(),
                variant.base_layer.clone(),
                variant.mip.clone(),
            )
        };
        // SAFETY: exclusively held current context and validated ranges. The
        // provider owns raw state, so this utility restores the CPU-shadowed
        // SSBO/program state and resets its other private bindings to their
        // canonical baseline without querying the driver.
        unsafe {
            let scratch = match self.gl.create_buffer() {
                Ok(v) => v,
                Err(_) => return Err(GlError::OutOfMemory { operation: op }),
            };
            let sampler = match self.gl.create_sampler() {
                Ok(v) => v,
                Err(_) => {
                    self.gl.delete_buffer(scratch);
                    return Err(GlError::OutOfMemory { operation: op });
                }
            };
            let old_generic = self.generic_storage_buffer;
            let old_index = self.storage_bindings.get(&0).copied();
            let previous_program = self.current_native_program;
            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_texture(target, Some(texture));
            let packed = matches!(
                desc.format,
                GlFormat::Depth24PlusStencil8 | GlFormat::Depth32FloatStencil8
            );
            if packed {
                self.gl.tex_parameter_i32(
                    target,
                    glow::DEPTH_STENCIL_TEXTURE_MODE,
                    if stencil {
                        glow::STENCIL_INDEX as i32
                    } else {
                        glow::DEPTH_COMPONENT as i32
                    },
                );
            }
            self.gl.sampler_parameter_i32(
                sampler,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST_MIPMAP_NEAREST as i32,
            );
            self.gl
                .sampler_parameter_i32(sampler, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            self.gl
                .sampler_parameter_i32(sampler, glow::TEXTURE_COMPARE_MODE, glow::NONE as i32);
            self.gl.bind_sampler(0, Some(sampler));
            self.gl
                .bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(scratch));
            self.gl
                .buffer_data_size(glow::SHADER_STORAGE_BUFFER, size, glow::STREAM_COPY);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(scratch));
            self.gl.use_program(Some(program));
            self.gl.uniform_1_u32(Some(&count_location), count);
            self.gl
                .uniform_1_u32(Some(&width_location), source.extent.width);
            self.gl
                .uniform_1_u32(Some(&height_location), source.extent.height);
            self.gl
                .uniform_1_u32(Some(&origin_x_location), source.origin[0]);
            self.gl
                .uniform_1_u32(Some(&origin_y_location), source.origin[1]);
            if let Some(base_layer_location) = base_layer_location.as_ref() {
                self.gl.uniform_1_u32(Some(base_layer_location), z);
            }
            self.gl.uniform_1_i32(
                Some(&mip_location),
                i32::try_from(source.subresource.mip_level)
                    .map_err(|_| Self::validation(op, "mip exceeds GLint"))?,
            );
            self.gl.memory_barrier(glow::TEXTURE_FETCH_BARRIER_BIT);
            self.gl.dispatch_compute(count.div_ceil(64), 1, 1);
            self.gl.memory_barrier(glow::BUFFER_UPDATE_BARRIER_BIT);
            let mut result = self.driver_error(op);
            if result.is_ok() {
                self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(scratch));
                self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
                for layer in 0..source.extent.depth_or_layers {
                    for y in 0..source.extent.height {
                        for x in 0..source.extent.width {
                            let i = (u64::from(layer) * u64::from(source.extent.height)
                                + u64::from(y))
                                * u64::from(source.extent.width)
                                + u64::from(x);
                            let to = dest.offset
                                + layout.offset
                                + u64::from(layout.bytes_per_row)
                                    * (u64::from(layout.rows_per_image) * u64::from(layer)
                                        + u64::from(y))
                                + u64::from(x) * bpp as u64;
                            let source_offset = i.checked_mul(4).ok_or_else(|| {
                                Self::validation(op, "scratch source offset overflow")
                            })?;
                            let copy_size =
                                u64::try_from(bpp).expect("pixel component count fits u64");
                            if let Some(functions) = self.buffer_functions {
                                (functions.copy_buffer_sub_data)(
                                    glow::COPY_READ_BUFFER,
                                    glow::COPY_WRITE_BUFFER,
                                    isize::try_from(source_offset).map_err(|_| {
                                        Self::validation(op, "scratch source exceeds native range")
                                    })?,
                                    isize::try_from(to).map_err(|_| {
                                        Self::validation(op, "destination exceeds native range")
                                    })?,
                                    isize::try_from(copy_size).expect("byte component fits isize"),
                                );
                            } else {
                                self.gl.copy_buffer_sub_data(
                                    glow::COPY_READ_BUFFER,
                                    glow::COPY_WRITE_BUFFER,
                                    i32::try_from(source_offset).map_err(|_| {
                                        Self::validation(
                                            op,
                                            "GL buffer ABI entry points unavailable",
                                        )
                                    })?,
                                    i32::try_from(to).map_err(|_| {
                                        Self::validation(
                                            op,
                                            "GL buffer ABI entry points unavailable",
                                        )
                                    })?,
                                    i32::try_from(copy_size).expect("pixel component fits i32"),
                                );
                            }
                        }
                    }
                }
                result = self.defer_driver_error(op);
            }
            if packed {
                self.gl.tex_parameter_i32(
                    target,
                    glow::DEPTH_STENCIL_TEXTURE_MODE,
                    glow::DEPTH_COMPONENT as i32,
                );
            }
            self.gl.bind_texture(target, None);
            self.gl.bind_sampler(0, None);
            self.gl.active_texture(glow::TEXTURE0);
            if let Some(binding) =
                old_index.filter(|binding| binding.buffer.is_some() && binding.size > 0)
            {
                self.bind_buffer_range_wide(
                    glow::SHADER_STORAGE_BUFFER,
                    0,
                    binding.buffer,
                    binding.offset,
                    binding.size,
                    op,
                )?;
            } else {
                self.gl.bind_buffer_base(
                    glow::SHADER_STORAGE_BUFFER,
                    0,
                    old_index.and_then(|binding| binding.buffer),
                );
            }
            self.gl
                .bind_buffer(glow::SHADER_STORAGE_BUFFER, old_generic);
            self.gl.use_program(previous_program);
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, None);
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            self.gl.delete_sampler(sampler);
            self.gl.delete_buffer(scratch);
            self.context_state
                .event(StateEvent::ScopedRawAccess(ScopedRawAccess::all()));
            result.and_then(|()| self.defer_driver_error(op))
        }
    }
}
