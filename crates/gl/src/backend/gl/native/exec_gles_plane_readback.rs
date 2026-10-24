//! GLES depth/stencil buffer copies using a private compute shader.
//! Each texel is written as one scratch uint and copied to its byte-sized
//! destination without touching surrounding row padding.
use super::super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    GlBufferRange, GlCapability, GlError, GlFamilyApi as _, GlFamilyProfile, GlFormat,
    GlFormatResourceKind, GlPixelFormat, GlPixelLayout, GlTextureAspect, GlTextureDesc,
    GlTextureDimension, GlTextureRegion,
};
use glow::HasContext as _;

impl NativeGlProvider {
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
        let sampler_type = if stencil { "usampler" } else { "sampler" };
        let target = if desc.dimension == GlTextureDimension::D2 {
            glow::TEXTURE_2D
        } else {
            glow::TEXTURE_2D_ARRAY
        };
        let binding = if target == glow::TEXTURE_2D {
            glow::TEXTURE_BINDING_2D
        } else {
            glow::TEXTURE_BINDING_2D_ARRAY
        };
        let tex_type = if target == glow::TEXTURE_2D {
            "2D"
        } else {
            "2DArray"
        };
        let coords = if target == glow::TEXTURE_2D {
            format!(
                "ivec2(int(x+{}u),int(y+{}u))",
                source.origin[0], source.origin[1]
            )
        } else {
            format!(
                "ivec3(int(x+{}u),int(y+{}u),int(layer+{}u))",
                source.origin[0], source.origin[1], z
            )
        };
        let convert = if stencil {
            "v"
        } else if bpp == 2 {
            "uint(round(clamp(v,0.0,1.0)*65535.0))"
        } else {
            "floatBitsToUint(v)"
        };
        let code = format!(
            "#version 310 es\nprecision highp int; precision highp float; precision highp {sampler_type}{tex_type};\nlayout(local_size_x=64) in; layout(binding=0) uniform {sampler_type}{tex_type} image; layout(std430,binding=0) writeonly buffer D {{ uint out_data[]; }}; void main() {{ uint i=gl_GlobalInvocationID.x; if(i>={count}u)return; uint x=i%{}u; uint y=(i/{}u)%{}u; uint layer=i/({}u*{}u); {} v=texelFetch(image,{coords},{}).r; out_data[i]={convert}; }}",
            source.extent.width,
            source.extent.width,
            source.extent.height,
            source.extent.width,
            source.extent.height,
            if stencil { "uint" } else { "float" },
            source.subresource.mip_level
        );
        // SAFETY: exclusively held current context, validated ranges. All
        // temporary objects, native bindings, and modified texture mode are restored.
        unsafe {
            let shader = self
                .gl
                .create_shader(glow::COMPUTE_SHADER)
                .map_err(|_| GlError::OutOfMemory { operation: op })?;
            self.gl.shader_source(shader, &code);
            self.gl.compile_shader(shader);
            if !self.gl.get_shader_compile_status(shader) {
                let message = self.gl.get_shader_info_log(shader);
                self.gl.delete_shader(shader);
                return Err(GlError::Driver {
                    operation: op,
                    message,
                });
            }
            let program = match self.gl.create_program() {
                Ok(v) => v,
                Err(_) => {
                    self.gl.delete_shader(shader);
                    return Err(GlError::OutOfMemory { operation: op });
                }
            };
            self.gl.attach_shader(program, shader);
            self.gl.link_program(program);
            self.gl.delete_shader(shader);
            if !self.gl.get_program_link_status(program) {
                let message = self.gl.get_program_info_log(program);
                self.gl.delete_program(program);
                return Err(GlError::Driver {
                    operation: op,
                    message,
                });
            }
            let scratch = match self.gl.create_buffer() {
                Ok(v) => v,
                Err(_) => {
                    self.gl.delete_program(program);
                    return Err(GlError::OutOfMemory { operation: op });
                }
            };
            let sampler = match self.gl.create_sampler() {
                Ok(v) => v,
                Err(_) => {
                    self.gl.delete_buffer(scratch);
                    self.gl.delete_program(program);
                    return Err(GlError::OutOfMemory { operation: op });
                }
            };
            let handle = |raw: i32| std::num::NonZeroU32::new(raw as u32).map(glow::NativeBuffer);
            let old_generic = handle(
                self.gl
                    .get_parameter_i32(glow::SHADER_STORAGE_BUFFER_BINDING),
            );
            let old_index = handle(
                self.gl
                    .get_parameter_indexed_i32(glow::SHADER_STORAGE_BUFFER_BINDING, 0),
            );
            let old_offset = self
                .gl
                .get_parameter_indexed_i64(glow::SHADER_STORAGE_BUFFER_START, 0);
            let old_size = self
                .gl
                .get_parameter_indexed_i64(glow::SHADER_STORAGE_BUFFER_SIZE, 0);
            if old_index.is_some()
                && (old_offset < 0
                    || old_size < 0
                    || old_offset > i32::MAX as i64
                    || old_size > i32::MAX as i64)
            {
                self.gl.delete_sampler(sampler);
                self.gl.delete_buffer(scratch);
                self.gl.delete_program(program);
                return Err(GlError::Unsupported {
                    operation: op,
                    reason: "existing SSBO binding cannot be restored by glow",
                });
            }
            let old_copy_read = handle(self.gl.get_parameter_i32(glow::COPY_READ_BUFFER_BINDING));
            let old_copy_write = handle(self.gl.get_parameter_i32(glow::COPY_WRITE_BUFFER_BINDING));
            let previous_program =
                std::num::NonZeroU32::new(self.gl.get_parameter_i32(glow::CURRENT_PROGRAM) as u32)
                    .map(glow::NativeProgram);
            let old_active = self.gl.get_parameter_i32(glow::ACTIVE_TEXTURE) as u32;
            self.gl.active_texture(glow::TEXTURE0);
            let old_texture = std::num::NonZeroU32::new(self.gl.get_parameter_i32(binding) as u32)
                .map(glow::NativeTexture);
            let old_sampler =
                std::num::NonZeroU32::new(self.gl.get_parameter_i32(glow::SAMPLER_BINDING) as u32)
                    .map(glow::NativeSampler);
            self.gl.bind_texture(target, Some(texture));
            let packed = matches!(
                desc.format,
                GlFormat::Depth24PlusStencil8 | GlFormat::Depth32FloatStencil8
            );
            let old_mode = if packed {
                let v = self
                    .gl
                    .get_tex_parameter_i32(target, glow::DEPTH_STENCIL_TEXTURE_MODE);
                self.gl.tex_parameter_i32(
                    target,
                    glow::DEPTH_STENCIL_TEXTURE_MODE,
                    if stencil {
                        glow::STENCIL_INDEX as i32
                    } else {
                        glow::DEPTH_COMPONENT as i32
                    },
                );
                Some(v)
            } else {
                None
            };
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
                result = self.driver_error(op);
            }
            if let Some(mode) = old_mode {
                self.gl
                    .tex_parameter_i32(target, glow::DEPTH_STENCIL_TEXTURE_MODE, mode);
            }
            self.gl.bind_texture(target, old_texture);
            self.gl.bind_sampler(0, old_sampler);
            self.gl.active_texture(old_active);
            if old_index.is_some() && old_size > 0 {
                self.bind_buffer_range_wide(
                    glow::SHADER_STORAGE_BUFFER,
                    0,
                    old_index,
                    u64::try_from(old_offset)
                        .map_err(|_| Self::validation(op, "saved SSBO offset is negative"))?,
                    u64::try_from(old_size)
                        .map_err(|_| Self::validation(op, "saved SSBO size is negative"))?,
                    op,
                )?;
            } else {
                self.gl
                    .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, old_index);
            }
            self.gl
                .bind_buffer(glow::SHADER_STORAGE_BUFFER, old_generic);
            self.gl.use_program(previous_program);
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, old_copy_read);
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, old_copy_write);
            self.gl.delete_sampler(sampler);
            self.gl.delete_buffer(scratch);
            self.gl.delete_program(program);
            result.and_then(|()| self.driver_error(op))
        }
    }
}
