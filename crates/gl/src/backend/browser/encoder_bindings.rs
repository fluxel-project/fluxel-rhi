//! Immediate WebGL2 bind-group and vertex-input verbs.
//!
//! These operations update actual WebGL binding state during the Core encoder
//! call. They keep no CPU-side binding packet for a later draw or submit.

use crate::backend::api::{
    GlTextureDimension, GlVertexAttribute, GlVertexBufferLayout, GlVertexFormat, GlVertexStepMode,
};
use fluxel_rhi_core::api::{Error, Result};
use web_sys::WebGl2RenderingContext as Gl;

use super::owner::{
    BrowserBuffer, BrowserEncoder, BrowserRasterProgram, BrowserSampler, BrowserTexture,
    BrowserVertexArray,
};

impl BrowserEncoder {
    /// Binds a texture and optional sampler to a texture unit immediately.
    /// `None` clears any sampler left on the unit for sampler-free fetches.
    /// Program uniform assignment is deliberately separate because a bind group
    /// may reuse the same native texture unit across several programs.
    pub(crate) fn bind_sampled_texture(
        &mut self,
        unit: u32,
        texture: &BrowserTexture,
        sampler: Option<&BrowserSampler>,
    ) -> Result<()> {
        self.same_owner(&texture.owner)?;
        let texture_raw = texture.texture()?;
        if let Some(sampler) = sampler {
            self.same_owner(&sampler.owner)?;
        }
        self.require_raster()?;
        let active_unit = Gl::TEXTURE0.checked_add(unit).ok_or(Error::InvalidRange)?;
        let target = match texture.desc.dimension {
            GlTextureDimension::D2 => Gl::TEXTURE_2D,
            GlTextureDimension::D2Array => Gl::TEXTURE_2D_ARRAY,
            GlTextureDimension::D3 => Gl::TEXTURE_3D,
            GlTextureDimension::Cube => Gl::TEXTURE_CUBE_MAP,
            _ => return Err(Error::Unsupported("WebGL2 sampled texture dimension")),
        };
        self.with_context("bind_rgba8_texture", |raw| {
            raw.active_texture(active_unit);
            raw.bind_texture(target, Some(texture_raw));
            raw.bind_sampler(unit, sampler.map(|sampler| &sampler.raw));
        })
    }

    pub(crate) fn bind_rgba8_texture(
        &mut self,
        unit: u32,
        texture: &BrowserTexture,
        sampler: &BrowserSampler,
    ) -> Result<()> {
        self.bind_sampled_texture(unit, texture, Some(sampler))
    }

    /// Updates only the texture half of a combined texture unit.
    pub(crate) fn bind_sampled_texture_only(
        &mut self,
        unit: u32,
        texture: &BrowserTexture,
    ) -> Result<()> {
        self.same_owner(&texture.owner)?;
        let texture_raw = texture.texture()?;
        self.require_raster()?;
        let active_unit = Gl::TEXTURE0.checked_add(unit).ok_or(Error::InvalidRange)?;
        let target = match texture.desc.dimension {
            GlTextureDimension::D2 => Gl::TEXTURE_2D,
            GlTextureDimension::D2Array => Gl::TEXTURE_2D_ARRAY,
            GlTextureDimension::D3 => Gl::TEXTURE_3D,
            GlTextureDimension::Cube => Gl::TEXTURE_CUBE_MAP,
            _ => return Err(Error::Unsupported("WebGL2 sampled texture dimension")),
        };
        self.with_context("bind_sampled_texture_only", |raw| {
            raw.active_texture(active_unit);
            raw.bind_texture(target, Some(texture_raw));
        })
    }

    pub(crate) fn set_depth_stencil_texture_mode(
        &mut self,
        texture: &BrowserTexture,
        stencil: bool,
    ) -> Result<()> {
        self.same_owner(&texture.owner)?;
        let supported = self.owner.stencil_texturing_supported()?;
        if !supported {
            return if stencil {
                Err(Error::Unsupported("WebGL2 stencil texturing extension"))
            } else {
                Ok(())
            };
        }
        let texture_raw = texture.texture()?;
        let target = match texture.desc.dimension {
            GlTextureDimension::D2 => Gl::TEXTURE_2D,
            GlTextureDimension::D2Array => Gl::TEXTURE_2D_ARRAY,
            _ => return Err(Error::Unsupported("WebGL2 depth stencil texture dimension")),
        };
        self.with_context("depth stencil texture mode", |raw| {
            raw.bind_texture(target, Some(texture_raw));
            raw.tex_parameteri(
                target,
                0x90EA,
                if stencil {
                    0x1901
                } else {
                    Gl::DEPTH_COMPONENT as i32
                },
            );
        })
    }

    /// Updates only the sampler half of a combined texture unit.
    pub(crate) fn bind_sampler_only(
        &mut self,
        unit: u32,
        sampler: Option<&BrowserSampler>,
    ) -> Result<()> {
        if let Some(sampler) = sampler {
            self.same_owner(&sampler.owner)?;
        }
        self.require_raster()?;
        self.with_context("bind_sampler_only", |raw| {
            raw.bind_sampler(unit, sampler.map(|sampler| &sampler.raw))
        })
    }

    /// Binds a named ESSL uniform block to one WebGL indexed buffer point.
    ///
    /// This is the direct lowering for 02's `Scene` uniform. The range check
    /// happens before issuing either block association or buffer-range state,
    /// so an invalid Core binding cannot leave a partially changed native pass.
    pub(crate) fn bind_uniform_buffer_named(
        &mut self,
        program: &BrowserRasterProgram,
        binding: u32,
        buffer: &BrowserBuffer,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        self.same_owner(&program.owner)?;
        self.same_owner(&buffer.owner)?;
        self.require_raster()?;
        let end = offset.checked_add(size).ok_or(Error::InvalidRange)?;
        if size == 0 || end > buffer.size {
            return Err(Error::InvalidRange);
        }
        let offset = i32::try_from(offset).map_err(|_| Error::InvalidRange)?;
        let size = i32::try_from(size).map_err(|_| Error::InvalidRange)?;
        self.with_context("bind_uniform_buffer_named", |raw| {
            raw.bind_buffer_range_with_f64_and_f64(
                Gl::UNIFORM_BUFFER,
                binding,
                Some(&buffer.raw),
                f64::from(offset),
                f64::from(size),
            );
        })
    }

    /// Configures one float vertex attribute on the VAO currently bound by the
    /// encoder. Attribute state is real WebGL VAO state, never a replay record.
    pub(crate) fn set_f32_vertex_attribute(
        &mut self,
        buffer: &BrowserBuffer,
        location: u32,
        components: i32,
        stride: i32,
        offset: i32,
    ) -> Result<()> {
        self.same_owner(&buffer.owner)?;
        self.require_raster()?;
        if !(1..=4).contains(&components) || stride < 0 || offset < 0 {
            return Err(Error::InvalidRange);
        }
        self.with_context("set_f32_vertex_attribute", |raw| {
            raw.bind_buffer(Gl::ARRAY_BUFFER, Some(&buffer.raw));
            raw.enable_vertex_attrib_array(location);
            raw.vertex_attrib_pointer_with_i32(
                location,
                components,
                Gl::FLOAT,
                false,
                stride,
                offset,
            );
        })
    }

    /// Re-emits one pipeline attribute for a newly selected Core buffer slot.
    ///
    /// WebGL2 stores the concrete ARRAY_BUFFER name in the VAO.  The immutable
    /// pipeline owns the structural layout, while this immediate call supplies
    /// the current buffer and binding base.  No vertex input is queued here.
    pub(crate) fn set_vertex_attribute(
        &mut self,
        buffer: &BrowserBuffer,
        layout: &GlVertexBufferLayout,
        attribute: &GlVertexAttribute,
        binding_offset: u64,
    ) -> Result<()> {
        self.same_owner(&buffer.owner)?;
        self.require_raster()?;
        // Core zero stride means one constant address; WebGL's legacy
        // vertexAttribPointer zero stride instead means tightly packed.
        if layout.stride == 0 {
            return self.with_context("disable_constant_vertex_attribute", |raw| {
                raw.disable_vertex_attrib_array(attribute.location);
            });
        }
        let stride = layout.stride;
        let offset = binding_offset
            .checked_add(u64::from(attribute.offset))
            .and_then(|value| i32::try_from(value).ok())
            .ok_or(Error::InvalidRange)?;
        let stride = i32::try_from(stride).map_err(|_| Error::InvalidRange)?;
        let path = vertex_attribute_path(attribute.format);
        self.with_context("set_vertex_attribute", |raw| {
            raw.bind_buffer(Gl::ARRAY_BUFFER, Some(&buffer.raw));
            match path {
                VertexAttributePath::Float(components, kind, normalized) => raw
                    .vertex_attrib_pointer_with_i32(
                        attribute.location,
                        components,
                        kind,
                        normalized,
                        stride,
                        offset,
                    ),
                VertexAttributePath::Integer(components, kind) => raw
                    .vertex_attrib_i_pointer_with_i32(
                        attribute.location,
                        components,
                        kind,
                        stride,
                        offset,
                    ),
            }
            raw.enable_vertex_attrib_array(attribute.location);
            raw.vertex_attrib_divisor(
                attribute.location,
                match layout.step_mode {
                    GlVertexStepMode::Vertex => 0,
                    GlVertexStepMode::Instance => 1,
                },
            );
        })
    }

    /// Binds an index buffer to the currently bound VAO for indexed draws.
    pub(crate) fn set_index_buffer(&mut self, buffer: &BrowserBuffer) -> Result<()> {
        self.same_owner(&buffer.owner)?;
        self.require_raster()?;
        self.with_context("set_index_buffer", |raw| {
            raw.bind_buffer(Gl::ELEMENT_ARRAY_BUFFER, Some(&buffer.raw));
        })
    }
}

enum VertexAttributePath {
    Float(i32, u32, bool),
    Integer(i32, u32),
}

const fn vertex_attribute_path(format: GlVertexFormat) -> VertexAttributePath {
    use GlVertexFormat as F;
    match format {
        F::Uint8x2 => VertexAttributePath::Integer(2, Gl::UNSIGNED_BYTE),
        F::Uint8x4 => VertexAttributePath::Integer(4, Gl::UNSIGNED_BYTE),
        F::Sint8x2 => VertexAttributePath::Integer(2, Gl::BYTE),
        F::Sint8x4 => VertexAttributePath::Integer(4, Gl::BYTE),
        F::Unorm8x2 => VertexAttributePath::Float(2, Gl::UNSIGNED_BYTE, true),
        F::Unorm8x4 => VertexAttributePath::Float(4, Gl::UNSIGNED_BYTE, true),
        F::Snorm8x2 => VertexAttributePath::Float(2, Gl::BYTE, true),
        F::Snorm8x4 => VertexAttributePath::Float(4, Gl::BYTE, true),
        F::Uint16x2 => VertexAttributePath::Integer(2, Gl::UNSIGNED_SHORT),
        F::Uint16x4 => VertexAttributePath::Integer(4, Gl::UNSIGNED_SHORT),
        F::Sint16x2 => VertexAttributePath::Integer(2, Gl::SHORT),
        F::Sint16x4 => VertexAttributePath::Integer(4, Gl::SHORT),
        F::Unorm16x2 => VertexAttributePath::Float(2, Gl::UNSIGNED_SHORT, true),
        F::Unorm16x4 => VertexAttributePath::Float(4, Gl::UNSIGNED_SHORT, true),
        F::Snorm16x2 => VertexAttributePath::Float(2, Gl::SHORT, true),
        F::Snorm16x4 => VertexAttributePath::Float(4, Gl::SHORT, true),
        F::Float16x2 => VertexAttributePath::Float(2, Gl::HALF_FLOAT, false),
        F::Float16x4 => VertexAttributePath::Float(4, Gl::HALF_FLOAT, false),
        F::Float64 | F::Float64x2 | F::Float64x3 | F::Float64x4 => {
            panic!("Float64 vertex attributes are rejected by WebGL2 VAO creation")
        }
        F::Float32 => VertexAttributePath::Float(1, Gl::FLOAT, false),
        F::Float32x2 => VertexAttributePath::Float(2, Gl::FLOAT, false),
        F::Float32x3 => VertexAttributePath::Float(3, Gl::FLOAT, false),
        F::Float32x4 => VertexAttributePath::Float(4, Gl::FLOAT, false),
        F::Uint32 => VertexAttributePath::Integer(1, Gl::UNSIGNED_INT),
        F::Uint32x2 => VertexAttributePath::Integer(2, Gl::UNSIGNED_INT),
        F::Uint32x3 => VertexAttributePath::Integer(3, Gl::UNSIGNED_INT),
        F::Uint32x4 => VertexAttributePath::Integer(4, Gl::UNSIGNED_INT),
        F::Sint32 => VertexAttributePath::Integer(1, Gl::INT),
        F::Sint32x2 => VertexAttributePath::Integer(2, Gl::INT),
        F::Sint32x3 => VertexAttributePath::Integer(3, Gl::INT),
        F::Sint32x4 => VertexAttributePath::Integer(4, Gl::INT),
    }
}
