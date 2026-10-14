//! Immediate WebGL2 bind-group and vertex-input verbs.
//!
//! These operations update actual WebGL binding state during the Core encoder
//! call. They keep no CPU-side binding packet for a later draw or submit.

use crate::backend::gl::api::{
    GlTextureDimension, GlVertexAttribute, GlVertexBufferLayout, GlVertexFormat, GlVertexStepMode,
};
use fluxel_rhi_core::api::{Error, Result};
use web_sys::WebGl2RenderingContext as Gl;

use super::owner::{
    BrowserBuffer, BrowserEncoder, BrowserRasterProgram, BrowserSampler, BrowserTexture,
    BrowserVertexArray,
};

impl BrowserEncoder {
    /// Binds one sampled texture and sampler to a texture unit immediately.
    /// Program uniform assignment is deliberately separate because a bind group
    /// may reuse the same native texture unit across several programs.
    pub(crate) fn bind_rgba8_texture(
        &mut self,
        unit: u32,
        texture: &BrowserTexture,
        sampler: &BrowserSampler,
    ) -> Result<()> {
        self.same_owner(&texture.owner)?;
        self.same_owner(&sampler.owner)?;
        self.require_raster()?;
        let active_unit = Gl::TEXTURE0.checked_add(unit).ok_or(Error::InvalidRange)?;
        let target = match texture.desc.dimension {
            GlTextureDimension::D2 => Gl::TEXTURE_2D,
            GlTextureDimension::D2Array => Gl::TEXTURE_2D_ARRAY,
            _ => return Err(Error::Unsupported("WebGL2 sampled texture dimension")),
        };
        self.with_context("bind_rgba8_texture", |raw| {
            raw.active_texture(active_unit);
            raw.bind_texture(target, Some(&texture.raw));
            raw.bind_sampler(unit, Some(&sampler.raw));
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
        self.with_context("bind_uniform_buffer_named", |raw| {
            raw.bind_buffer_range_with_f64_and_f64(
                Gl::UNIFORM_BUFFER,
                binding,
                Some(&buffer.raw),
                offset as f64,
                size as f64,
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
        let stride = if layout.stride == 0 {
            vertex_format_size(attribute.format)
        } else {
            layout.stride
        };
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
        F::Uint8x2 => VertexAttributePath::Float(2, Gl::UNSIGNED_BYTE, false),
        F::Uint8x4 => VertexAttributePath::Float(4, Gl::UNSIGNED_BYTE, false),
        F::Sint8x2 => VertexAttributePath::Float(2, Gl::BYTE, false),
        F::Sint8x4 => VertexAttributePath::Float(4, Gl::BYTE, false),
        F::Unorm8x2 => VertexAttributePath::Float(2, Gl::UNSIGNED_BYTE, true),
        F::Unorm8x4 => VertexAttributePath::Float(4, Gl::UNSIGNED_BYTE, true),
        F::Snorm8x2 => VertexAttributePath::Float(2, Gl::BYTE, true),
        F::Snorm8x4 => VertexAttributePath::Float(4, Gl::BYTE, true),
        F::Uint16x2 => VertexAttributePath::Float(2, Gl::UNSIGNED_SHORT, false),
        F::Uint16x4 => VertexAttributePath::Float(4, Gl::UNSIGNED_SHORT, false),
        F::Sint16x2 => VertexAttributePath::Float(2, Gl::SHORT, false),
        F::Sint16x4 => VertexAttributePath::Float(4, Gl::SHORT, false),
        F::Unorm16x2 => VertexAttributePath::Float(2, Gl::UNSIGNED_SHORT, true),
        F::Unorm16x4 => VertexAttributePath::Float(4, Gl::UNSIGNED_SHORT, true),
        F::Snorm16x2 => VertexAttributePath::Float(2, Gl::SHORT, true),
        F::Snorm16x4 => VertexAttributePath::Float(4, Gl::SHORT, true),
        F::Float16x2 => VertexAttributePath::Float(2, Gl::HALF_FLOAT, false),
        F::Float16x4 => VertexAttributePath::Float(4, Gl::HALF_FLOAT, false),
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

const fn vertex_format_size(format: GlVertexFormat) -> u32 {
    use GlVertexFormat as F;
    match format {
        F::Uint8x2 | F::Sint8x2 | F::Unorm8x2 | F::Snorm8x2 => 2,
        F::Uint8x4 | F::Sint8x4 | F::Unorm8x4 | F::Snorm8x4 => 4,
        F::Uint16x2 | F::Sint16x2 | F::Unorm16x2 | F::Snorm16x2 | F::Float16x2 => 4,
        F::Uint16x4 | F::Sint16x4 | F::Unorm16x4 | F::Snorm16x4 | F::Float16x4 => 8,
        F::Float32 | F::Uint32 | F::Sint32 => 4,
        F::Float32x2 | F::Uint32x2 | F::Sint32x2 => 8,
        F::Float32x3 | F::Uint32x3 | F::Sint32x3 => 12,
        F::Float32x4 | F::Uint32x4 | F::Sint32x4 => 16,
    }
}
