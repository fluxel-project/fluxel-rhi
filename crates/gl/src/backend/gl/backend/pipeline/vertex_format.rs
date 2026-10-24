//! Convert Core vertex attribute formats to native GL declarations.

use super::*;

pub(super) fn lower_vertex_format(
    value: VertexFormat,
) -> Result<crate::backend::gl::api::GlVertexFormat> {
    use crate::backend::gl::api::GlVertexFormat as Gl;
    Ok(match value {
        VertexFormat::Float32(1) => Gl::Float32,
        VertexFormat::Float32(2) => Gl::Float32x2,
        VertexFormat::Float32(3) => Gl::Float32x3,
        VertexFormat::Float32(4) => Gl::Float32x4,
        VertexFormat::Uint32(1) => Gl::Uint32,
        VertexFormat::Uint32(2) => Gl::Uint32x2,
        VertexFormat::Uint32(3) => Gl::Uint32x3,
        VertexFormat::Uint32(4) => Gl::Uint32x4,
        VertexFormat::Sint32(1) => Gl::Sint32,
        VertexFormat::Sint32(2) => Gl::Sint32x2,
        VertexFormat::Sint32(3) => Gl::Sint32x3,
        VertexFormat::Sint32(4) => Gl::Sint32x4,
        VertexFormat::Unorm8(2) => Gl::Unorm8x2,
        VertexFormat::Unorm8(4) => Gl::Unorm8x4,
        VertexFormat::Uint8(2) => Gl::Uint8x2,
        VertexFormat::Uint8(4) => Gl::Uint8x4,
        VertexFormat::Sint8(2) => Gl::Sint8x2,
        VertexFormat::Sint8(4) => Gl::Sint8x4,
        VertexFormat::Snorm8(2) => Gl::Snorm8x2,
        VertexFormat::Snorm8(4) => Gl::Snorm8x4,
        VertexFormat::Uint16(2) => Gl::Uint16x2,
        VertexFormat::Uint16(4) => Gl::Uint16x4,
        VertexFormat::Sint16(2) => Gl::Sint16x2,
        VertexFormat::Sint16(4) => Gl::Sint16x4,
        VertexFormat::Unorm16(2) => Gl::Unorm16x2,
        VertexFormat::Unorm16(4) => Gl::Unorm16x4,
        VertexFormat::Snorm16(2) => Gl::Snorm16x2,
        VertexFormat::Snorm16(4) => Gl::Snorm16x4,
        VertexFormat::Float16(2) => Gl::Float16x2,
        VertexFormat::Float16(4) => Gl::Float16x4,
        VertexFormat::Float64(1) => Gl::Float64,
        VertexFormat::Float64(2) => Gl::Float64x2,
        VertexFormat::Float64(3) => Gl::Float64x3,
        VertexFormat::Float64(4) => Gl::Float64x4,
        _ => return Err(Error::Unsupported("GL vertex format")),
    })
}
