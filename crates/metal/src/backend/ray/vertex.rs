//! Metal acceleration-geometry vertex format and layout lowering.
use fluxel_rhi_core::api;

pub(super) fn acceleration_vertex_format(
    format: api::VertexFormat,
) -> api::Result<(objc2_metal::MTLAttributeFormat, u64, u64)> {
    use objc2_metal::MTLAttributeFormat as Mtl;
    let (format, size) = match format {
        api::VertexFormat::Float32(1) => (Mtl::Float, 4),
        api::VertexFormat::Float32(2) => (Mtl::Float2, 8),
        api::VertexFormat::Float32(3) => (Mtl::Float3, 12),
        api::VertexFormat::Float32(4) => (Mtl::Float4, 16),
        api::VertexFormat::Uint32(1) => (Mtl::UInt, 4),
        api::VertexFormat::Uint32(2) => (Mtl::UInt2, 8),
        api::VertexFormat::Uint32(3) => (Mtl::UInt3, 12),
        api::VertexFormat::Uint32(4) => (Mtl::UInt4, 16),
        api::VertexFormat::Sint32(1) => (Mtl::Int, 4),
        api::VertexFormat::Sint32(2) => (Mtl::Int2, 8),
        api::VertexFormat::Sint32(3) => (Mtl::Int3, 12),
        api::VertexFormat::Sint32(4) => (Mtl::Int4, 16),
        api::VertexFormat::Float16(1) => (Mtl::Half, 2),
        api::VertexFormat::Float16(2) => (Mtl::Half2, 4),
        api::VertexFormat::Float16(3) => (Mtl::Half3, 6),
        api::VertexFormat::Float16(4) => (Mtl::Half4, 8),
        api::VertexFormat::Uint16(1) => (Mtl::UShort, 2),
        api::VertexFormat::Uint16(2) => (Mtl::UShort2, 4),
        api::VertexFormat::Uint16(3) => (Mtl::UShort3, 6),
        api::VertexFormat::Uint16(4) => (Mtl::UShort4, 8),
        api::VertexFormat::Sint16(1) => (Mtl::Short, 2),
        api::VertexFormat::Sint16(2) => (Mtl::Short2, 4),
        api::VertexFormat::Sint16(3) => (Mtl::Short3, 6),
        api::VertexFormat::Sint16(4) => (Mtl::Short4, 8),
        api::VertexFormat::Uint8(1) => (Mtl::UChar, 1),
        api::VertexFormat::Uint8(2) => (Mtl::UChar2, 2),
        api::VertexFormat::Uint8(3) => (Mtl::UChar3, 3),
        api::VertexFormat::Uint8(4) => (Mtl::UChar4, 4),
        api::VertexFormat::Sint8(1) => (Mtl::Char, 1),
        api::VertexFormat::Sint8(2) => (Mtl::Char2, 2),
        api::VertexFormat::Sint8(3) => (Mtl::Char3, 3),
        api::VertexFormat::Sint8(4) => (Mtl::Char4, 4),
        api::VertexFormat::Unorm8(1) => (Mtl::UCharNormalized, 1),
        api::VertexFormat::Unorm8(2) => (Mtl::UChar2Normalized, 2),
        api::VertexFormat::Unorm8(3) => (Mtl::UChar3Normalized, 3),
        api::VertexFormat::Unorm8(4) => (Mtl::UChar4Normalized, 4),
        api::VertexFormat::Snorm8(1) => (Mtl::CharNormalized, 1),
        api::VertexFormat::Snorm8(2) => (Mtl::Char2Normalized, 2),
        api::VertexFormat::Snorm8(3) => (Mtl::Char3Normalized, 3),
        api::VertexFormat::Snorm8(4) => (Mtl::Char4Normalized, 4),
        api::VertexFormat::Unorm16(1) => (Mtl::UShortNormalized, 2),
        api::VertexFormat::Unorm16(2) => (Mtl::UShort2Normalized, 4),
        api::VertexFormat::Unorm16(3) => (Mtl::UShort3Normalized, 6),
        api::VertexFormat::Unorm16(4) => (Mtl::UShort4Normalized, 8),
        api::VertexFormat::Snorm16(1) => (Mtl::ShortNormalized, 2),
        api::VertexFormat::Snorm16(2) => (Mtl::Short2Normalized, 4),
        api::VertexFormat::Snorm16(3) => (Mtl::Short3Normalized, 6),
        api::VertexFormat::Snorm16(4) => (Mtl::Short4Normalized, 8),
        _ => return Err(api::Error::Unsupported("Metal vertex format")),
    };
    let alignment = match format {
        Mtl::Float
        | Mtl::Float2
        | Mtl::Float3
        | Mtl::Float4
        | Mtl::Int
        | Mtl::Int2
        | Mtl::Int3
        | Mtl::Int4
        | Mtl::UInt
        | Mtl::UInt2
        | Mtl::UInt3
        | Mtl::UInt4 => 4,
        Mtl::Half
        | Mtl::Half2
        | Mtl::Half3
        | Mtl::Half4
        | Mtl::UShort
        | Mtl::UShort2
        | Mtl::UShort3
        | Mtl::UShort4
        | Mtl::Short
        | Mtl::Short2
        | Mtl::Short3
        | Mtl::Short4
        | Mtl::UShortNormalized
        | Mtl::UShort2Normalized
        | Mtl::UShort3Normalized
        | Mtl::UShort4Normalized
        | Mtl::ShortNormalized
        | Mtl::Short2Normalized
        | Mtl::Short3Normalized
        | Mtl::Short4Normalized => 2,
        _ => 1,
    };
    Ok((format, size, alignment))
}
