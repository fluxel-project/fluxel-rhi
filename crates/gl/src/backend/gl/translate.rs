//! Stateless lowering from current Core descriptions to private GL values.
//!
//! This module has no resource ownership, command cache, or hazard logic.
//! It selects exact GL representations before a provider allocates or calls GL.

use fluxel_rhi_core::api::{
    self, BufferUsage, ShaderFormat, TextureDimension, TextureFormat, TextureUsage,
};

use super::backend::Gl;
use super::{api::*, profile::GlProfile};

fn unsupported(subject: &'static str) -> api::Error {
    api::Error::Unsupported(subject)
}

/// Lowers immutable buffer roles without choosing a mutable GL binding target.
pub(crate) fn lower_buffer_desc(value: &api::BufferDesc) -> api::Result<GlBufferDesc> {
    let mut usage = GlBufferUsage::EMPTY;
    for (core, gl) in [
        (BufferUsage::COPY_SRC, GlBufferUsage::COPY_SOURCE),
        (BufferUsage::COPY_DST, GlBufferUsage::COPY_DESTINATION),
        (BufferUsage::VERTEX, GlBufferUsage::VERTEX),
        (BufferUsage::INDEX, GlBufferUsage::INDEX),
        (BufferUsage::UNIFORM, GlBufferUsage::UNIFORM),
        (BufferUsage::STORAGE, GlBufferUsage::STORAGE),
        (BufferUsage::INDIRECT, GlBufferUsage::INDIRECT),
        (BufferUsage::MAP_READ, GlBufferUsage::MAP_READ),
        (BufferUsage::MAP_WRITE, GlBufferUsage::MAP_WRITE),
        // GL has no query-buffer resolve command in this backend. Query
        // results are observed synchronously and uploaded into this allocation.
        (BufferUsage::QUERY_RESOLVE, GlBufferUsage::COPY_DESTINATION),
    ] {
        if value.usage.contains(core) {
            usage = usage | gl;
        }
    }
    if value.usage.contains(BufferUsage::ACCEL_BUILD_INPUT)
        || value.usage.contains(BufferUsage::ACCEL_SCRATCH)
        || value.usage.contains(BufferUsage::SHADER_TABLE)
    {
        return Err(unsupported("GL buffer usage"));
    }
    let result = GlBufferDesc {
        size: value.size,
        usage,
    };
    result.validate().map_err(|_| api::Error::InvalidRange)?;
    Ok(result)
}

/// Lowers texture allocation facts while keeping array layers distinct from 3D depth.
pub(crate) fn lower_texture_desc(value: &api::TextureDesc) -> api::Result<GlTextureDesc> {
    // Core keeps the allocation description backend-neutral. Reject facts we
    // cannot represent before reducing D2 depth to GL's array-layer count;
    // otherwise an invalid public D2 extent would silently become valid.
    if value.extent.contains(&0)
        || value.array_layers == 0
        || (value.dimension != TextureDimension::D3 && value.extent[2] != 1)
        || (value.dimension == TextureDimension::D1 && value.extent[1] != 1)
        || value
            .view_formats
            .iter()
            .any(|format| *format != value.format)
    {
        return Err(api::Error::InvalidRange);
    }
    let dimension = match (value.dimension, value.array_layers) {
        (TextureDimension::D1, 1) => GlTextureDimension::D1,
        (TextureDimension::D2, 1) => GlTextureDimension::D2,
        (TextureDimension::D2, _) => GlTextureDimension::D2Array,
        (TextureDimension::D3, 1) => GlTextureDimension::D3,
        _ => return Err(api::Error::InvalidRange),
    };
    let mut usage = GlTextureUsage::EMPTY;
    for (core, gl) in [
        (TextureUsage::COPY_SRC, GlTextureUsage::COPY_SOURCE),
        (TextureUsage::COPY_DST, GlTextureUsage::COPY_DESTINATION),
        (TextureUsage::TEXTURE_BINDING, GlTextureUsage::SAMPLED),
        (
            TextureUsage::STORAGE_BINDING,
            GlTextureUsage::STORAGE_BINDING,
        ),
        (
            TextureUsage::RENDER_ATTACHMENT,
            GlTextureUsage::RENDER_ATTACHMENT,
        ),
    ] {
        if value.usage.contains(core) {
            usage = usage | gl;
        }
    }
    let depth_or_layers = if value.dimension == TextureDimension::D3 {
        value.extent[2]
    } else {
        value.array_layers
    };
    let result = GlTextureDesc {
        dimension,
        extent: GlExtent3d {
            width: value.extent[0],
            height: value.extent[1],
            depth_or_layers,
        },
        mip_level_count: value.mip_levels,
        sample_count: value.samples,
        format: lower_texture_format(value.format)?,
        usage,
    };
    result.validate().map_err(|_| api::Error::InvalidRange)?;
    Ok(result)
}

/// Lowers a Core texture region after the resource wrapper supplies its typed
/// native identity and immutable descriptor. Core intentionally hides native
/// handles, so this layer cannot recover either fact from `TextureRegion`.
/// Coordinates remain upper-left here; each provider performs the one native
/// GL Y conversion at its execution boundary.
pub(crate) fn lower_texture_region(
    texture: TextureId,
    descriptor: GlTextureDesc,
    value: &api::TextureRegion<'_, Gl>,
) -> api::Result<GlTextureRegion> {
    let aspect = match value.aspect.0 {
        1 => GlTextureAspect::Color,
        2 => GlTextureAspect::DepthOnly,
        4 => GlTextureAspect::StencilOnly,
        _ => return Err(unsupported("GL texture copy aspect")),
    };
    let (base_layer, layer_count, origin) =
        texture_region_coordinates(descriptor.dimension, value.origin, value.extent);
    let result = GlTextureRegion {
        subresource: GlTextureSubresource {
            texture,
            aspect,
            mip_level: value.mip,
            base_layer,
            layer_count,
        },
        origin,
        extent: GlExtent3d {
            width: value.extent[0],
            height: value.extent[1],
            depth_or_layers: value.extent[2],
        },
    };
    result
        .validate_for(descriptor)
        .map_err(|_| api::Error::InvalidRange)?;
    Ok(result)
}

fn texture_region_coordinates(
    dimension: GlTextureDimension,
    origin: [u32; 3],
    extent: [u32; 3],
) -> (u32, u32, [u32; 3]) {
    match dimension {
        // A 3D texture has one subresource layer. Its Z origin and depth are
        // region coordinates, retained below instead of being reinterpreted
        // as array-layer selection.
        GlTextureDimension::D3 => (0, 1, origin),
        // Array Z selects subresources. Once that selection is represented in
        // `base_layer`, the region origin is local to the selected subresources
        // and must start at zero; retaining the absolute Z double-counts it in
        // validation and again when lowering to glTexSubImage3D.
        _ => (origin[2], extent[2], [origin[0], origin[1], 0]),
    }
}

/// Lowers Core row layout to GL pixel-store facts without creating staging.
pub(crate) fn lower_texel_copy_layout(
    layout: api::TexelCopyBufferLayout,
    format: GlFormat,
) -> api::Result<GlPixelLayout> {
    let pixel_format = match format {
        GlFormat::Rgba8Unorm | GlFormat::Rgba8Srgb => GlPixelFormat::Rgba8,
        GlFormat::Bgra8Unorm | GlFormat::Bgra8Srgb => GlPixelFormat::Bgra8,
        GlFormat::R8Unorm => GlPixelFormat::Red8,
        GlFormat::Depth16Unorm => GlPixelFormat::Depth16,
        GlFormat::Depth24PlusStencil8 => GlPixelFormat::Depth24Stencil8,
        GlFormat::Depth32Float => GlPixelFormat::Depth32Float,
        _ => return Err(unsupported("GL pixel transfer format")),
    };
    if layout.bytes_per_row == 0 || layout.rows_per_image == 0 {
        return Err(api::Error::InvalidRange);
    }
    let alignment = [8, 4, 2, 1]
        .into_iter()
        .find(|alignment| layout.bytes_per_row % u32::from(*alignment) == 0)
        .expect("one divides every nonzero row stride");
    Ok(GlPixelLayout {
        format: pixel_format,
        bytes_per_row: layout.bytes_per_row,
        rows_per_image: layout.rows_per_image,
        offset: layout.offset,
        alignment,
        repack: GlRepackPolicy::Disallow,
    })
}

/// Lowers one direct GLSL entry and returns its backend-private source key.
pub(crate) fn lower_shader_source(
    profile: GlProfile,
    desc: &api::ShaderDesc,
    entry: &api::ShaderEntryPoint,
) -> api::Result<(GlShaderKey, GlShaderSource)> {
    let dialect = match (profile, desc.format) {
        (GlProfile::Desktop { .. }, ShaderFormat::Glsl(version)) => {
            GlShaderDialect::Desktop { version }
        }
        (GlProfile::Embedded { .. } | GlProfile::WebGl2, ShaderFormat::GlslEs(version)) => {
            GlShaderDialect::Embedded { version }
        }
        _ => return Err(unsupported("GL direct shader format/profile")),
    };
    let stage = match entry.stage {
        api::ShaderStage::Vertex => GlShaderStage::Vertex,
        api::ShaderStage::Fragment => GlShaderStage::Fragment,
        api::ShaderStage::Compute => GlShaderStage::Compute,
        _ => return Err(unsupported("GL shader stage")),
    };
    let text = core::str::from_utf8(&desc.bytes)
        .map_err(|_| api::Error::Shader("GLSL source is not UTF-8".into()))?
        .to_owned();
    let key = GlShaderKey(source_key(&desc.bytes, &entry.name));
    let source = GlShaderSource {
        stage,
        dialect,
        entry_point: entry.name.clone(),
        source_hash: key,
        text,
        debug_name: desc.label.clone(),
    };
    source
        .validate_for(profile)
        .map_err(|_| api::Error::InvalidState)?;
    Ok((key, source))
}

fn source_key(bytes: &[u8], entry: &str) -> [u8; 32] {
    let mut state = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes.iter().chain(entry.as_bytes()) {
        state = (state ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
    }
    let mut key = [0; 32];
    for index in 0..4 {
        key[index * 8..(index + 1) * 8]
            .copy_from_slice(&state.wrapping_add(index as u64).to_le_bytes());
    }
    key
}

pub(crate) fn lower_texture_format(value: TextureFormat) -> api::Result<GlFormat> {
    use GlFormat::*;
    Ok(match value {
        TextureFormat::R8Unorm => R8Unorm,
        TextureFormat::R8Snorm => R8Snorm,
        TextureFormat::R8Uint => R8Uint,
        TextureFormat::R8Sint => R8Sint,
        TextureFormat::Rg8Unorm => Rg8Unorm,
        TextureFormat::Rg8Snorm => Rg8Snorm,
        TextureFormat::Rg8Uint => Rg8Uint,
        TextureFormat::Rg8Sint => Rg8Sint,
        TextureFormat::Rgba8Unorm => Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb => Rgba8Srgb,
        TextureFormat::Rgba8Snorm => Rgba8Snorm,
        TextureFormat::Rgba8Uint => Rgba8Uint,
        TextureFormat::Rgba8Sint => Rgba8Sint,
        TextureFormat::Bgra8Unorm => Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb => Bgra8Srgb,
        TextureFormat::R16Unorm => R16Unorm,
        TextureFormat::R16Snorm => R16Snorm,
        TextureFormat::R16Uint => R16Uint,
        TextureFormat::R16Sint => R16Sint,
        TextureFormat::R16Float => R16Float,
        TextureFormat::Rg16Unorm => Rg16Unorm,
        TextureFormat::Rg16Snorm => Rg16Snorm,
        TextureFormat::Rg16Uint => Rg16Uint,
        TextureFormat::Rg16Sint => Rg16Sint,
        TextureFormat::Rg16Float => Rg16Float,
        TextureFormat::Rgba16Unorm => Rgba16Unorm,
        TextureFormat::Rgba16Snorm => Rgba16Snorm,
        TextureFormat::Rgba16Uint => Rgba16Uint,
        TextureFormat::Rgba16Sint => Rgba16Sint,
        TextureFormat::Rgba16Float => Rgba16Float,
        TextureFormat::R32Uint => R32Uint,
        TextureFormat::R32Sint => R32Sint,
        TextureFormat::R32Float => R32Float,
        TextureFormat::Rg32Uint => Rg32Uint,
        TextureFormat::Rg32Sint => Rg32Sint,
        TextureFormat::Rg32Float => Rg32Float,
        TextureFormat::Rgba32Uint => Rgba32Uint,
        TextureFormat::Rgba32Sint => Rgba32Sint,
        TextureFormat::Rgba32Float => Rgba32Float,
        TextureFormat::Rgb10a2Unorm => Rgb10a2Unorm,
        TextureFormat::Rgb10a2Uint => Rgb10a2Uint,
        TextureFormat::Rg11b10Ufloat => Rg11b10Ufloat,
        TextureFormat::Rgb9e5Ufloat => Rgb9e5Ufloat,
        TextureFormat::Depth16Unorm => Depth16Unorm,
        TextureFormat::Depth24Plus => Depth24Unorm,
        TextureFormat::Depth24PlusStencil8 => Depth24PlusStencil8,
        TextureFormat::Depth32Float => Depth32Float,
        TextureFormat::Depth32FloatStencil8 => Depth32FloatStencil8,
        TextureFormat::Stencil8 => Stencil8,
        TextureFormat::R64Uint => R64Uint,
        _ => {
            return Err(unsupported(
                "GL compressed or 64-bit two-channel texture format",
            ));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_upload_z_becomes_the_subresource_base_once() {
        assert_eq!(
            texture_region_coordinates(GlTextureDimension::D2Array, [7, 11, 3], [32, 16, 1],),
            (3, 1, [7, 11, 0]),
        );
    }

    #[test]
    fn volume_upload_keeps_z_as_a_region_coordinate() {
        assert_eq!(
            texture_region_coordinates(GlTextureDimension::D3, [7, 11, 3], [32, 16, 2],),
            (0, 1, [7, 11, 3]),
        );
    }
}
