//! Immediate native GL transfer commands.
//!
//! Commands carry only resolved owner-table names and scalar regions.  They are
//! executed while the encoder is open; this module never receives a recorded
//! payload and never creates a submission/fence.

use super::driver::{NativeOwnedProvider, NativePlatformContext, map_gl};
use crate::api::command::BlitFilter;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::query::QueryType;
use crate::api::resource::transfer::{ReadbackTexelLayout, ReadbackTicket};
use crate::api::resource::{
    BufferRange, Extent3d, Origin3d, TextureAspect, TextureSubresourceLayers,
};
use crate::backend::gl::api::{
    GlAttachmentTarget, GlBlitRegion, GlBufferRange, GlCopyDomainApi as _, GlExtent3d,
    GlFilterMode, GlPixelLayout, GlTextureAspect, GlTextureRegion, GlTextureSubresource,
    GlTextureView,
};
use crate::backend::gl::platform::GlObjectName;

/// Owned transfer operands after public-handle resolution.
pub(super) enum Command {
    CopyBuffer {
        source: GlObjectName,
        source_range: BufferRange,
        destination: GlObjectName,
        destination_range: BufferRange,
    },
    CopyTexture {
        source: TextureRegion,
        destination: TextureRegion,
    },
    /// CPU staging is transient owner-thread data: this is immediate encoder
    /// lowering, never a deferred portable packet.
    CopyBufferToTexture {
        source: GlObjectName,
        source_range: BufferRange,
        destination: TextureRegion,
        layout: GlPixelLayout,
    },
    CopyTextureToBuffer {
        source: TextureRegion,
        destination: GlObjectName,
        destination_range: BufferRange,
        layout: GlPixelLayout,
    },
    UploadBuffer {
        destination: GlObjectName,
        range: BufferRange,
        bytes: Vec<u8>,
    },
    UploadTexture {
        destination: TextureRegion,
        layout: GlPixelLayout,
        bytes: Vec<u8>,
    },
    ReadBuffer {
        source: GlObjectName,
        range: BufferRange,
        ticket: ReadbackTicket,
    },
    ReadTexture {
        source: TextureRegion,
        layout: GlPixelLayout,
        ticket: ReadbackTicket,
    },
    ResolveTexture {
        source: TextureRegion,
        destination: TextureRegion,
    },
    BlitTexture {
        source: TextureRegion,
        destination: TextureRegion,
        filter: BlitFilter,
    },
    ClearBuffer,
    ClearTexture,
    CopyExternalImageToTexture,
    QueryBegin {
        set: GlObjectName,
        index: u32,
    },
    QueryEnd {
        set: GlObjectName,
        index: u32,
    },
    WriteTimestamp {
        set: GlObjectName,
        index: u32,
    },
    ResolveQuerySet {
        set: GlObjectName,
        first_query: u32,
        query_count: u32,
        destination: GlObjectName,
        destination_offset: u64,
    },
}

/// A texture region after public texture handles have been resolved.
#[derive(Clone, Copy)]
pub(super) struct TextureRegion {
    pub texture: GlObjectName,
    pub layers: TextureSubresourceLayers,
    pub origin: Origin3d,
    pub extent: Extent3d,
}

/// Executes a resolved transfer directly on the context-owning worker.
pub(super) fn execute<C: NativePlatformContext>(
    owner: &mut NativeOwnedProvider<C>,
    command: Command,
) -> RhiResult<()> {
    match command {
        Command::CopyBuffer {
            source,
            source_range,
            destination,
            destination_range,
        } => {
            const OP: &str = "NativeGlDriver::typed copy-buffer";
            owner.ready(OP)?;
            let source = GlBufferRange {
                buffer: owner.buffer_id(source, OP)?,
                offset: source_range.offset,
                size: source_range.size,
            };
            let destination = GlBufferRange {
                buffer: owner.buffer_id(destination, OP)?,
                offset: destination_range.offset,
                size: destination_range.size,
            };
            owner
                .provider
                .copy_buffer_range(source, destination)
                .map_err(|error| map_gl(error, OP))
        }
        Command::CopyTexture {
            source,
            destination,
        } => {
            const OP: &str = "NativeGlDriver::typed copy-texture";
            owner.ready(OP)?;
            owner
                .provider
                .copy_texture_region(
                    texture_region(owner, source, OP)?,
                    texture_region(owner, destination, OP)?,
                )
                .map_err(|error| map_gl(error, OP))
        }
        Command::CopyBufferToTexture {
            source,
            source_range,
            destination,
            layout,
        } => {
            const OP: &str = "NativeGlDriver::typed copy-buffer-to-texture";
            owner.ready(OP)?;
            let source = buffer_range(owner, source, source_range, OP)?;
            let destination = texture_region(owner, destination, OP)?;
            let bytes = owner
                .provider
                .read_buffer(source)
                .map_err(|error| map_gl(error, OP))?;
            owner
                .provider
                .upload_texture(destination, layout, &bytes)
                .map_err(|error| map_gl(error, OP))
        }
        Command::CopyTextureToBuffer {
            source,
            destination,
            destination_range,
            layout,
        } => {
            const OP: &str = "NativeGlDriver::typed copy-texture-to-buffer";
            owner.ready(OP)?;
            let source = texture_region(owner, source, OP)?;
            let destination = buffer_range(owner, destination, destination_range, OP)?;
            let bytes = owner
                .provider
                .read_texture(source, layout)
                .map_err(|error| map_gl(error, OP))?;
            owner
                .provider
                .upload_buffer(destination, &bytes.bytes)
                .map_err(|error| map_gl(error, OP))
        }
        Command::UploadBuffer {
            destination,
            range,
            bytes,
        } => {
            const OP: &str = "NativeGlDriver::typed upload-buffer";
            owner.ready(OP)?;
            owner
                .provider
                .upload_buffer(buffer_range(owner, destination, range, OP)?, &bytes)
                .map_err(|error| map_gl(error, OP))
        }
        Command::UploadTexture {
            destination,
            layout,
            bytes,
        } => {
            const OP: &str = "NativeGlDriver::typed upload-texture";
            owner.ready(OP)?;
            owner
                .provider
                .upload_texture(texture_region(owner, destination, OP)?, layout, &bytes)
                .map_err(|error| map_gl(error, OP))
        }
        Command::ReadBuffer {
            source,
            range,
            ticket,
        } => {
            const OP: &str = "NativeGlDriver::typed read-buffer";
            owner.ready(OP)?;
            let bytes = owner
                .provider
                .read_buffer(buffer_range(owner, source, range, OP)?)
                .map_err(|error| map_gl(error, OP))?;
            owner.retain_typed_readback(ticket, bytes, None);
            Ok(())
        }
        Command::ReadTexture {
            source,
            layout,
            ticket,
        } => {
            const OP: &str = "NativeGlDriver::typed read-texture";
            owner.ready(OP)?;
            let result = owner
                .provider
                .read_texture(texture_region(owner, source, OP)?, layout)
                .map_err(|error| map_gl(error, OP))?;
            let layout = ReadbackTexelLayout {
                bytes_per_row: result.layout.bytes_per_row,
                rows_per_image: result.layout.rows_per_image,
                total_size: result.bytes.len() as u64,
            };
            owner.retain_typed_readback(ticket, result.bytes, Some(layout));
            Ok(())
        }
        Command::ResolveTexture {
            source,
            destination,
        } => blit(
            owner,
            source,
            destination,
            GlFilterMode::Nearest,
            "resolve-texture",
        ),
        Command::BlitTexture {
            source,
            destination,
            filter,
        } => {
            let filter = match filter {
                BlitFilter::Nearest => GlFilterMode::Nearest,
                BlitFilter::Linear => GlFilterMode::Linear,
                _ => return unsupported("unknown texture blit filter"),
            };
            blit(owner, source, destination, filter, "blit-texture")
        }
        Command::ClearBuffer => unsupported("buffer clear"),
        Command::ClearTexture => unsupported("texture clear"),
        Command::CopyExternalImageToTexture => unsupported("external image copy"),
        Command::QueryBegin { set, index } => {
            const OP: &str = "NativeGlDriver::typed query-begin";
            owner.ready(OP)?;
            let (query, ty) = owner.query_id(set, index, None, OP)?;
            let state_key = owner.canonical(OP)?;
            owner.execute_query_begin(query, ty, state_key)
        }
        Command::QueryEnd { set, index } => {
            const OP: &str = "NativeGlDriver::typed query-end";
            owner.ready(OP)?;
            let (_, ty) = owner.query_id(set, index, None, OP)?;
            owner.execute_query_end(ty)
        }
        Command::WriteTimestamp { set, index } => {
            const OP: &str = "NativeGlDriver::typed write-timestamp";
            owner.ready(OP)?;
            let (query, _) = owner.query_id(set, index, Some(QueryType::Timestamp), OP)?;
            owner.execute_timestamp(query)
        }
        Command::ResolveQuerySet {
            set,
            first_query,
            query_count,
            destination,
            destination_offset,
        } => {
            const OP: &str = "NativeGlDriver::typed resolve-query-set";
            owner.ready(OP)?;
            let end = first_query.checked_add(query_count).ok_or_else(|| {
                RhiError::new(RhiErrorKind::InvalidUsage, "query resolve range overflows").at(OP)
            })?;
            let mut queries = Vec::with_capacity(query_count as usize);
            for index in first_query..end {
                let (query, ty) = owner.query_id(set, index, Some(QueryType::Occlusion), OP)?;
                if ty != QueryType::Occlusion {
                    return unsupported("non-occlusion query resolve");
                }
                queries.push(query);
            }
            let size = u64::from(query_count).checked_mul(8).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "query resolve byte count overflows",
                )
                .at(OP)
            })?;
            owner.execute_query_resolve(
                queries,
                GlBufferRange {
                    buffer: owner.buffer_id(destination, OP)?,
                    offset: destination_offset,
                    size,
                },
            )
        }
    }
}

fn unsupported(route: &'static str) -> RhiResult<()> {
    Err(unsupported_error(route))
}

fn unsupported_error(route: &'static str) -> RhiError {
    RhiError::new(
        RhiErrorKind::Unsupported,
        format!("native GL has no proved typed {route} route"),
    )
    .at("NativeGlDriver::typed copy")
}

fn buffer_range<C: NativePlatformContext>(
    owner: &NativeOwnedProvider<C>,
    name: GlObjectName,
    range: BufferRange,
    operation: &'static str,
) -> RhiResult<GlBufferRange> {
    Ok(GlBufferRange {
        buffer: owner.buffer_id(name, operation)?,
        offset: range.offset,
        size: range.size,
    })
}

fn texture_region<C: NativePlatformContext>(
    owner: &NativeOwnedProvider<C>,
    region: TextureRegion,
    operation: &'static str,
) -> RhiResult<GlTextureRegion> {
    let aspect = match region.layers.aspect {
        TextureAspect::Color => GlTextureAspect::Color,
        TextureAspect::Depth => GlTextureAspect::DepthOnly,
        TextureAspect::Stencil => GlTextureAspect::StencilOnly,
        TextureAspect::Plane0 | TextureAspect::Plane1 | TextureAspect::Plane2 => {
            return Err(unsupported_error("multi-planar texture transfer"));
        }
        _ => return Err(unsupported_error("unknown texture aspect")),
    };
    Ok(GlTextureRegion {
        subresource: GlTextureSubresource {
            texture: owner.texture_id(region.texture, operation)?,
            aspect,
            mip_level: region.layers.mip_level,
            base_layer: region.layers.base_layer,
            layer_count: region.layers.layer_count,
        },
        origin: [region.origin.x, region.origin.y, region.origin.z],
        extent: GlExtent3d {
            width: region.extent.width,
            height: region.extent.height,
            depth_or_layers: region.extent.depth,
        },
    })
}

fn blit<C: NativePlatformContext>(
    owner: &mut NativeOwnedProvider<C>,
    source: TextureRegion,
    destination: TextureRegion,
    filter: GlFilterMode,
    operation: &'static str,
) -> RhiResult<()> {
    const OP: &str = "NativeGlDriver::typed texture-blit";
    owner.ready(OP)?;
    let source_view = texture_view(owner, source, operation)?;
    let destination_view = texture_view(owner, destination, operation)?;
    owner.execute_texture_blit(
        source_view,
        destination_view,
        GlBlitRegion {
            src_offset: [source.origin.x, source.origin.y],
            src_extent: [source.extent.width, source.extent.height],
            dst_offset: [destination.origin.x, destination.origin.y],
            dst_extent: [destination.extent.width, destination.extent.height],
        },
        filter,
    )
}

fn texture_view<C: NativePlatformContext>(
    owner: &NativeOwnedProvider<C>,
    region: TextureRegion,
    operation: &'static str,
) -> RhiResult<GlTextureView> {
    if region.layers.aspect != TextureAspect::Color || region.layers.layer_count != 1 {
        return Err(unsupported_error("non-color or multi-layer texture blit"));
    }
    let texture = owner.texture_id(region.texture, operation)?;
    let (_, descriptor) = owner
        .provider
        .texture(operation, texture)
        .map_err(|error| map_gl(error, operation))?;
    let extent = descriptor
        .mip_extent(region.layers.mip_level)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture blit mip is outside its allocation",
            )
            .at(operation)
        })?;
    Ok(GlTextureView {
        target: GlAttachmentTarget::Texture(texture),
        format: descriptor.format,
        mip_level: region.layers.mip_level,
        array_layer: region.layers.base_layer,
        layer_count: 1,
        width: extent.width,
        height: extent.height,
        sample_count: descriptor.sample_count,
    })
}
