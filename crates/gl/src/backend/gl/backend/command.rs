//! Direct-copy and upload command SPI for the immediate GL backend.
//!
//! Each accepted verb calls the current context provider before returning.
//! The encoder keeps only issued/indeterminate facts required by immediate
//! close and completion, never a deferred command list.

use std::rc::Rc;

use fluxel_rhi_core::api::{
    self, CommandBackend, Error, LaneId, ResourceTransition, Result, TexelCopyBufferLayout,
    TextureRegion,
};

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
use super::spi::native_error;
use super::{Gl, GlDevice, types::*};

/// Lowers a Core texture region without changing the public coordinate system.
/// Native and browser executors perform their profile-specific Y conversion at
/// the call that touches GL, exactly once.
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
fn lower_region(value: TextureRegion<'_, Gl>) -> Result<crate::backend::gl::api::GlTextureRegion> {
    // SAFETY: CommandBackend receives this region from Core while the Texture
    // borrow is live. The native reference is used synchronously only to read
    // immutable backend allocation identity and descriptor for lowering.
    let texture = unsafe { value.texture.native_ref() };
    crate::backend::gl::translate::lower_texture_region(texture.id, texture.desc, &value)
}

impl CommandBackend for Gl {
    fn begin_encoder(device: &GlDevice, lane: LaneId) -> Result<GlEncoder> {
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        device
            .0
            .with_native("begin immediate GL encoder", |provider| {
                use crate::backend::gl::api::GlFamilyApi as _;
                provider.assert_ready("begin immediate GL encoder")
            })
            .map_err(native_error)?;
        let mut encoder = GlEncoder::new(device.0.clone(), lane.device);
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            encoder.browser = Some(device.0.owner.begin_encoder(lane)?);
        }
        Ok(encoder)
    }

    fn transition(encoder: &mut GlEncoder, ops: &[ResourceTransition<'_, Gl>]) -> Result<()> {
        let mut image_write_to_sample = false;
        for op in ops {
            match op {
                ResourceTransition::Texture(barrier) => {
                    // A repeated attachment state is common across immediate
                    // frames. GL command ordering already orders these uses;
                    // accepting this exact no-op avoids inventing layout
                    // history or emitting a needless native barrier.
                    let unchanged_state = barrier.ownership.is_none()
                        && barrier
                            .layout
                            .is_some_and(|(before, after)| before == after)
                        && barrier.usage.from == barrier.usage.to
                        && barrier.usage.from.accesses.0 & api::AccessMask::SHADER_WRITE.0 == 0;
                    let enter_color_attachment = barrier.ownership.is_none()
                        && ((barrier.layout
                            == Some((
                                api::ImageLayout::Undefined,
                                api::ImageLayout::ColorAttachment,
                            ))
                            && barrier.usage.from.stages == api::StageMask::EMPTY
                            && barrier.usage.from.accesses == api::AccessMask::EMPTY)
                            || (barrier.layout
                                == Some((
                                    api::ImageLayout::ShaderReadOnly,
                                    api::ImageLayout::ColorAttachment,
                                ))
                                && barrier.usage.from.stages == api::StageMask::FRAGMENT
                                && barrier.usage.from.accesses == api::AccessMask::SHADER_READ))
                        && barrier.usage.to.stages == api::StageMask::COLOR_OUTPUT
                        && barrier.usage.to.accesses == api::AccessMask::COLOR_WRITE;
                    let framebuffer_write_to_sample = barrier.ownership.is_none()
                        && barrier.layout
                            == Some((
                                api::ImageLayout::ColorAttachment,
                                api::ImageLayout::ShaderReadOnly,
                            ))
                        && barrier.usage.from.stages == api::StageMask::COLOR_OUTPUT
                        && barrier.usage.from.accesses == api::AccessMask::COLOR_WRITE
                        && barrier.usage.to.stages == api::StageMask::FRAGMENT
                        && barrier.usage.to.accesses == api::AccessMask::SHADER_READ;
                    let upload_to_sample = barrier.ownership.is_none()
                        && barrier.layout
                            == Some((api::ImageLayout::Common, api::ImageLayout::ShaderReadOnly))
                        && barrier
                            .usage
                            .to
                            .accesses
                            .contains(api::AccessMask::SHADER_READ)
                        && barrier.usage.to.accesses.0 & api::AccessMask::SHADER_WRITE.0 == 0;
                    let sample_to_common = barrier.ownership.is_none()
                        && barrier.layout
                            == Some((api::ImageLayout::ShaderReadOnly, api::ImageLayout::Common))
                        && barrier
                            .usage
                            .from
                            .accesses
                            .contains(api::AccessMask::SHADER_READ)
                        && barrier.usage.from.accesses.0 & api::AccessMask::SHADER_WRITE.0 == 0
                        && barrier.usage.to.accesses == api::AccessMask::EMPTY;
                    let enter_storage_write = barrier.ownership.is_none()
                        && ((barrier.layout
                            == Some((api::ImageLayout::Undefined, api::ImageLayout::General))
                            && barrier.usage.from.stages == api::StageMask::HOST
                            && barrier.usage.from.accesses == api::AccessMask::EMPTY)
                            || (barrier.layout
                                == Some((
                                    api::ImageLayout::ShaderReadOnly,
                                    api::ImageLayout::General,
                                ))
                                && barrier.usage.from.stages == api::StageMask::FRAGMENT
                                && barrier
                                    .usage
                                    .from
                                    .accesses
                                    .contains(api::AccessMask::SHADER_READ)))
                        && barrier.usage.to.stages == api::StageMask::COMPUTE
                        && barrier
                            .usage
                            .to
                            .accesses
                            .contains(api::AccessMask::SHADER_WRITE);
                    let leave_storage_write = barrier.ownership.is_none()
                        && barrier.layout
                            == Some((api::ImageLayout::General, api::ImageLayout::ShaderReadOnly))
                        && barrier
                            .usage
                            .from
                            .accesses
                            .contains(api::AccessMask::SHADER_WRITE)
                        && barrier.usage.from.stages == api::StageMask::COMPUTE
                        && barrier.usage.to.stages == api::StageMask::FRAGMENT
                        && barrier
                            .usage
                            .to
                            .accesses
                            .contains(api::AccessMask::SHADER_READ);
                    // The 09 mip chain supplies these exact per-level copy
                    // boundaries. GL issues upload and framebuffer blit on one
                    // immediate context, so command order is the visibility
                    // mechanism; this accepts no inferred resource history.
                    let upload_to_copy_source = barrier.ownership.is_none()
                        && barrier.layout
                            == Some((api::ImageLayout::Common, api::ImageLayout::CopySrc))
                        && barrier.usage.from.stages == api::StageMask::COPY
                        && barrier.usage.from.accesses == api::AccessMask::COPY_WRITE
                        && barrier.usage.to.stages == api::StageMask::COPY
                        && barrier.usage.to.accesses == api::AccessMask::COPY_READ;
                    let undefined_to_copy_destination = barrier.ownership.is_none()
                        && barrier.layout
                            == Some((api::ImageLayout::Undefined, api::ImageLayout::CopyDst))
                        && barrier.usage.from.stages == api::StageMask::EMPTY
                        && barrier.usage.from.accesses == api::AccessMask::EMPTY
                        && barrier.usage.to.stages == api::StageMask::COPY
                        && barrier.usage.to.accesses == api::AccessMask::COPY_WRITE;
                    let copy_destination_to_source = barrier.ownership.is_none()
                        && barrier.layout
                            == Some((api::ImageLayout::CopyDst, api::ImageLayout::CopySrc))
                        && barrier.usage.from.stages == api::StageMask::COPY
                        && barrier.usage.from.accesses == api::AccessMask::COPY_WRITE
                        && barrier.usage.to.stages == api::StageMask::COPY
                        && barrier.usage.to.accesses == api::AccessMask::COPY_READ;
                    let attachment_to_copy_source = barrier.ownership.is_none()
                        && barrier.layout
                            == Some((api::ImageLayout::ColorAttachment, api::ImageLayout::CopySrc))
                        && barrier.usage.from.stages == api::StageMask::COLOR_OUTPUT
                        && barrier.usage.from.accesses == api::AccessMask::COLOR_WRITE
                        && barrier.usage.to.stages == api::StageMask::COPY
                        && barrier.usage.to.accesses == api::AccessMask::COPY_READ;
                    let copy_source_to_attachment = barrier.ownership.is_none()
                        && barrier.layout
                            == Some((api::ImageLayout::CopySrc, api::ImageLayout::ColorAttachment))
                        && barrier.usage.from.stages == api::StageMask::COPY
                        && barrier.usage.from.accesses == api::AccessMask::COPY_READ
                        && barrier.usage.to.stages == api::StageMask::COLOR_OUTPUT
                        && barrier.usage.to.accesses == api::AccessMask::COLOR_WRITE;
                    let copy_to_sample = barrier.ownership.is_none()
                        && ((barrier.layout
                            == Some((
                                api::ImageLayout::CopySrc,
                                api::ImageLayout::ShaderReadOnly,
                            ))
                            && barrier.usage.from.stages == api::StageMask::COPY
                            && barrier.usage.from.accesses == api::AccessMask::COPY_READ)
                            || (barrier.layout
                                == Some((
                                    api::ImageLayout::CopyDst,
                                    api::ImageLayout::ShaderReadOnly,
                                ))
                                && barrier.usage.from.stages == api::StageMask::COPY
                                && barrier.usage.from.accesses == api::AccessMask::COPY_WRITE))
                        && barrier.usage.to.stages == api::StageMask::FRAGMENT
                        && barrier.usage.to.accesses == api::AccessMask::SHADER_READ;
                    image_write_to_sample |= leave_storage_write;
                    if !unchanged_state
                        && !enter_color_attachment
                        && !upload_to_sample
                        && !sample_to_common
                        && !enter_storage_write
                        && !leave_storage_write
                        && !framebuffer_write_to_sample
                        && !upload_to_copy_source
                        && !undefined_to_copy_destination
                        && !copy_destination_to_source
                        && !attachment_to_copy_source
                        && !copy_source_to_attachment
                        && !copy_to_sample
                    {
                        return Err(Error::Unsupported("GL texture transition"));
                    }
                }
                // Host API writes, framebuffer output followed by a later
                // texture-fetch command, and same-context vertex/UBO reads are
                // ordered by GL itself. Keep this narrow: shader writes and
                // cross-lane ownership need a real glMemoryBarrier/sync route
                // and must not be silently accepted as a no-op.
                ResourceTransition::Buffer(barrier)
                    if barrier.ownership.is_none()
                        && barrier.usage.to.accesses.0 & api::AccessMask::SHADER_WRITE.0 == 0
                        && barrier.usage.from.accesses.0 & api::AccessMask::SHADER_WRITE.0 == 0 => {
                }
                _ => return Err(Error::Unsupported("GL resource transition kind")),
            }
        }
        // GL 4.6 core §7.13.2 specifies that API texture updates such as
        // TexSubImage are automatically synchronized with later GL commands.
        // Thus this exact upload-to-sampling edge needs no glMemoryBarrier on
        // the same immediate context. The planned storage-write-to-sampling
        // edge is different: `glMemoryBarrier` makes shader-image stores
        // visible to later texture fetches without tracking GL image layouts.
        if image_write_to_sample {
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            {
                use crate::backend::gl::api::{GlMemoryBarrier, GlSyncApi as _};
                let result = encoder
                    .state
                    .with_native("make GL storage image visible to sampling", |provider| {
                        provider.memory_barrier(GlMemoryBarrier::TEXTURE_FETCH)
                    })
                    .map_err(native_error);
                if result.is_ok() {
                    encoder.issued = true;
                } else {
                    encoder.indeterminate = true;
                }
                return result;
            }
            #[cfg(not(any(
                all(windows, feature = "native-gl-wgl"),
                feature = "native-gles-egl"
            )))]
            return Err(Error::Unsupported("GL storage image transition"));
        }
        Ok(())
    }

    fn copy_buffer(
        encoder: &mut GlEncoder,
        src: &GlBuffer,
        src_offset: u64,
        dst: &GlBuffer,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<()> {
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::{GlBufferRange, GlCopyDomainApi as _};
            let result = encoder
                .state
                .with_native("copy GL buffer", |provider| {
                    provider.copy_buffer_range(
                        GlBufferRange {
                            buffer: src.id,
                            offset: src_offset,
                            size: bytes,
                        },
                        GlBufferRange {
                            buffer: dst.id,
                            offset: dst_offset,
                            size: bytes,
                        },
                    )
                })
                .map_err(native_error);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
            let result = browser.copy_browser_buffer(
                &src.browser,
                src_offset,
                &dst.browser,
                dst_offset,
                bytes,
            );
            if result.is_ok() {
                encoder.issued = true;
            } else {
                // The browser may report an error after it accepted a GL call.
                // Core must never classify that prefix as a clean rejection.
                encoder.indeterminate = true;
            }
            result
        }
        #[cfg(not(any(
            all(windows, feature = "native-gl-wgl"),
            feature = "native-gles-egl",
            all(target_arch = "wasm32", feature = "webgl2")
        )))]
        Err(Error::Unsupported("GL platform context"))
    }

    fn copy_texture(
        encoder: &mut GlEncoder,
        src: TextureRegion<'_, Gl>,
        dst: TextureRegion<'_, Gl>,
    ) -> Result<()> {
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::GlCopyDomainApi as _;
            let src = lower_region(src)?;
            let dst = lower_region(dst)?;
            let result = encoder
                .state
                .with_native("copy GL texture", |provider| {
                    provider.copy_texture_region(src, dst)
                })
                .map_err(native_error);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            // WebGL2 has no texture-copy entry point. Its owner implements this
            // exact-size copy with short-lived FBOs and a nearest framebuffer
            // blit. Keep the operation inside this synchronous encoder call.
            // SAFETY: Core retains both Texture borrows for the duration of
            // this function; each browser wrapper is only inspected/used now.
            let source = unsafe { src.texture.native_ref() };
            // SAFETY: same Core borrow guarantee as for `source`.
            let destination = unsafe { dst.texture.native_ref() };
            if src.aspect != api::TextureAspects::COLOR || dst.aspect != api::TextureAspects::COLOR
            {
                return Err(Error::Unsupported(
                    "WebGL2 texture copy supports color aspects only",
                ));
            }
            if src.origin[2] != 0
                || dst.origin[2] != 0
                || src.extent[2] != 1
                || dst.extent[2] != 1
                || src.extent != dst.extent
            {
                return Err(Error::InvalidRange);
            }
            let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
            let result = browser.blit_texture_regions(
                &source.browser,
                src.mip,
                [src.origin[0], src.origin[1]],
                [src.extent[0], src.extent[1]],
                &destination.browser,
                dst.mip,
                [dst.origin[0], dst.origin[1]],
                [dst.extent[0], dst.extent[1]],
                crate::backend::gl::api::GlFilterMode::Nearest,
            );
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(not(any(
            all(windows, feature = "native-gl-wgl"),
            feature = "native-gles-egl",
            all(target_arch = "wasm32", feature = "webgl2")
        )))]
        Err(Error::Unsupported("GL platform context"))
    }

    fn blit(
        encoder: &mut GlEncoder,
        src: TextureRegion<'_, Gl>,
        dst: TextureRegion<'_, Gl>,
        filter: api::FilterMode,
    ) -> Result<()> {
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::{GlFramebufferApi as _, GlTextureUsage};

            // SAFETY: Core keeps both texture handles borrowed for this
            // synchronous command; the descriptors and IDs are immutable.
            let source = unsafe { src.texture.native_ref() };
            // SAFETY: same synchronous borrow guarantee as for `source`.
            let destination = unsafe { dst.texture.native_ref() };
            let source_desc = source.desc;
            let destination_desc = destination.desc;
            let source_region =
                crate::backend::gl::translate::lower_texture_region(source.id, source_desc, &src)?;
            let destination_region = crate::backend::gl::translate::lower_texture_region(
                destination.id,
                destination_desc,
                &dst,
            )?;
            if source_desc.format != destination_desc.format
                || source_desc.sample_count != 1
                || destination_desc.sample_count != 1
                || source_desc.dimension != crate::backend::gl::api::GlTextureDimension::D2
                || destination_desc.dimension != crate::backend::gl::api::GlTextureDimension::D2
                || !source_desc.usage.contains(GlTextureUsage::COPY_SOURCE)
                || !destination_desc
                    .usage
                    .contains(GlTextureUsage::COPY_DESTINATION)
                || source_region.subresource.aspect
                    != crate::backend::gl::api::GlTextureAspect::Color
                || destination_region.subresource.aspect
                    != crate::backend::gl::api::GlTextureAspect::Color
                || source_region.subresource.base_layer != 0
                || destination_region.subresource.base_layer != 0
                || source_region.subresource.layer_count != 1
                || destination_region.subresource.layer_count != 1
                || source_region.extent.depth_or_layers != 1
                || destination_region.extent.depth_or_layers != 1
            {
                return Err(Error::Unsupported("GL color texture blit shape or usage"));
            }
            let source_mip = source_desc
                .mip_extent(source_region.subresource.mip_level)
                .ok_or(Error::InvalidRange)?;
            let destination_mip = destination_desc
                .mip_extent(destination_region.subresource.mip_level)
                .ok_or(Error::InvalidRange)?;
            // Core rectangles use top-left coordinates. The two mip heights
            // differ, so convert each endpoint with its own allocation facts.
            let source_y = source_mip
                .height
                .checked_sub(
                    source_region.origin[1]
                        .checked_add(source_region.extent.height)
                        .ok_or(Error::InvalidRange)?,
                )
                .ok_or(Error::InvalidRange)?;
            let destination_y = destination_mip
                .height
                .checked_sub(
                    destination_region.origin[1]
                        .checked_add(destination_region.extent.height)
                        .ok_or(Error::InvalidRange)?,
                )
                .ok_or(Error::InvalidRange)?;
            let source_view = crate::backend::gl::api::GlTextureView {
                target: crate::backend::gl::api::GlAttachmentTarget::Texture(source.id),
                format: source_desc.format,
                mip_level: source_region.subresource.mip_level,
                array_layer: 0,
                layer_count: 1,
                width: source_mip.width,
                height: source_mip.height,
                sample_count: 1,
            };
            let destination_view = crate::backend::gl::api::GlTextureView {
                target: crate::backend::gl::api::GlAttachmentTarget::Texture(destination.id),
                format: destination_desc.format,
                mip_level: destination_region.subresource.mip_level,
                array_layer: 0,
                layer_count: 1,
                width: destination_mip.width,
                height: destination_mip.height,
                sample_count: 1,
            };
            let source_framebuffer = crate::backend::gl::api::GlFramebufferDescriptor {
                color_attachments: vec![source_view],
                depth_stencil_attachment: None,
                draw_buffers: vec![0],
            };
            let destination_framebuffer = crate::backend::gl::api::GlFramebufferDescriptor {
                color_attachments: vec![destination_view],
                depth_stencil_attachment: None,
                draw_buffers: vec![0],
            };
            let region = crate::backend::gl::api::GlBlitRegion {
                src_offset: [source_region.origin[0], source_y],
                src_extent: [source_region.extent.width, source_region.extent.height],
                dst_offset: [destination_region.origin[0], destination_y],
                dst_extent: [
                    destination_region.extent.width,
                    destination_region.extent.height,
                ],
            };
            let filter = match filter {
                api::FilterMode::Nearest => crate::backend::gl::api::GlFilterMode::Nearest,
                api::FilterMode::Linear => crate::backend::gl::api::GlFilterMode::Linear,
            };
            let result = encoder
                .state
                .with_native("blit GL texture mip", |provider| {
                    let source_fbo = provider.create_framebuffer(&source_framebuffer)?;
                    let destination_fbo =
                        match provider.create_framebuffer(&destination_framebuffer) {
                            Ok(framebuffer) => framebuffer,
                            Err(error) => {
                                let _ = provider.destroy_framebuffer(source_fbo);
                                return Err(error);
                            }
                        };
                    let blit = provider.blit_framebuffer(
                        source_fbo,
                        destination_fbo,
                        region,
                        filter,
                        crate::backend::gl::api::GlBlitMask {
                            color: true,
                            depth: false,
                            stencil: false,
                        },
                    );
                    let drop_destination = provider.destroy_framebuffer(destination_fbo);
                    let drop_source = provider.destroy_framebuffer(source_fbo);
                    blit.and(drop_destination).and(drop_source)
                })
                .map_err(native_error);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            // SAFETY: Core holds both texture borrows through this synchronous
            // blit. The browser wrappers are used immediately and never retained.
            let source = unsafe { src.texture.native_ref() };
            let destination = unsafe { dst.texture.native_ref() };
            if src.aspect != api::TextureAspects::COLOR || dst.aspect != api::TextureAspects::COLOR
            {
                return Err(Error::Unsupported(
                    "WebGL2 texture blit supports color aspects only",
                ));
            }
            if src.origin[2] != 0 || dst.origin[2] != 0 || src.extent[2] != 1 || dst.extent[2] != 1
            {
                return Err(Error::InvalidRange);
            }
            let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
            let result = browser.blit_texture_regions(
                &source.browser,
                src.mip,
                [src.origin[0], src.origin[1]],
                [src.extent[0], src.extent[1]],
                &destination.browser,
                dst.mip,
                [dst.origin[0], dst.origin[1]],
                [dst.extent[0], dst.extent[1]],
                match filter {
                    api::FilterMode::Nearest => crate::backend::gl::api::GlFilterMode::Nearest,
                    api::FilterMode::Linear => crate::backend::gl::api::GlFilterMode::Linear,
                },
            );
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(not(any(
            all(windows, feature = "native-gl-wgl"),
            feature = "native-gles-egl",
            all(target_arch = "wasm32", feature = "webgl2")
        )))]
        {
            let _ = (encoder, src, dst, filter);
            Err(Error::Unsupported("GL texture blit platform"))
        }
    }

    fn buffer_to_texture(
        _: &mut GlEncoder,
        _: &GlBuffer,
        _: TexelCopyBufferLayout,
        _: TextureRegion<'_, Gl>,
    ) -> Result<()> {
        Err(Error::Unsupported("GL buffer-to-texture copy"))
    }

    fn texture_to_buffer(
        encoder: &mut GlEncoder,
        src: TextureRegion<'_, Gl>,
        dst: &GlBuffer,
        layout: TexelCopyBufferLayout,
    ) -> Result<()> {
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::{
                GlBufferRange, GlBufferUsage, GlCopyDomainApi as _, GlTextureUsage,
            };

            // SAFETY: Core keeps the texture region borrowed for this direct
            // operation; the native reference contributes immutable allocation
            // metadata and is never retained beyond the current GL call.
            let source = unsafe { src.texture.native_ref() };
            if !Rc::ptr_eq(&encoder.state, &source.state)
                || !Rc::ptr_eq(&encoder.state, &dst.state)
                || !source.desc.usage.contains(GlTextureUsage::COPY_SOURCE)
                || !dst.desc.usage.contains(GlBufferUsage::COPY_DESTINATION)
            {
                return Err(Error::InvalidRange);
            }
            let source_region = lower_region(src)?;
            let pixel_layout =
                crate::backend::gl::translate::lower_texel_copy_layout(layout, source.desc.format)?;
            let required = pixel_layout
                .required_bytes(source_region)
                .map_err(|_| Error::InvalidRange)?;
            let write_size = required
                .checked_sub(layout.offset)
                .ok_or(Error::InvalidRange)?;
            let write_end = layout
                .offset
                .checked_add(write_size)
                .ok_or(Error::InvalidRange)?;
            if write_end > dst.desc.size {
                return Err(Error::InvalidRange);
            }

            let result = encoder
                .state
                .with_native("read GL texture into buffer", |provider| {
                    // `read_texture` performs the historical GL readPixels
                    // operation synchronously; upload the returned top-left
                    // bytes before this command call returns instead of
                    // retaining a CPU copy operation for later replay.
                    let readback = provider.read_texture(source_region, pixel_layout)?;
                    let start = usize::try_from(layout.offset).map_err(|_| {
                        crate::backend::gl::api::GlError::OutOfMemory {
                            operation: "read-texture-to-buffer",
                        }
                    })?;
                    let end = usize::try_from(required).map_err(|_| {
                        crate::backend::gl::api::GlError::OutOfMemory {
                            operation: "read-texture-to-buffer",
                        }
                    })?;
                    let bytes = readback.bytes.get(start..end).ok_or_else(|| {
                        crate::backend::gl::api::GlError::Validation {
                            operation: "read-texture-to-buffer",
                            message: "native readback returned a short pixel buffer".into(),
                        }
                    })?;
                    provider.upload_buffer(
                        GlBufferRange {
                            buffer: dst.id,
                            offset: layout.offset,
                            size: write_size,
                        },
                        bytes,
                    )
                })
                .map_err(native_error);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                // The pixel read or destination upload may have reached GL
                // before a later native error was observed.
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            // SAFETY: Core retains `src.texture` for the complete synchronous
            // immediate verb. The browser wrapper is borrowed only while
            // `readPixels` and the destination `bufferSubData` execute.
            let source = unsafe { src.texture.native_ref() };
            use crate::backend::gl::api::{
                GlBufferUsage, GlFormat, GlTextureDimension, GlTextureUsage,
            };
            if !Rc::ptr_eq(&encoder.state, &source.state)
                || !Rc::ptr_eq(&encoder.state, &dst.state)
                || !source.desc.usage.contains(GlTextureUsage::COPY_SOURCE)
                || !dst.desc.usage.contains(GlBufferUsage::COPY_DESTINATION)
                || !matches!(
                    source.desc.format,
                    GlFormat::Rgba8Unorm | GlFormat::Rgba8Srgb
                )
                || source.desc.dimension != GlTextureDimension::D2
                || src.aspect != api::TextureAspects::COLOR
            {
                return Err(Error::Unsupported(
                    "WebGL2 texture-to-buffer requires an RGBA8 2D color rectangle",
                ));
            }
            if src.origin[2] != 0 || src.extent[2] != 1 {
                return Err(Error::InvalidRange);
            }
            let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
            let result = browser.copy_rgba8_texture_to_buffer(
                &source.browser,
                src.mip,
                [src.origin[0], src.origin[1]],
                [src.extent[0], src.extent[1]],
                &dst.browser,
                layout.offset,
                layout.bytes_per_row,
                layout.rows_per_image,
            );
            if result.is_ok() {
                encoder.issued = true;
            } else {
                // readPixels may have reached the context before a following
                // buffer upload or error observation failed.
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(not(any(
            all(windows, feature = "native-gl-wgl"),
            feature = "native-gles-egl",
            all(target_arch = "wasm32", feature = "webgl2")
        )))]
        {
            let _ = (encoder, src, dst, layout);
            Err(Error::Unsupported("GL texture-to-buffer platform"))
        }
    }

    fn write_buffer(
        encoder: &mut GlEncoder,
        dst: &GlBuffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()> {
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::{GlBufferRange, GlCopyDomainApi as _};
            let size = u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?;
            let result = encoder
                .state
                .with_native("upload GL buffer", |provider| {
                    provider.upload_buffer(
                        GlBufferRange {
                            buffer: dst.id,
                            offset,
                            size,
                        },
                        bytes,
                    )
                })
                .map_err(native_error);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
            let result = browser.write_browser_buffer(&dst.browser, offset, bytes);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                // A browser error can arrive after BufferSubData reached GL.
                encoder.indeterminate = true;
            }
            result
        }
        #[cfg(not(any(
            all(windows, feature = "native-gl-wgl"),
            feature = "native-gles-egl",
            all(target_arch = "wasm32", feature = "webgl2")
        )))]
        Err(Error::Unsupported("GL platform context"))
    }

    fn write_texture(
        encoder: &mut GlEncoder,
        dst: TextureRegion<'_, Gl>,
        layout: TexelCopyBufferLayout,
        bytes: &[u8],
    ) -> Result<()> {
        // SAFETY: Core holds the texture borrow for this synchronous command;
        // its descriptor is immutable allocation metadata used only to choose
        // the validated native pixel-layout lowering.
        let texture = unsafe { dst.texture.native_ref() };
        let layout =
            crate::backend::gl::translate::lower_texel_copy_layout(layout, texture.desc.format)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            let destination = lower_region(dst)?;
            use crate::backend::gl::api::GlCopyDomainApi as _;
            let result = encoder
                .state
                .with_native("upload GL texture", |provider| {
                    provider.upload_texture(destination, layout, bytes)
                })
                .map_err(native_error);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            {
                let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
                let result = browser.write_texture_region(
                    &texture.browser,
                    dst.mip,
                    dst.origin,
                    dst.extent,
                    layout.bytes_per_row,
                    layout.rows_per_image,
                    bytes,
                );
                if result.is_ok() {
                    encoder.issued = true;
                } else {
                    encoder.indeterminate = true;
                }
                return result;
            }
            #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
            {
                let _ = (encoder, dst, layout, bytes);
                Err(Error::Unsupported("GL platform context"))
            }
        }
    }
}
