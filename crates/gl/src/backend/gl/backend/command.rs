//! Immediate encoder creation and explicit resource transition handling for GL.
//!
//! Transfer commands live in `command/transfer.rs`; neither module retains a
//! deferred command list.

use fluxel_rhi_core::api::{
    self, CommandBackend, Error, LaneId, ResourceTransition, Result, TexelCopyBufferLayout,
    TextureRegion,
};

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
use super::spi::native_error;
use super::{Gl, GlDevice, types::*};

mod transfer;

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
        transfer::copy_buffer(encoder, src, src_offset, dst, dst_offset, bytes)
    }

    fn copy_texture(
        encoder: &mut GlEncoder,
        src: TextureRegion<'_, Gl>,
        dst: TextureRegion<'_, Gl>,
    ) -> Result<()> {
        transfer::copy_texture(encoder, src, dst)
    }

    fn blit(
        encoder: &mut GlEncoder,
        src: TextureRegion<'_, Gl>,
        dst: TextureRegion<'_, Gl>,
        filter: api::FilterMode,
    ) -> Result<()> {
        transfer::blit(encoder, src, dst, filter)
    }

    fn buffer_to_texture(
        encoder: &mut GlEncoder,
        src: &GlBuffer,
        layout: TexelCopyBufferLayout,
        dst: TextureRegion<'_, Gl>,
    ) -> Result<()> {
        transfer::buffer_to_texture(encoder, src, layout, dst)
    }

    fn texture_to_buffer(
        encoder: &mut GlEncoder,
        src: TextureRegion<'_, Gl>,
        dst: &GlBuffer,
        layout: TexelCopyBufferLayout,
    ) -> Result<()> {
        transfer::texture_to_buffer(encoder, src, dst, layout)
    }

    fn write_buffer(
        encoder: &mut GlEncoder,
        dst: &GlBuffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()> {
        transfer::write_buffer(encoder, dst, offset, bytes)
    }

    fn write_texture(
        encoder: &mut GlEncoder,
        dst: TextureRegion<'_, Gl>,
        layout: TexelCopyBufferLayout,
        bytes: &[u8],
    ) -> Result<()> {
        transfer::write_texture(encoder, dst, layout, bytes)
    }
}
