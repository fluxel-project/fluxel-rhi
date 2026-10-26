//! Immediate encoder creation and explicit resource transition handling for GL.
//!
//! Transfer commands live in `command/transfer.rs`; neither module retains a
//! deferred command list.

use fluxel_rhi_core::api::{
    self, Error, LaneId, ResourceTransition, Result, TexelCopyBufferLayout, TextureRegion,
};
use fluxel_rhi_core::backend::CommandBackend;

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
use super::spi::native_error;
use super::{Gl, GlDevice, types::*};

mod clear;
mod resolve;
mod transfer;
mod transition;

impl CommandBackend for Gl {
    fn clear_texture(
        encoder: &mut GlEncoder,
        destination: &GlTexture,
        mips: std::ops::Range<u32>,
        layers: std::ops::Range<u32>,
        value: api::TextureClearValue,
    ) -> Result<()> {
        clear::texture(encoder, destination, mips, layers, value)
    }
    fn clear_buffer(
        encoder: &mut GlEncoder,
        destination: &GlBuffer,
        range: std::ops::Range<u64>,
        value: u32,
    ) -> Result<()> {
        clear::buffer(encoder, destination, range, value)
    }
    fn begin_encoder(device: &GlDevice, lane: LaneId) -> Result<GlEncoder> {
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        device
            .0
            .with_native("begin immediate GL encoder", |provider| {
                use crate::backend::api::GlFamilyApi as _;
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
        transition::transition(encoder, ops)
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

    fn resolve(
        encoder: &mut GlEncoder,
        src: TextureRegion<'_, Gl>,
        dst: TextureRegion<'_, Gl>,
    ) -> Result<()> {
        resolve::texture(encoder, src, dst)
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
