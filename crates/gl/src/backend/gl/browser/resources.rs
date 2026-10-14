//! Browser-local resource and pipeline allocation.
//!
//! These constructors create WebGL objects immediately for one `BrowserOwner`.
//! They never cache commands or transfer JavaScript handles into Core.

use crate::backend::gl::api::GlFamilyApi as _;
use fluxel_rhi_core::api::{Error, Result};
use web_sys::WebGl2RenderingContext as Gl;

use super::owner::{
    BrowserBuffer, BrowserOwner, BrowserRasterProgram, BrowserSampler, BrowserTexture,
    BrowserVertexArray,
};

impl BrowserOwner {
    /// Allocates a dynamic byte-addressable buffer for the static backend.
    pub(crate) fn create_buffer(&self, bytes: u64) -> Result<BrowserBuffer> {
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        let raw = state
            .context
            .raw
            .create_buffer()
            .ok_or(Error::OutOfMemory)?;
        state.context.raw.bind_buffer(Gl::ARRAY_BUFFER, Some(&raw));
        state
            .context
            .raw
            .buffer_data_with_f64(Gl::ARRAY_BUFFER, bytes as f64, Gl::DYNAMIC_DRAW);
        if state.context.raw.get_error() != Gl::NO_ERROR {
            state.context.raw.delete_buffer(Some(&raw));
            return Err(Error::Backend("WebGL2 buffer allocation failed".into()));
        }
        Ok(BrowserBuffer {
            owner: self.clone(),
            raw,
            size: bytes,
        })
    }

    /// Allocates immutable browser texture storage from the already validated
    /// Core descriptor. The caller owns format/dimension admission; this owner
    /// selects only WebGL2's matching target and immutable-storage call.
    pub(crate) fn create_texture(
        &self,
        desc: super::super::api::GlTextureDesc,
    ) -> Result<BrowserTexture> {
        use super::super::api::GlTextureDimension;

        // The SPI can be called without Core's support probe. Validate the
        // complete descriptor before allocating a browser object so oversized
        // mip chains cannot reach WebGL as lossy integer arguments.
        desc.validate().map_err(|_| Error::InvalidRange)?;
        let target = match desc.dimension {
            GlTextureDimension::D2 => Gl::TEXTURE_2D,
            GlTextureDimension::D2Array => Gl::TEXTURE_2D_ARRAY,
            _ => return Err(Error::Unsupported("WebGL2 texture dimension")),
        };
        let internal = super::format_map::internal_format(desc.format)
            .ok_or(Error::Unsupported("WebGL2 texture format"))?;
        if desc.sample_count != 1 {
            return Err(Error::Unsupported("WebGL2 multisample texture storage"));
        }
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        let limits = state.context.discovery().limits();
        if desc.extent.width > limits.max_texture_size
            || desc.extent.height > limits.max_texture_size
            || (matches!(desc.dimension, GlTextureDimension::D2Array)
                && desc.extent.depth_or_layers > limits.max_array_texture_layers)
            || desc.extent.width > i32::MAX as u32
            || desc.extent.height > i32::MAX as u32
            || desc.extent.depth_or_layers > i32::MAX as u32
            || desc.mip_level_count > i32::MAX as u32
        {
            return Err(Error::InvalidRange);
        }
        let raw = state
            .context
            .raw
            .create_texture()
            .ok_or(Error::OutOfMemory)?;
        state.context.raw.bind_texture(target, Some(&raw));
        match desc.dimension {
            GlTextureDimension::D2 => state.context.raw.tex_storage_2d(
                target,
                desc.mip_level_count as i32,
                internal,
                desc.extent.width as i32,
                desc.extent.height as i32,
            ),
            GlTextureDimension::D2Array => state.context.raw.tex_storage_3d(
                target,
                desc.mip_level_count as i32,
                internal,
                desc.extent.width as i32,
                desc.extent.height as i32,
                desc.extent.depth_or_layers as i32,
            ),
            _ => unreachable!(),
        }
        if state.context.raw.get_error() != Gl::NO_ERROR {
            state.context.raw.delete_texture(Some(&raw));
            return Err(Error::Backend(
                "WebGL2 RGBA8 texture allocation failed".into(),
            ));
        }
        Ok(BrowserTexture {
            owner: self.clone(),
            raw,
            desc,
        })
    }

    /// Compiles, links, and reflects an already lowered ESSL raster program.
    pub(crate) fn create_raster_program(
        &self,
        descriptor: &super::super::api::GlProgramDescriptor,
    ) -> Result<(BrowserRasterProgram, super::super::api::GlProgramReflection)> {
        use super::super::api::GlShaderApi as _;
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        let (id, reflection) = state
            .context
            .create_program(descriptor)
            .map_err(|error| Error::Backend(format!("{error:?}")))?;
        let raw = state
            .context
            .program("create raster program", id)
            .map_err(|error| Error::Backend(format!("{error:?}")))?
            .raw
            .clone();
        Ok((
            BrowserRasterProgram {
                owner: self.clone(),
                raw,
            },
            reflection,
        ))
    }

    /// Creates a vertex-array container. Attribute and index bindings are set
    /// through the immediate encoder so their GL calls remain owner-local.
    pub(crate) fn create_vertex_array(&self) -> Result<BrowserVertexArray> {
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        let raw = state
            .context
            .raw
            .create_vertex_array()
            .ok_or(Error::OutOfMemory)?;
        Ok(BrowserVertexArray {
            owner: self.clone(),
            raw,
        })
    }

    /// Creates a WebGL2 sampler from the already admitted Core descriptor.
    pub(crate) fn create_sampler(
        &self,
        desc: super::super::api::GlSamplerDesc,
    ) -> Result<BrowserSampler> {
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        let raw = state
            .context
            .raw
            .create_sampler()
            .ok_or(Error::OutOfMemory)?;
        super::format_map::configure_sampler(&state.context.raw, &raw, desc);
        if state.context.raw.get_error() != Gl::NO_ERROR {
            state.context.raw.delete_sampler(Some(&raw));
            return Err(Error::Backend("WebGL2 sampler allocation failed".into()));
        }
        Ok(BrowserSampler {
            owner: self.clone(),
            raw,
        })
    }
}
