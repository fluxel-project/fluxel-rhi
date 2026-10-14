//! Browser-local resource and pipeline allocation.
//!
//! These constructors create WebGL objects immediately for one `BrowserOwner`.
//! They never cache commands or transfer JavaScript handles into Core.

use crate::backend::gl::api::{GlBufferDesc, GlBufferUsage, GlFamilyApi as _};
use fluxel_rhi_core::api::{Error, Result};
use web_sys::WebGl2RenderingContext as Gl;

use super::owner::{
    BrowserBuffer, BrowserOwner, BrowserRasterProgram, BrowserSampler, BrowserTexture,
    BrowserVertexArray,
};

impl BrowserOwner {
    /// Allocates a dynamic byte-addressable buffer for the static backend.
    pub(crate) fn create_buffer(&self, desc: GlBufferDesc) -> Result<BrowserBuffer> {
        self.drain_pending_deletes()?;
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        let is_index = desc.usage.contains(GlBufferUsage::INDEX);
        let target = if is_index {
            // ELEMENT_ARRAY_BUFFER is VAO state. Resource creation happens
            // outside a raster pass, and the default VAO prevents allocating
            // an index buffer from replacing a pipeline VAO's live binding.
            state.context.raw.bind_vertex_array(None);
            Gl::ELEMENT_ARRAY_BUFFER
        } else {
            Gl::ARRAY_BUFFER
        };
        let raw = state
            .context
            .raw
            .create_buffer()
            .ok_or(Error::OutOfMemory)?;
        state.context.raw.bind_buffer(target, Some(&raw));
        state
            .context
            .raw
            .buffer_data_with_f64(target, desc.size as f64, Gl::DYNAMIC_DRAW);
        if state.context.raw.get_error() != Gl::NO_ERROR {
            state.context.raw.delete_buffer(Some(&raw));
            return Err(Error::Backend("WebGL2 buffer allocation failed".into()));
        }
        Ok(BrowserBuffer {
            owner: self.clone(),
            raw,
            size: desc.size,
            lifetime: std::rc::Rc::new(()),
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

        self.drain_pending_deletes()?;
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
        // A sampler may request mip filtering even when this immutable image
        // intentionally has one level. Clamp the texture's mip range to the
        // storage actually allocated; otherwise WebGL treats the texture as
        // incomplete and samples black for ordinary glTF material textures.
        state
            .context
            .raw
            .tex_parameteri(target, Gl::TEXTURE_BASE_LEVEL, 0);
        state.context.raw.tex_parameteri(
            target,
            Gl::TEXTURE_MAX_LEVEL,
            desc.mip_level_count as i32 - 1,
        );
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
            lifetime: std::rc::Rc::new(()),
        })
    }

    /// Compiles, links, and reflects an already lowered ESSL raster program.
    pub(crate) fn create_raster_program(
        &self,
        descriptor: &super::super::api::GlProgramDescriptor,
    ) -> Result<(BrowserRasterProgram, super::super::api::GlProgramReflection)> {
        use super::super::api::GlShaderApi as _;
        self.drain_pending_deletes()?;
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
                lifetime: std::rc::Rc::new(()),
            },
            reflection,
        ))
    }

    /// Creates a vertex-array container. Attribute and index bindings are set
    /// through the immediate encoder so their GL calls remain owner-local.
    pub(crate) fn create_vertex_array(&self) -> Result<BrowserVertexArray> {
        self.drain_pending_deletes()?;
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
            lifetime: std::rc::Rc::new(()),
        })
    }

    /// Creates a WebGL2 sampler from the already admitted Core descriptor.
    pub(crate) fn create_sampler(
        &self,
        desc: super::super::api::GlSamplerDesc,
    ) -> Result<BrowserSampler> {
        self.drain_pending_deletes()?;
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
            lifetime: std::rc::Rc::new(()),
        })
    }
}
