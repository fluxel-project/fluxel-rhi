//! Browser-local resource and pipeline allocation.
//!
//! These constructors create WebGL objects immediately for one `BrowserOwner`.
//! They never cache commands or transfer JavaScript handles into Core.

use crate::backend::api::{GlBufferDesc, GlBufferUsage, GlFamilyApi as _, GlFormat};
use fluxel_rhi_core::api::{Error, Result};
use wasm_bindgen::JsCast;
use web_sys::{WebGl2RenderingContext as Gl, WebGlBuffer};

use super::owner::{
    BrowserBuffer, BrowserOwner, BrowserRasterProgram, BrowserSampler, BrowserTexture,
    BrowserVertexArray,
};

/// Saves WebGL's actual client-unpack state around the array/D3 initialization
/// workaround. Browser encoders can legitimately leave a scoped non-default
/// unpack state while resource allocation is re-entered, so this must not
/// assume a global canonical state.
struct ScopedUnpackState<'a> {
    gl: &'a Gl,
    buffer: Option<WebGlBuffer>,
    values: [i32; 6],
}

impl<'a> ScopedUnpackState<'a> {
    fn enter(gl: &'a Gl) -> Self {
        let parameters = [
            Gl::UNPACK_ALIGNMENT,
            Gl::UNPACK_ROW_LENGTH,
            Gl::UNPACK_IMAGE_HEIGHT,
            Gl::UNPACK_SKIP_PIXELS,
            Gl::UNPACK_SKIP_ROWS,
            Gl::UNPACK_SKIP_IMAGES,
        ];
        let values = parameters.map(|parameter| {
            gl.get_parameter(parameter)
                .ok()
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0) as i32
        });
        let buffer = gl
            .get_parameter(Gl::PIXEL_UNPACK_BUFFER_BINDING)
            .ok()
            .and_then(|value| value.dyn_into::<WebGlBuffer>().ok());
        gl.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, None);
        for (parameter, value) in parameters.into_iter().zip([1, 0, 0, 0, 0, 0]) {
            gl.pixel_storei(parameter, value);
        }
        Self { gl, buffer, values }
    }
}

impl Drop for ScopedUnpackState<'_> {
    fn drop(&mut self) {
        for (parameter, value) in [
            Gl::UNPACK_ALIGNMENT,
            Gl::UNPACK_ROW_LENGTH,
            Gl::UNPACK_IMAGE_HEIGHT,
            Gl::UNPACK_SKIP_PIXELS,
            Gl::UNPACK_SKIP_ROWS,
            Gl::UNPACK_SKIP_IMAGES,
        ]
        .into_iter()
        .zip(self.values)
        {
            self.gl.pixel_storei(parameter, value);
        }
        self.gl
            .bind_buffer(Gl::PIXEL_UNPACK_BUFFER, self.buffer.as_ref());
    }
}

impl BrowserOwner {
    /// Allocates a dynamic byte-addressable buffer for the static backend.
    pub(crate) fn create_buffer(&self, desc: GlBufferDesc) -> Result<BrowserBuffer> {
        desc.validate().map_err(|_| Error::InvalidRange)?;
        let size = i32::try_from(desc.size).map_err(|_| Error::InvalidRange)?;
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
            .buffer_data_with_i32(target, size, Gl::DYNAMIC_DRAW);
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
            GlTextureDimension::D3 => Gl::TEXTURE_3D,
            _ => return Err(Error::Unsupported("WebGL2 texture dimension")),
        };
        let internal = super::format_map::internal_format(desc.format)
            .ok_or(Error::Unsupported("WebGL2 texture format"))?;
        let mut state = self.state.borrow_mut();
        state.ensure_active()?;
        let limits = state.context.discovery().limits();
        let xy_limit = if desc.dimension == GlTextureDimension::D3 {
            limits.max_3d_texture_size
        } else {
            limits.max_texture_size
        };
        if desc.extent.width > xy_limit
            || desc.extent.height > xy_limit
            || (desc.dimension == GlTextureDimension::D3
                && desc.extent.depth_or_layers > limits.max_3d_texture_size)
            || (matches!(desc.dimension, GlTextureDimension::D2Array)
                && desc.extent.depth_or_layers > limits.max_array_texture_layers)
            || desc.extent.width > i32::MAX as u32
            || desc.extent.height > i32::MAX as u32
            || desc.extent.depth_or_layers > i32::MAX as u32
            || desc.mip_level_count > i32::MAX as u32
        {
            return Err(Error::InvalidRange);
        }
        let renderbuffer_only =
            desc.sample_count > 1 || desc.format == super::super::api::GlFormat::Stencil8;
        if renderbuffer_only {
            use super::super::api::{GlFormatResourceKind, GlTextureUsage};
            let standalone_stencil_copy =
                desc.format == GlFormat::Stencil8 && desc.sample_count == 1;
            let admitted_usage = GlTextureUsage::RENDER_ATTACHMENT
                | GlTextureUsage::COPY_SOURCE
                | GlTextureUsage::COPY_DESTINATION;
            let resolve_usage = GlTextureUsage::RENDER_ATTACHMENT | GlTextureUsage::COPY_SOURCE;
            if desc.dimension != GlTextureDimension::D2
                || desc.extent.depth_or_layers != 1
                || desc.mip_level_count != 1
                || if standalone_stencil_copy {
                    desc.usage.is_empty() || !admitted_usage.contains(desc.usage)
                } else {
                    desc.usage.is_empty() || !resolve_usage.contains(desc.usage)
                }
            {
                return Err(Error::Unsupported(
                    "WebGL2 renderbuffer requires single-layer attachment storage",
                ));
            }
            let facts = state.context.discovery().formats().get_for(
                GlFormatResourceKind::Renderbuffer,
                desc.format,
                desc.sample_count,
            );
            if !facts.is_some_and(|f| f.renderable) {
                return Err(Error::Unsupported(
                    "WebGL2 renderbuffer format/sample count",
                ));
            }
            let raw = state
                .context
                .raw
                .create_renderbuffer()
                .ok_or(Error::OutOfMemory)?;
            state
                .context
                .raw
                .bind_renderbuffer(Gl::RENDERBUFFER, Some(&raw));
            if desc.sample_count > 1 {
                state.context.raw.renderbuffer_storage_multisample(
                    Gl::RENDERBUFFER,
                    desc.sample_count as i32,
                    internal,
                    desc.extent.width as i32,
                    desc.extent.height as i32,
                );
            } else {
                state.context.raw.renderbuffer_storage(
                    Gl::RENDERBUFFER,
                    internal,
                    desc.extent.width as i32,
                    desc.extent.height as i32,
                );
            }
            state.context.raw.bind_renderbuffer(Gl::RENDERBUFFER, None);
            if state.context.raw.is_context_lost() {
                state.mark_lost();
                return Err(Error::DeviceLost);
            }
            if state.context.raw.get_error() != Gl::NO_ERROR {
                state.context.raw.delete_renderbuffer(Some(&raw));
                return Err(Error::Backend(
                    "WebGL2 renderbuffer allocation failed".into(),
                ));
            }
            return Ok(BrowserTexture {
                owner: self.clone(),
                raw: None,
                renderbuffer: Some(raw),
                desc,
                lifetime: std::rc::Rc::new(()),
            });
        }
        let raw = state
            .context
            .raw
            .create_texture()
            .ok_or(Error::OutOfMemory)?;
        state.context.raw.bind_texture(target, Some(&raw));
        // texelFetch can access integer textures without a sampler object;
        // WebGL's default linear filter would make those textures incomplete.
        if super::ordinary_format::encoding(desc.format).is_some_and(|v| v.integer || v.float32)
            || matches!(
                desc.format,
                GlFormat::Depth16Unorm
                    | GlFormat::Depth24Unorm
                    | GlFormat::Depth32Float
                    | GlFormat::Depth24PlusStencil8
                    | GlFormat::Depth32FloatStencil8
            )
        {
            state.context.raw.tex_parameteri(
                target,
                Gl::TEXTURE_MIN_FILTER,
                Gl::NEAREST_MIPMAP_NEAREST as i32,
            );
            state
                .context
                .raw
                .tex_parameteri(target, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
        }
        match desc.dimension {
            GlTextureDimension::D2 => state.context.raw.tex_storage_2d(
                target,
                desc.mip_level_count as i32,
                internal,
                desc.extent.width as i32,
                desc.extent.height as i32,
            ),
            GlTextureDimension::D2Array | GlTextureDimension::D3 => {
                state.context.raw.tex_storage_3d(
                    target,
                    desc.mip_level_count as i32,
                    internal,
                    desc.extent.width as i32,
                    desc.extent.height as i32,
                    desc.extent.depth_or_layers as i32,
                )
            }
            _ => unreachable!(),
        }
        if state.context.raw.is_context_lost() {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        if state.context.raw.get_error() != Gl::NO_ERROR {
            state.context.raw.delete_texture(Some(&raw));
            return Err(Error::Backend(
                "WebGL2 texture storage allocation failed".into(),
            ));
        }
        // Some browser implementations retain lazy per-slice initialization
        // after immutable allocation. An untouched slice can then lose its
        // first framebuffer clear, so explicitly touch every subresource
        // before exposing the allocation. One 1x1 zero column covers every
        // slice in a mip with one native call; this avoids the former
        // per-slice call loop while preserving the driver workaround. Reuse
        // the largest column source for every mip: WebGL consumes only the
        // requested prefix, so it avoids one CPU allocation and JS typed-array
        // conversion per level.
        if matches!(
            desc.dimension,
            GlTextureDimension::D2Array | GlTextureDimension::D3
        ) {
            if let Some(encoding) = super::ordinary_format::encoding(desc.format) {
                let (format, ty) = (encoding.pixel, encoding.ty);
                let gl = &state.context.raw;
                let max_slices = desc.extent.depth_or_layers;
                let byte_count = usize::try_from(max_slices)
                    .ok()
                    .and_then(|slices| slices.checked_mul(encoding.bytes as usize))
                    .ok_or(Error::InvalidRange)?;
                // Build the reusable client source before changing GL state,
                // so an allocation/conversion failure cannot leave a partial
                // unpack scope behind.
                let zeros = super::ordinary_format::typed_pixels(&vec![0; byte_count], ty)?;
                let unpack = ScopedUnpackState::enter(gl);
                let mut failed = false;
                for mip in 0..desc.mip_level_count {
                    let slices = if desc.dimension == GlTextureDimension::D3 {
                        desc.extent
                            .depth_or_layers
                            .checked_shr(mip)
                            .unwrap_or(0)
                            .max(1)
                    } else {
                        desc.extent.depth_or_layers
                    };
                    failed |= gl
                        .tex_sub_image_3d_with_opt_array_buffer_view(
                            target,
                            mip as i32,
                            0,
                            0,
                            0,
                            1,
                            1,
                            slices as i32,
                            format,
                            ty,
                            Some(&zeros),
                        )
                        .is_err();
                }
                // Restore before deleting the failed allocation, preserving
                // the caller's actual PBO and unpack values on every exit.
                drop(unpack);
                if failed {
                    gl.delete_texture(Some(&raw));
                    if gl.is_context_lost() {
                        state.mark_lost();
                        return Err(Error::DeviceLost);
                    }
                    return Err(Error::Backend(
                        "WebGL2 texture subresource initialization failed".into(),
                    ));
                }
            }
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
        if state.context.raw.is_context_lost() {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        if state.context.raw.get_error() != Gl::NO_ERROR {
            state.context.raw.delete_texture(Some(&raw));
            return Err(Error::Backend(
                "WebGL2 RGBA8 texture allocation failed".into(),
            ));
        }
        Ok(BrowserTexture {
            owner: self.clone(),
            raw: Some(raw),
            renderbuffer: None,
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
                id,
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
