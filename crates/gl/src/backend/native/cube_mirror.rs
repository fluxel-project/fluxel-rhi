//! Native Cube mirrors for sampling a D2 array when texture views are absent.
//!
//! The source allocation remains authoritative. The caller synchronizes each
//! mirror immediately before a draw that can sample it, after prior GPU writes.
//! The mirror is an ordinary provider-owned texture and is retired normally.
use super::provider::NativeGlProvider;
use crate::backend::api::{
    GlError, GlFamilyApi as _, GlFormat, GlFormatResourceKind, GlTextureDesc, GlTextureDimension,
    GlTextureUsage, TextureId,
};
use crate::backend::state::{ScopedRawAccess, StateEvent};
use glow::HasContext as _;

impl NativeGlProvider {
    #[cfg(test)]
    pub(crate) fn disable_texture_views_for_cube_mirror_test(&mut self) {
        self.texture_view_entry = None;
    }

    pub(crate) fn create_cube_mirror(
        &mut self,
        parent: TextureId,
        desc: GlTextureDesc,
    ) -> Result<TextureId, GlError> {
        const OP: &str = "create cube mirror";
        self.assert_ready(OP)?;
        let (_, source) = self.texture(OP, parent)?;
        desc.validate()
            .map_err(|_| Self::validation(OP, "invalid cube mirror descriptor"))?;
        if source.dimension != GlTextureDimension::D2Array
            || desc.dimension != GlTextureDimension::Cube
            || !cube_mirror_format(source.format)
            || desc.format != source.format
            || source.sample_count != 1
            || desc.sample_count != 1
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "cube mirror requires a supported single-sample color 2D array",
            });
        }
        if !self
            .discovery
            .formats()
            .get_for(GlFormatResourceKind::Texture, source.format, 1)
            .is_some_and(|facts| facts.renderable)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "cube mirror source format is not framebuffer renderable",
            });
        }
        if desc.extent.width > self.discovery.limits().max_cube_map_texture_size {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "cube mirror exceeds cube texture size limit",
            });
        }
        let id = TextureId::new(self.context_stamp(), self.slot(OP)?, 0);
        // SAFETY: the provider owns the current context and the fresh name.
        let name = unsafe { self.gl.create_texture() }.map_err(|message| GlError::Driver {
            operation: OP,
            message,
        })?;
        // Mutable storage covers desktop GL 4.0 and GLES 3.0 without requiring
        // immutable texture storage; no upload buffer may be interpreted as data.
        unsafe {
            self.gl.bind_texture(glow::TEXTURE_CUBE_MAP, Some(name));
            self.gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
            for mip in 0..desc.mip_level_count {
                let size = desc.mip_extent(mip).expect("validated mip");
                for face in 0..6 {
                    let (external, element) = cube_external(source.format);
                    self.gl.tex_image_2d(
                        glow::TEXTURE_CUBE_MAP_POSITIVE_X + face,
                        mip as i32,
                        cube_internal(source.format) as i32,
                        size.width as i32,
                        size.height as i32,
                        0,
                        external,
                        element,
                        glow::PixelUnpackData::Slice(None),
                    );
                }
            }
            self.gl
                .tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_BASE_LEVEL, 0);
            self.gl.tex_parameter_i32(
                glow::TEXTURE_CUBE_MAP,
                glow::TEXTURE_MAX_LEVEL,
                desc.mip_level_count as i32 - 1,
            );
            self.gl.tex_parameter_i32(
                glow::TEXTURE_CUBE_MAP,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST_MIPMAP_NEAREST as i32,
            );
            self.gl.tex_parameter_i32(
                glow::TEXTURE_CUBE_MAP,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            self.gl.bind_texture(glow::TEXTURE_CUBE_MAP, None);
        }
        self.context_state
            .event(StateEvent::ScopedRawAccess(ScopedRawAccess::all()));
        if let Err(error) = self.driver_error(OP) {
            // SAFETY: name is unpublished and belongs to this current context.
            unsafe { self.gl.delete_texture(name) };
            return Err(error);
        }
        self.textures.insert(id, (name, desc));
        Ok(id)
    }

    pub(crate) fn sync_cube_mirror(
        &mut self,
        parent: TextureId,
        mirror: TextureId,
        base_mip: u32,
        base_layer: u32,
    ) -> Result<(), GlError> {
        const OP: &str = "sync cube mirror";
        self.assert_ready(OP)?;
        let (src, source) = self.texture(OP, parent)?;
        let (dst, view) = self.texture(OP, mirror)?;
        let size = source
            .mip_extent(base_mip)
            .ok_or_else(|| Self::validation(OP, "source mip out of bounds"))?;
        if source.dimension != GlTextureDimension::D2Array
            || view.dimension != GlTextureDimension::Cube
            || !cube_mirror_format(source.format)
            || view.format != source.format
            || source.sample_count != 1
            || view.sample_count != 1
            || size.width != view.extent.width
            || size.height != view.extent.height
            || base_mip
                .checked_add(view.mip_level_count)
                .is_none_or(|end| end > source.mip_level_count)
            || base_layer
                .checked_add(6)
                .is_none_or(|end| end > source.extent.depth_or_layers)
        {
            return Err(Self::validation(
                OP,
                "cube mirror does not match source range",
            ));
        }
        let image_writes = source.usage.contains(GlTextureUsage::STORAGE_BINDING);
        if image_writes {
            let profile = self.discovery.context().profile();
            let available = match profile {
                crate::backend::api::GlFamilyProfile::Desktop { major, minor } => {
                    major > 4
                        || (major == 4 && minor >= 2)
                        || self
                            .discovery
                            .extensions()
                            .raw_reported_names()
                            .any(|n| n == "GL_ARB_shader_image_load_store")
                }
                crate::backend::api::GlFamilyProfile::Embedded { major, minor } => {
                    major > 3 || (major == 3 && minor >= 1)
                }
                _ => false,
            };
            if !available {
                return Err(GlError::Unsupported {
                    operation: OP,
                    reason: "framebuffer barrier unavailable for storage image source",
                });
            }
        }
        // SAFETY: both textures are live in this context and each mip/layer was
        // range-checked. The scratch bindings are returned to the provider's
        // canonical empty baseline; the state mirror is invalidated below.
        unsafe {
            // Shader image writes require framebuffer visibility before the
            // source is read through the temporary framebuffer.
            if image_writes {
                self.gl.memory_barrier(glow::FRAMEBUFFER_BARRIER_BIT);
            }
            let scratch = self
                .gl
                .create_framebuffer()
                .map_err(|message| GlError::Driver {
                    operation: OP,
                    message,
                })?;
            self.gl
                .bind_framebuffer(glow::READ_FRAMEBUFFER, Some(scratch));
            self.gl.read_buffer(glow::COLOR_ATTACHMENT0);
            self.gl.bind_texture(glow::TEXTURE_CUBE_MAP, Some(dst));
            for mip in 0..view.mip_level_count {
                let size = view.mip_extent(mip).expect("validated mip");
                for face in 0..6 {
                    self.gl.framebuffer_texture_layer(
                        glow::READ_FRAMEBUFFER,
                        glow::COLOR_ATTACHMENT0,
                        Some(src),
                        (base_mip + mip) as i32,
                        (base_layer + face) as i32,
                    );
                    self.gl.copy_tex_sub_image_2d(
                        glow::TEXTURE_CUBE_MAP_POSITIVE_X + face,
                        mip as i32,
                        0,
                        0,
                        0,
                        0,
                        size.width as i32,
                        size.height as i32,
                    );
                }
            }
            self.gl.bind_texture(glow::TEXTURE_CUBE_MAP, None);
            self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            self.gl.delete_framebuffer(scratch);
            self.context_state
                .event(StateEvent::ScopedRawAccess(ScopedRawAccess::all()));
            // Source/view shape and attachment compatibility were validated
            // before entering this recurring mirror path. Do not synchronously
            // query FBO status for every mip/face; any driver rejection joins
            // the command stream's publication-boundary diagnostic.
            self.defer_driver_error(OP)
        }
    }
}

fn cube_mirror_format(format: GlFormat) -> bool {
    matches!(
        format,
        GlFormat::Rgba8Unorm | GlFormat::Rgba8Uint | GlFormat::Rgba8Sint
    )
}

fn cube_internal(format: GlFormat) -> u32 {
    match format {
        GlFormat::Rgba8Unorm => glow::RGBA8,
        GlFormat::Rgba8Uint => glow::RGBA8UI,
        GlFormat::Rgba8Sint => glow::RGBA8I,
        _ => unreachable!("validated cube mirror format"),
    }
}

fn cube_external(format: GlFormat) -> (u32, u32) {
    match format {
        GlFormat::Rgba8Unorm => (glow::RGBA, glow::UNSIGNED_BYTE),
        GlFormat::Rgba8Uint => (glow::RGBA_INTEGER, glow::UNSIGNED_BYTE),
        GlFormat::Rgba8Sint => (glow::RGBA_INTEGER, glow::BYTE),
        _ => unreachable!("validated cube mirror format"),
    }
}
