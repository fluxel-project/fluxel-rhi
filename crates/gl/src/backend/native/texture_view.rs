//! Native texture views own independent names sharing immutable storage.
use super::provider::NativeGlProvider;
use crate::backend::api::{
    GlDiscoverySnapshot, GlError, GlFamilyApi as _, GlFamilyProfile, GlTextureDesc,
    GlTextureDimension, GlVersion, TextureId,
};
use core::ffi::c_void;
use glow::HasContext as _;

pub(super) type TextureViewFn = unsafe extern "system" fn(u32, u32, u32, u32, u32, u32, u32, u32);

pub(super) unsafe fn load(
    discovery: &GlDiscoverySnapshot,
    loader: impl Fn(&str) -> *const c_void,
) -> Option<TextureViewFn> {
    let profile = discovery.context().profile();
    let extensions = discovery.extensions();
    let has = |name| extensions.raw_reported_names().any(|n| n == name);
    let names: &[&str] = match profile {
        GlFamilyProfile::Desktop { major, minor }
            if GlVersion::new(4, 3).is_met_by(GlVersion::new(major, minor))
                || has("GL_ARB_texture_view") =>
        {
            &["glTextureView"]
        }
        GlFamilyProfile::Embedded { .. }
            if has("GL_OES_texture_view") && has("GL_EXT_texture_view") =>
        {
            &["glTextureViewOES", "glTextureViewEXT"]
        }
        GlFamilyProfile::Embedded { .. } if has("GL_OES_texture_view") => &["glTextureViewOES"],
        GlFamilyProfile::Embedded { .. } if has("GL_EXT_texture_view") => &["glTextureViewEXT"],
        _ => return None,
    };
    let address = names
        .iter()
        .map(|name| loader(name))
        .find(|address| !address.is_null())?;
    // SAFETY: this name and signature belong to the discovered current context.
    Some(unsafe { core::mem::transmute(address) })
}

impl NativeGlProvider {
    pub(crate) fn stencil_texturing_supported(&self) -> bool {
        match self.discovery.context().profile() {
            GlFamilyProfile::Desktop { major, minor } => {
                major > 4
                    || (major == 4 && minor >= 3)
                    || self
                        .discovery
                        .extensions()
                        .raw_reported_names()
                        .any(|n| n == "GL_ARB_stencil_texturing")
            }
            GlFamilyProfile::Embedded { major, minor } => {
                major > 3
                    || (major == 3 && minor >= 1)
                    || self
                        .discovery
                        .extensions()
                        .raw_reported_names()
                        .any(|n| n == "GL_ANGLE_stencil_texturing")
            }
            _ => false,
        }
    }
    /// Select the sampled plane of a depth/stencil texture object.
    pub(crate) fn set_depth_stencil_texture_mode(
        &mut self,
        id: TextureId,
        stencil: bool,
    ) -> Result<(), GlError> {
        const OP: &str = "stencil-texture-view";
        if !self.stencil_texturing_supported() {
            return if stencil {
                Err(GlError::Unsupported {
                    operation: OP,
                    reason: "stencil texturing unavailable",
                })
            } else {
                Ok(())
            };
        }
        let (raw, desc) = self.texture(OP, id)?;
        let (target, binding) = match (desc.dimension, desc.sample_count > 1) {
            (GlTextureDimension::D2, false) => (glow::TEXTURE_2D, glow::TEXTURE_BINDING_2D),
            (GlTextureDimension::D2Array, false) => {
                (glow::TEXTURE_2D_ARRAY, glow::TEXTURE_BINDING_2D_ARRAY)
            }
            (GlTextureDimension::D2, true) => (
                glow::TEXTURE_2D_MULTISAMPLE,
                glow::TEXTURE_BINDING_2D_MULTISAMPLE,
            ),
            (GlTextureDimension::D2Array, true) => (
                glow::TEXTURE_2D_MULTISAMPLE_ARRAY,
                glow::TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY,
            ),
            _ => {
                return Err(GlError::Unsupported {
                    operation: OP,
                    reason: "stencil view target",
                });
            }
        };
        // SAFETY: the view belongs to this current context; its prior binding is restored.
        unsafe {
            let previous = self.gl.get_parameter_i32(binding) as u32;
            self.gl.bind_texture(target, Some(raw));
            self.gl.tex_parameter_i32(
                target,
                0x90EA,
                if stencil {
                    glow::STENCIL_INDEX as i32
                } else {
                    glow::DEPTH_COMPONENT as i32
                },
            );
            self.gl.bind_texture(
                target,
                std::num::NonZeroU32::new(previous).map(glow::NativeTexture),
            );
        }
        self.driver_error(OP)
    }
    pub(crate) fn supports_texture_view_format(
        &self,
        source: crate::backend::api::GlFormat,
        target: crate::backend::api::GlFormat,
    ) -> bool {
        if source == target {
            return true;
        }
        if self.texture_view_entry.is_none() || !compatible_formats(source, target) {
            return false;
        }
        // OES_texture_view does not include EXT_texture_norm16 formats in its
        // compatibility table, unlike the desktop GL table.
        if matches!(
            self.discovery.context().profile(),
            GlFamilyProfile::Embedded { .. }
        ) && [source, target].into_iter().any(|format| {
            matches!(
                format,
                crate::backend::api::GlFormat::R16Unorm
                    | crate::backend::api::GlFormat::R16Snorm
                    | crate::backend::api::GlFormat::Rg16Unorm
                    | crate::backend::api::GlFormat::Rg16Snorm
                    | crate::backend::api::GlFormat::Rgba16Unorm
                    | crate::backend::api::GlFormat::Rgba16Snorm
            )
        }) {
            return false;
        }
        !(matches!(
            self.discovery.context().profile(),
            GlFamilyProfile::Desktop { .. }
        ) && source.compressed_info().is_some_and(|info| {
            matches!(
                info.family,
                crate::backend::api::GlCompressedTextureFamily::Etc2
                    | crate::backend::api::GlCompressedTextureFamily::Eac
            )
        }))
    }
    pub(crate) fn create_texture_view_resource(
        &mut self,
        parent: TextureId,
        desc: GlTextureDesc,
        base_mip: u32,
        base_layer: u32,
    ) -> Result<TextureId, GlError> {
        const OP: &str = "create-texture-view";
        self.assert_ready(OP)?;
        let (original, source) = self.texture(OP, parent)?;
        if desc.dimension == GlTextureDimension::CubeArray {
            let supported = match self.discovery.context().profile() {
                GlFamilyProfile::Desktop { .. } => true,
                GlFamilyProfile::Embedded { major, minor } => {
                    major > 3
                        || (major == 3 && minor >= 2)
                        || self
                            .discovery
                            .extensions()
                            .raw_reported_names()
                            .any(|name| {
                                name == "GL_EXT_texture_cube_map_array"
                                    || name == "GL_OES_texture_cube_map_array"
                            })
                }
                _ => false,
            };
            if !supported {
                return Err(GlError::Unsupported {
                    operation: OP,
                    reason: "cube array texture unavailable",
                });
            }
        }
        let entry = self.texture_view_entry.ok_or(GlError::Unsupported {
            operation: OP,
            reason: "texture view entry point unavailable",
        })?;
        desc.validate()
            .map_err(|_| Self::validation(OP, "invalid view description"))?;
        if !self.supports_texture_view_format(source.format, desc.format)
            || source.sample_count != desc.sample_count
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "texture view format reinterpretation",
            });
        }
        let mip_end = base_mip
            .checked_add(desc.mip_level_count)
            .ok_or_else(|| Self::validation(OP, "view mip overflow"))?;
        let mip = source
            .mip_extent(base_mip)
            .ok_or_else(|| Self::validation(OP, "view mip out of bounds"))?;
        let layer_count = match desc.dimension {
            GlTextureDimension::D1 => 1,
            GlTextureDimension::D2 => 1,
            GlTextureDimension::Cube => 6,
            GlTextureDimension::CubeArray => desc.extent.depth_or_layers,
            GlTextureDimension::D2Array => desc.extent.depth_or_layers,
            GlTextureDimension::D3 => 1,
            _ => {
                return Err(GlError::Unsupported {
                    operation: OP,
                    reason: "texture view dimension",
                });
            }
        };
        if mip_end > source.mip_level_count
            || base_layer.checked_add(layer_count).is_none_or(|end| {
                end > match source.dimension {
                    GlTextureDimension::D1 | GlTextureDimension::D2 | GlTextureDimension::D3 => 1,
                    _ => source.extent.depth_or_layers,
                }
            })
            || desc.extent.width != mip.width
            || desc.extent.height != mip.height
        {
            return Err(Self::validation(OP, "view range out of bounds"));
        }
        let legal = matches!(
            (source.dimension, desc.dimension),
            (GlTextureDimension::D1, GlTextureDimension::D1)
                | (GlTextureDimension::D2, GlTextureDimension::D2)
                | (GlTextureDimension::D2, GlTextureDimension::D2Array)
                | (GlTextureDimension::D2Array, GlTextureDimension::D2)
                | (GlTextureDimension::D2Array, GlTextureDimension::D2Array)
                | (GlTextureDimension::D2Array, GlTextureDimension::Cube)
                | (GlTextureDimension::D2Array, GlTextureDimension::CubeArray)
                | (GlTextureDimension::D3, GlTextureDimension::D3)
        );
        if !legal {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "texture view target combination",
            });
        }
        let target = match desc.dimension {
            GlTextureDimension::D1 => glow::TEXTURE_1D,
            GlTextureDimension::D2 if desc.sample_count > 1 => glow::TEXTURE_2D_MULTISAMPLE,
            GlTextureDimension::D2Array if desc.sample_count > 1 => {
                glow::TEXTURE_2D_MULTISAMPLE_ARRAY
            }
            GlTextureDimension::D2 => glow::TEXTURE_2D,
            GlTextureDimension::D2Array => glow::TEXTURE_2D_ARRAY,
            GlTextureDimension::Cube => glow::TEXTURE_CUBE_MAP,
            GlTextureDimension::CubeArray => glow::TEXTURE_CUBE_MAP_ARRAY,
            GlTextureDimension::D3 => glow::TEXTURE_3D,
            _ => unreachable!(),
        };
        let internal =
            super::resource::native_texture_format(desc.format).ok_or(GlError::Unsupported {
                operation: OP,
                reason: "texture view format",
            })?;
        let id = TextureId::new(self.context_stamp(), self.slot(OP)?, 0);
        // SAFETY: current context; the fresh name has never been bound, and the
        // parent remains live for this call. GL validates storage compatibility.
        let raw = unsafe { self.gl.create_texture() }.map_err(|message| GlError::Driver {
            operation: OP,
            message,
        })?;
        unsafe {
            entry(
                raw.0.get(),
                target,
                original.0.get(),
                internal,
                base_mip,
                desc.mip_level_count,
                base_layer,
                layer_count,
            );
        }
        if let Err(error) = self.driver_error(OP) {
            unsafe { self.gl.delete_texture(raw) };
            return Err(error);
        }
        if desc.sample_count == 1 {
            unsafe {
                let binding = match target {
                    glow::TEXTURE_1D => glow::TEXTURE_BINDING_1D,
                    glow::TEXTURE_2D => glow::TEXTURE_BINDING_2D,
                    glow::TEXTURE_2D_ARRAY => glow::TEXTURE_BINDING_2D_ARRAY,
                    glow::TEXTURE_CUBE_MAP => glow::TEXTURE_BINDING_CUBE_MAP,
                    glow::TEXTURE_CUBE_MAP_ARRAY => glow::TEXTURE_BINDING_CUBE_MAP_ARRAY,
                    glow::TEXTURE_3D => glow::TEXTURE_BINDING_3D,
                    _ => unreachable!(),
                };
                let previous = self.gl.get_parameter_i32(binding) as u32;
                self.gl.bind_texture(target, Some(raw));
                self.gl.tex_parameter_i32(
                    target,
                    glow::TEXTURE_MAX_LEVEL,
                    desc.mip_level_count as i32 - 1,
                );
                self.gl.tex_parameter_i32(
                    target,
                    glow::TEXTURE_MIN_FILTER,
                    glow::NEAREST_MIPMAP_NEAREST as i32,
                );
                self.gl
                    .tex_parameter_i32(target, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
                self.gl.bind_texture(
                    target,
                    std::num::NonZeroU32::new(previous).map(glow::NativeTexture),
                );
            }
            if let Err(error) = self.driver_error(OP) {
                unsafe { self.gl.delete_texture(raw) };
                return Err(error);
            }
        }
        self.textures.insert(id, (raw, desc));
        Ok(id)
    }
}

/// Classes from GL 4.6 table 8.22 and OES_texture_view table 8.X.2.
pub(crate) fn compatible_formats(
    a: crate::backend::api::GlFormat,
    b: crate::backend::api::GlFormat,
) -> bool {
    use crate::backend::api::GlFormat as F;
    if a == b {
        return true;
    }
    if let (F::Astc { block: x, .. }, F::Astc { block: y, .. }) = (a, b) {
        return x == y;
    }
    fn class(f: F) -> Option<u8> {
        Some(match f {
            F::R8Unorm | F::R8Snorm | F::R8Uint | F::R8Sint => 15,
            F::Rg8Unorm
            | F::Rg8Snorm
            | F::Rg8Uint
            | F::Rg8Sint
            | F::R16Uint
            | F::R16Sint
            | F::R16Float
            | F::R16Unorm
            | F::R16Snorm => 16,
            F::Rgba8Unorm
            | F::Rgba8Srgb
            | F::Rgba8Snorm
            | F::Rgba8Uint
            | F::Rgba8Sint
            | F::Rg16Uint
            | F::Rg16Sint
            | F::Rg16Float
            | F::Rg16Unorm
            | F::Rg16Snorm
            | F::R32Uint
            | F::R32Sint
            | F::R32Float
            | F::Rgb9e5Ufloat
            | F::Rgb10a2Uint
            | F::Rgb10a2Unorm
            | F::Rg11b10Ufloat => 1,
            F::Rgba16Uint
            | F::Rgba16Sint
            | F::Rgba16Float
            | F::Rgba16Unorm
            | F::Rgba16Snorm
            | F::Rg32Uint
            | F::Rg32Sint
            | F::Rg32Float => 17,
            F::Rgba32Uint | F::Rgba32Sint | F::Rgba32Float => 18,
            F::Bc1RgbUnorm | F::Bc1RgbSrgb => 2,
            F::Bc1RgbaUnorm | F::Bc1RgbaSrgb => 3,
            F::Bc2RgbaUnorm | F::Bc2RgbaSrgb => 4,
            F::Bc3RgbaUnorm | F::Bc3RgbaSrgb => 5,
            F::Bc4RUnorm | F::Bc4RSnorm => 6,
            F::Bc5RgUnorm | F::Bc5RgSnorm => 7,
            F::Bc6hRgbUfloat | F::Bc6hRgbSfloat => 8,
            F::Bc7RgbaUnorm | F::Bc7RgbaSrgb => 9,
            F::Etc2Rgb8Unorm | F::Etc2Rgb8Srgb => 10,
            F::Etc2Rgb8A1Unorm | F::Etc2Rgb8A1Srgb => 11,
            F::Etc2Rgba8Unorm | F::Etc2Rgba8Srgb => 12,
            F::EacR11Unorm | F::EacR11Snorm => 13,
            F::EacRg11Unorm | F::EacRg11Snorm => 14,
            _ => return None,
        })
    }
    class(a).zip(class(b)).is_some_and(|(a, b)| a == b)
}
