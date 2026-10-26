//! Direct constant fills of uncompressed, non-renderable color textures.
//!
//! sRGB and depth/stencil retain the framebuffer path: pixel-transfer clears
//! do not provide framebuffer sRGB conversion or depth-only preservation.
use super::provider::NativeGlProvider;
use crate::backend::api::{
    GlDiscoverySnapshot, GlError, GlFamilyProfile, GlTextureDesc, GlTextureDimension,
};
use fluxel_rhi_core::api::ClearColor;

pub(super) type ClearTextureFn = unsafe extern "system" fn(
    u32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    u32,
    u32,
    *const core::ffi::c_void,
);

/// # Safety
/// The loader belongs to the current context described by `discovery`.
pub(super) unsafe fn load(
    discovery: &GlDiscoverySnapshot,
    loader: &impl Fn(&str) -> *const core::ffi::c_void,
) -> Option<ClearTextureFn> {
    let has = |name| {
        discovery
            .extensions()
            .raw_reported_names()
            .any(|value| value == name)
    };
    let core = matches!(discovery.context().profile(), GlFamilyProfile::Desktop { major, minor } if (major, minor) >= (4,4));
    let name = if core || has("GL_ARB_clear_texture") {
        "glClearTexSubImage"
    } else if has("GL_EXT_clear_texture") {
        "glClearTexSubImageEXT"
    } else {
        return None;
    };
    let ptr = loader(name);
    if ptr.is_null() {
        return None;
    }
    // SAFETY: supported core/EXT variants share this exact GL signature.
    Some(unsafe { core::mem::transmute(ptr) })
}

impl NativeGlProvider {
    pub(super) fn clear_nonrenderable_color(
        &self,
        texture: glow::NativeTexture,
        desc: GlTextureDesc,
        mips: std::ops::Range<u32>,
        layers: std::ops::Range<u32>,
        color: ClearColor,
    ) -> Result<(), GlError> {
        let clear = self.clear_texture_entry.ok_or(GlError::Unsupported {
            operation: "clear-texture",
            reason: "non-renderable texture clear requires native clear-texture support",
        })?;
        let (format, kind, components) = match color {
            ClearColor::Float(values) => (glow::RGBA, glow::FLOAT, values.map(f32::to_bits)),
            ClearColor::Uint(values) => (glow::RGBA_INTEGER, glow::UNSIGNED_INT, values),
            ClearColor::Sint(values) => (glow::RGBA_INTEGER, glow::INT, values.map(|v| v as u32)),
        };
        for mip in mips {
            let extent = desc
                .mip_extent(mip)
                .ok_or_else(|| Self::validation("clear-texture", "invalid mip"))?;
            let (z, depth) = if desc.dimension == GlTextureDimension::D3 {
                (0, extent.depth_or_layers)
            } else {
                (layers.start, layers.end - layers.start)
            };
            // SAFETY: the caller authenticated the texture and ranges. The
            // component array is aligned for every 32-bit GL transfer type and
            // remains live for the synchronous native call.
            unsafe {
                clear(
                    texture.0.get(),
                    mip as i32,
                    0,
                    0,
                    z as i32,
                    extent.width as i32,
                    extent.height as i32,
                    depth as i32,
                    format,
                    kind,
                    components.as_ptr().cast(),
                );
            }
            self.driver_error("clear-texture")?;
        }
        Ok(())
    }
}
