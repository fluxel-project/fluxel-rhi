//! Immediate WebGL2 texture upload lowering.
//!
//! Uploads cross the browser boundary synchronously and restore their pixel
//! store state before returning; this module never retains source commands.

use fluxel_rhi_core::api::{Error, Result};
use web_sys::WebGl2RenderingContext as Gl;

use super::super::owner::{BrowserEncoder, BrowserTexture};

impl BrowserEncoder {
    /// Uploads one color region into a 2D or 2D-array immutable texture.
    ///
    /// Core coordinates are upper-left.  The Y inversion lives here at the
    /// WebGL boundary, once, and a 2D-array layer remains a Z slice rather
    /// than being folded into that orientation conversion.
    pub(crate) fn write_texture_region(
        &mut self,
        destination: &BrowserTexture,
        mip: u32,
        origin: [u32; 3],
        extent: [u32; 3],
        bytes_per_row: u32,
        rows_per_image: u32,
        bytes: &[u8],
    ) -> Result<()> {
        use crate::backend::gl::api::{GlFormat, GlTextureDimension};

        self.same_owner(&destination.owner)?;
        if !matches!(
            destination.desc.format,
            GlFormat::Rgba8Unorm | GlFormat::Rgba8Srgb
        ) || mip >= destination.desc.mip_level_count
            || extent.contains(&0)
            || bytes_per_row != extent[0].checked_mul(4).ok_or(Error::InvalidRange)?
            || rows_per_image != extent[1]
        {
            return Err(Error::Unsupported("WebGL2 texture upload layout"));
        }
        let width = destination
            .desc
            .extent
            .width
            .checked_shr(mip)
            .unwrap_or(0)
            .max(1);
        let height = destination
            .desc
            .extent
            .height
            .checked_shr(mip)
            .unwrap_or(0)
            .max(1);
        let layers = destination.desc.extent.depth_or_layers;
        if origin[0].checked_add(extent[0]).is_none_or(|x| x > width)
            || origin[1].checked_add(extent[1]).is_none_or(|y| y > height)
            || origin[2].checked_add(extent[2]).is_none_or(|z| z > layers)
            || bytes.len()
                != (extent[0] as usize)
                    .saturating_mul(extent[1] as usize)
                    .saturating_mul(extent[2] as usize)
                    .saturating_mul(4)
        {
            return Err(Error::InvalidRange);
        }
        let native_y = height
            .checked_sub(
                origin[1]
                    .checked_add(extent[1])
                    .ok_or(Error::InvalidRange)?,
            )
            .ok_or(Error::InvalidRange)?;
        let target = match destination.desc.dimension {
            GlTextureDimension::D2 => {
                if origin[2] != 0 || extent[2] != 1 {
                    return Err(Error::InvalidRange);
                }
                Gl::TEXTURE_2D
            }
            GlTextureDimension::D2Array => Gl::TEXTURE_2D_ARRAY,
            _ => return Err(Error::Unsupported("WebGL2 texture dimension")),
        };
        self.with_context("write_texture_region", |raw| {
            raw.bind_texture(target, Some(&destination.raw));
            raw.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, None);
            raw.pixel_storei(Gl::UNPACK_ALIGNMENT, 1);
            raw.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, 1);
            match destination.desc.dimension {
                GlTextureDimension::D2 => {
                    let _ = raw
                        .tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
                            target,
                            mip as i32,
                            origin[0] as i32,
                            native_y as i32,
                            extent[0] as i32,
                            extent[1] as i32,
                            Gl::RGBA,
                            Gl::UNSIGNED_BYTE,
                            Some(bytes),
                        );
                }
                GlTextureDimension::D2Array => {
                    let _ = raw.tex_sub_image_3d_with_opt_u8_array(
                        target,
                        mip as i32,
                        origin[0] as i32,
                        native_y as i32,
                        origin[2] as i32,
                        extent[0] as i32,
                        extent[1] as i32,
                        extent[2] as i32,
                        Gl::RGBA,
                        Gl::UNSIGNED_BYTE,
                        Some(bytes),
                    );
                }
                _ => unreachable!(),
            }
            raw.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, 0);
            raw.pixel_storei(Gl::UNPACK_ALIGNMENT, 4);
        })
    }
}
