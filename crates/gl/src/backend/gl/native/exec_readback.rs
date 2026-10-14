//! Native buffer, texture, and adopted-drawable readback.
//!
//! Readback is intentionally separate from copy/upload issuance: GL may block
//! while resolving a queryable texture or default framebuffer. Callers supply
//! Core upper-left regions; this module converts their native rectangle and
//! reverses GL rows exactly once before returning public row-zero bytes.

use super::super::provider::NativeGlProvider;
use super::{
    UnpackDirection, ensure_single_2d_rect, native_texture_region, place_rows,
    place_rows_for_extent, readback_encoding, rect_coords, transfer_format,
};
use crate::backend::gl::api::{
    GlBufferRange, GlError, GlExtent3d, GlFamilyApi as _, GlPixelFormat, GlPixelLayout, GlReadback,
    GlRepackPolicy, GlTextureRegion,
};

impl NativeGlProvider {
    pub(super) fn read_buffer_now(&mut self, source: GlBufferRange) -> Result<Vec<u8>, GlError> {
        use glow::HasContext as _;
        const OP: &str = "read-buffer";
        self.assert_ready(OP)?;
        let (name, desc) = self.buffer(OP, source.buffer)?;
        source
            .validate_for(desc)
            .map_err(|_| Self::validation(OP, "invalid source buffer range"))?;
        let offset = i32::try_from(source.offset)
            .map_err(|_| Self::validation(OP, "offset exceeds GLintptr"))?;
        let mut out = vec![
            0u8;
            usize::try_from(source.size)
                .map_err(|_| GlError::OutOfMemory { operation: OP })?
        ];
        // SAFETY: current-context contract; the exact byte slice was validated.
        unsafe {
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(name));
            self.gl
                .get_buffer_sub_data(glow::COPY_READ_BUFFER, offset, &mut out);
        }
        self.driver_error(OP)?;
        Ok(out)
    }

    pub(super) fn read_texture_now(
        &mut self,
        source: GlTextureRegion,
        layout: GlPixelLayout,
    ) -> Result<GlReadback, GlError> {
        use glow::HasContext as _;
        const OP: &str = "read-texture";
        self.assert_ready(OP)?;
        let (name, desc) = self.texture(OP, source.subresource.texture)?;
        source
            .validate_for(desc)
            .map_err(|_| Self::validation(OP, "invalid read region"))?;
        let native_source = native_texture_region(OP, desc, source)?;
        let (gl_format, gl_type) = readback_encoding(desc.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "format has no native readback encoding",
        })?;
        transfer_format(layout.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "pixel encoding has no native readback route",
        })?;
        ensure_single_2d_rect(OP, source)?;
        let needed = layout
            .required_bytes(source)
            .map_err(|_| Self::validation(OP, "invalid pixel layout"))?;
        // GL fills exactly the offset-trimmed body; the layout offset is a
        // client-side placement applied by this module afterwards.
        let body = needed
            .checked_sub(layout.offset)
            .filter(|body| *body > 0)
            .ok_or_else(|| Self::validation(OP, "layout offset leaves no readback body"))?;
        let (level, x, y, width, height) = rect_coords(OP, native_source)?;
        let body_len = usize::try_from(body).map_err(|_| GlError::OutOfMemory { operation: OP })?;
        let needed_len =
            usize::try_from(needed).map_err(|_| GlError::OutOfMemory { operation: OP })?;
        // Scratch framebuffer lifecycle: created, completed, used, and deleted
        // on every path so no half-initialized object survives a failure.
        // SAFETY: current-context contract; the scratch framebuffer exists
        // only inside this block and is unbound/deleted before every return.
        unsafe {
            let Ok(scratch) = self.gl.create_framebuffer() else {
                return Err(GlError::OutOfMemory { operation: OP });
            };
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(scratch));
            self.gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(name),
                level,
            );
            let mut outcome = match self.require_complete(OP) {
                Ok(()) => None,
                Err(error) => Some(Err(error)),
            };
            if outcome.is_none() {
                let saved = self.pixel_store;
                // `place_rows` below reverses these lower-left GL rows into
                // the public upper-left row-zero layout.
                self.apply_pack(layout);
                let mut body_bytes = vec![0u8; body_len];
                self.gl.read_pixels(
                    x,
                    y,
                    width,
                    height,
                    gl_format,
                    gl_type,
                    glow::PixelPackData::Slice(Some(&mut body_bytes)),
                );
                let read = self.driver_error(OP);
                self.restore_pixel_store(UnpackDirection::Pack, saved);
                outcome = match read {
                    Ok(()) => {
                        let mut bytes = vec![0u8; needed_len];
                        place_rows(layout, source, &body_bytes, &mut bytes);
                        Some(Ok(GlReadback { layout, bytes }))
                    }
                    Err(error) => Some(Err(error)),
                };
            }
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.delete_framebuffer(scratch);
            // The teardown must not leak its own error into the caller's next
            // observation.
            let _ = self.gl.get_error();
            outcome.expect("readback outcome was computed on every path")
        }
    }
}

impl NativeGlProvider {
    /// Reads the authenticated WGL/EGL default framebuffer of an acquired
    /// frame. The default framebuffer remains an external drawable: it is
    /// bound only for this owner-thread operation and never enters the owned
    /// framebuffer or texture tables.
    pub(super) fn read_default_framebuffer(
        &mut self,
        target: crate::backend::gl::api::GlDefaultFramebufferTarget,
    ) -> Result<GlReadback, GlError> {
        use glow::HasContext as _;
        const OP: &str = "read-default-framebuffer";
        self.assert_ready(OP)?;
        if target.context != self.context_stamp()
            || target.width == 0
            || target.height == 0
            || target.width > i32::MAX as u32
            || target.height > i32::MAX as u32
            || target.sample_count != 1
            || !matches!(
                target.color_format,
                crate::backend::gl::api::GlFormat::Rgba8Unorm
                    | crate::backend::gl::api::GlFormat::Rgba8Srgb
            )
        {
            return Err(Self::validation(
                OP,
                "unsupported acquired default framebuffer target",
            ));
        }
        let bytes_per_row = target
            .width
            .checked_mul(4)
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        let total = u64::from(bytes_per_row)
            .checked_mul(u64::from(target.height))
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        let len = usize::try_from(total).map_err(|_| GlError::OutOfMemory { operation: OP })?;
        let layout = GlPixelLayout {
            format: GlPixelFormat::Rgba8,
            bytes_per_row,
            rows_per_image: target.height,
            offset: 0,
            alignment: 4,
            repack: GlRepackPolicy::Disallow,
        };
        let mut native_rows = vec![0; len];
        let saved = self.pixel_store;
        // SAFETY: Phase A authenticated the live acquired frame for this
        // owner context. PACK state is restored before this method returns.
        unsafe {
            self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            self.apply_pack(layout);
            self.gl.read_pixels(
                0,
                0,
                target.width as i32,
                target.height as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut native_rows)),
            );
            let result = self.driver_error(OP);
            self.restore_pixel_store(UnpackDirection::Pack, saved);
            self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            result?;
        }
        let mut bytes = vec![0; len];
        place_rows_for_extent(
            layout,
            GlExtent3d {
                width: target.width,
                height: target.height,
                depth_or_layers: 1,
            },
            &native_rows,
            &mut bytes,
        );
        Ok(GlReadback { layout, bytes })
    }
}
