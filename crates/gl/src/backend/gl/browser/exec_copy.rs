//! Browser copy, upload, and readback execution.
//!
//! WebGL2 exposes no direct texture-to-texture copy, so `copy_texture_region`
//! and `read_texture` honestly take the framebuffer route: the source region
//! is attached to a per-operation scratch framebuffer and served with
//! `copyTexSubImage2D` / `readPixels`. This is the backend-provided route for
//! the common copy semantic (plan functional order 3), not a claim that a
//! `glCopyImageSubData`-style command exists. Region coordinates follow the
//! GL bottom-left convention; orientation is a Layer 2/3 concern.

use web_sys::WebGl2RenderingContext as Gl;

use super::super::api::{
    GlBufferRange, GlCopyDomainApi, GlError, GlFormatResourceKind, GlPixelLayout,
    GlPixelStoreState, GlReadback, GlTextureRegion, GlTextureUsage, validate_texture_copy,
};
use super::discovery::WebGl2BrowserDiscovery;

use super::exec_copy_layout::*;
impl GlCopyDomainApi for WebGl2BrowserDiscovery {
    fn copy_buffer_range(
        &mut self,
        source: GlBufferRange,
        destination: GlBufferRange,
    ) -> Result<(), GlError> {
        const OP: &str = "copy-buffer";
        self.assert_provider_ready(OP)?;
        let (source_raw, source_desc) = {
            let entry = self.buffer(OP, source.buffer)?;
            (entry.raw.clone(), entry.desc)
        };
        let (destination_raw, destination_desc) = {
            let entry = self.buffer(OP, destination.buffer)?;
            (entry.raw.clone(), entry.desc)
        };
        source
            .validate_for(source_desc)
            .map_err(|_| validation(OP, "invalid source buffer range"))?;
        destination
            .validate_for(destination_desc)
            .map_err(|_| validation(OP, "invalid destination buffer range"))?;
        if source.size != destination.size {
            return Err(validation(OP, "copy ranges have different sizes"));
        }
        // The copy targets are Layer 1-private scratch: bind immediately before
        // use, no restore on return (`GlCopyDomainApi` documents why).
        self.raw
            .bind_buffer(Gl::COPY_READ_BUFFER, Some(&source_raw));
        self.raw
            .bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&destination_raw));
        self.raw.copy_buffer_sub_data_with_f64_and_f64_and_f64(
            Gl::COPY_READ_BUFFER,
            Gl::COPY_WRITE_BUFFER,
            source.offset as f64,
            destination.offset as f64,
            source.size as f64,
        );
        self.driver_error(OP)
    }

    /// Copies one 2D color rectangle through a scratch framebuffer.
    ///
    /// Preconditions are validated before any state changes: identical
    /// formats, single samples, color aspect, live copy facts, and distinct
    /// mips. Everything else rejects with a structured error.
    fn copy_texture_region(
        &mut self,
        source: GlTextureRegion,
        destination: GlTextureRegion,
    ) -> Result<(), GlError> {
        const OP: &str = "copy-texture";
        self.assert_provider_ready(OP)?;
        let (source_raw, source_desc) = {
            let entry = self.texture(OP, source.subresource.texture)?;
            (entry.raw.clone(), entry.desc)
        };
        let (destination_raw, destination_desc) = {
            let entry = self.texture(OP, destination.subresource.texture)?;
            (entry.raw.clone(), entry.desc)
        };
        validate_texture_copy(source, source_desc, destination, destination_desc)
            .map_err(|_| validation(OP, "invalid texture copy"))?;
        if source.subresource.texture == destination.subresource.texture
            && source.subresource.mip_level == destination.subresource.mip_level
        {
            // Reading and writing one mip is a driver-dependent feedback loop;
            // reject deterministically instead of trusting the driver check.
            return Err(validation(
                OP,
                "copy source and destination name the same mip",
            ));
        }
        if !source_desc.usage.contains(GlTextureUsage::COPY_SOURCE)
            || !destination_desc
                .usage
                .contains(GlTextureUsage::COPY_DESTINATION)
        {
            return Err(validation(
                OP,
                "copy source or destination usage is missing",
            ));
        }
        let facts = self.snapshot.formats();
        let source_copy = facts
            .get_for(GlFormatResourceKind::Texture, source_desc.format, 1)
            .map(|fact| fact.copy_source)
            .unwrap_or(false);
        let destination_copy = facts
            .get_for(GlFormatResourceKind::Texture, destination_desc.format, 1)
            .map(|fact| fact.copy_destination)
            .unwrap_or(false);
        if !source_copy || !destination_copy {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "format lacks a proved copy fact on this context",
            });
        }
        ensure_single_2d_rect(OP, source)?;
        ensure_single_2d_rect(OP, destination)?;
        let (level, dst_x, dst_y, width, height) = rect_coords(OP, destination, destination_desc)?;
        let (_, src_x, src_y, _, _) = rect_coords(OP, source, source_desc)?;
        let src_level = i32::try_from(source.subresource.mip_level)
            .map_err(|_| validation(OP, "source mip level exceeds GLint"))?;

        // Scratch framebuffer lifecycle: created, completed, used, and deleted
        // on every path so no half-initialized object survives a failure.
        let scratch = self
            .raw
            .create_framebuffer()
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        self.raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&scratch));
        self.raw.framebuffer_texture_2d(
            Gl::FRAMEBUFFER,
            Gl::COLOR_ATTACHMENT0,
            Gl::TEXTURE_2D,
            Some(&source_raw),
            src_level,
        );
        if let Err(error) = self.require_complete(OP) {
            self.cleanup_scratch(&scratch);
            return Err(error);
        }
        self.raw
            .bind_texture(Gl::TEXTURE_2D, Some(&destination_raw));
        self.raw.copy_tex_sub_image_2d(
            Gl::TEXTURE_2D,
            level,
            dst_x,
            dst_y,
            src_x,
            src_y,
            width,
            height,
        );
        let result = self.driver_error(OP);
        self.cleanup_scratch(&scratch);
        result
    }

    fn upload_buffer(&mut self, destination: GlBufferRange, bytes: &[u8]) -> Result<(), GlError> {
        const OP: &str = "upload-buffer";
        self.assert_provider_ready(OP)?;
        let (raw, desc) = {
            let entry = self.buffer(OP, destination.buffer)?;
            (entry.raw.clone(), entry.desc)
        };
        destination
            .validate_for(desc)
            .map_err(|_| validation(OP, "invalid destination buffer range"))?;
        if u64::try_from(bytes.len()).ok() != Some(destination.size) {
            return Err(validation(OP, "byte length differs from the range"));
        }
        // COPY_WRITE_BUFFER keeps ARRAY_BUFFER and ELEMENT_ARRAY_BUFFER
        // vertex-state bindings untouched by transfer work, and the two copy
        // targets are the only ones an index buffer can still reach: creation
        // allocates it through ELEMENT_ARRAY_BUFFER, and this bind neither
        // revokes that association nor disturbs any bound vertex array.
        // Measured on the real adapter (AMD Radeon 780M through ANGLE/D3D11;
        // see the 0.15 series plan's 2026-09-17 browser entry) -- which is why
        // an index buffer needs no upload path of its own.
        self.raw.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&raw));
        self.raw.buffer_sub_data_with_f64_and_u8_array(
            Gl::COPY_WRITE_BUFFER,
            destination.offset as f64,
            bytes,
        );
        self.driver_error(OP)
    }

    fn read_buffer(&mut self, source: GlBufferRange) -> Result<Vec<u8>, GlError> {
        const OP: &str = "read-buffer";
        self.assert_provider_ready(OP)?;
        let (raw, desc) = {
            let entry = self.buffer(OP, source.buffer)?;
            (entry.raw.clone(), entry.desc)
        };
        source
            .validate_for(desc)
            .map_err(|_| validation(OP, "invalid source buffer range"))?;
        let mut out = vec![
            0u8;
            usize::try_from(source.size)
                .map_err(|_| GlError::OutOfMemory { operation: OP })?
        ];
        self.raw.bind_buffer(Gl::COPY_READ_BUFFER, Some(&raw));
        self.raw.get_buffer_sub_data_with_f64_and_u8_array(
            Gl::COPY_READ_BUFFER,
            source.offset as f64,
            &mut out,
        );
        self.driver_error(OP)?;
        Ok(out)
    }

    fn upload_texture(
        &mut self,
        destination: GlTextureRegion,
        layout: GlPixelLayout,
        bytes: &[u8],
    ) -> Result<(), GlError> {
        const OP: &str = "upload-texture";
        self.assert_provider_ready(OP)?;
        let (raw, desc) = {
            let entry = self.texture(OP, destination.subresource.texture)?;
            (entry.raw.clone(), entry.desc)
        };
        destination
            .validate_for(desc)
            .map_err(|_| validation(OP, "invalid texture region"))?;
        if desc.format.compressed_info().is_some() {
            return self.upload_compressed_mip(OP, &raw, desc, destination, bytes);
        }
        // Depth storage is filled through clears and FBO work in GL, never
        // from client pixels; every other unmapped format fails closed.
        let Some((gl_format, gl_type)) = super::format_map::upload_encoding(desc.format) else {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "format accepts no CPU pixel upload in WebGL2",
            });
        };
        super::format_map::transfer_format(layout.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "pixel encoding has no WebGL2 transfer route",
        })?;
        ensure_single_2d_rect(OP, destination)?;
        let needed = layout
            .required_bytes(destination)
            .map_err(|_| validation(OP, "invalid pixel layout"))?;
        if u64::try_from(bytes.len()).ok() != Some(needed) {
            return Err(validation(OP, "upload source length differs from layout"));
        }
        let (level, x, y, width, height) = rect_coords(OP, destination, desc)?;
        let payload = staged_unpack(OP, layout, destination, bytes)?;

        let saved = self.pixel_store;
        self.apply_unpack(layout);
        self.raw.bind_texture(Gl::TEXTURE_2D, Some(&raw));
        let result = self
            .raw
            .tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
                Gl::TEXTURE_2D,
                level,
                x,
                y,
                width,
                height,
                gl_format,
                gl_type,
                Some(payload.as_ref()),
            )
            .map_err(|value| js_failure(OP, value))
            .and_then(|()| self.driver_error(OP));
        // Pixel-store restoration is part of this action's state transaction.
        // A failed restore must not be left queued for the next unrelated
        // command, where it would be misattributed and poison the state cache.
        // Preserve the operation's primary error if it already failed, while
        // still drain the restore error before returning it.
        let restored = self.restore_pixel_store_checked(&saved, PixelDirection::Unpack);
        result.and(restored)
    }

    fn read_texture(
        &mut self,
        source: GlTextureRegion,
        layout: GlPixelLayout,
    ) -> Result<GlReadback, GlError> {
        const OP: &str = "read-texture";
        self.assert_provider_ready(OP)?;
        let (raw, desc) = {
            let entry = self.texture(OP, source.subresource.texture)?;
            (entry.raw.clone(), entry.desc)
        };
        source
            .validate_for(desc)
            .map_err(|_| validation(OP, "invalid read region"))?;
        let (gl_format, gl_type) =
            super::format_map::readback_encoding(desc.format).ok_or(GlError::Unsupported {
                operation: OP,
                reason: "format has no WebGL2 readback encoding",
            })?;
        super::format_map::transfer_format(layout.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "pixel encoding has no WebGL2 readback route",
        })?;
        ensure_single_2d_rect(OP, source)?;
        let needed = layout
            .required_bytes(source)
            .map_err(|_| validation(OP, "invalid pixel layout"))?;
        // GL fills exactly the offset-trimmed body; the layout offset is a
        // client-side placement applied by this module afterwards.
        let body = needed
            .checked_sub(layout.offset)
            .filter(|body| *body > 0)
            .ok_or_else(|| validation(OP, "layout offset leaves no readback body"))?;
        let (_, x, y, width, height) = rect_coords(OP, source, desc)?;

        let scratch = self
            .raw
            .create_framebuffer()
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        self.raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&scratch));
        self.raw.framebuffer_texture_2d(
            Gl::FRAMEBUFFER,
            Gl::COLOR_ATTACHMENT0,
            Gl::TEXTURE_2D,
            Some(&raw),
            source.subresource.mip_level as i32,
        );
        if let Err(error) = self.require_complete(OP) {
            self.cleanup_scratch(&scratch);
            return Err(error);
        }
        // readPixels consumes READ_FRAMEBUFFER; the FRAMEBUFFER binding above
        // selects this scratch framebuffer on both ends.
        let saved = self.pixel_store;
        let mut body_bytes =
            vec![0u8; usize::try_from(body).map_err(|_| GlError::OutOfMemory { operation: OP })?];
        self.apply_pack(layout);
        let read = self
            .raw
            .read_pixels_with_opt_u8_array(
                x,
                y,
                width,
                height,
                gl_format,
                gl_type,
                Some(&mut body_bytes),
            )
            .map_err(|value| js_failure(OP, value));
        let result = read.and_then(|()| self.driver_error(OP));
        let restored = self.restore_pixel_store_checked(&saved, PixelDirection::Pack);
        self.cleanup_scratch(&scratch);
        result.and(restored)?;

        let mut bytes =
            vec![0u8; usize::try_from(needed).map_err(|_| GlError::OutOfMemory { operation: OP })?];
        place_rows(layout, source, &body_bytes, &mut bytes);
        Ok(GlReadback { layout, bytes })
    }

    fn pixel_store(&self) -> GlPixelStoreState {
        self.pixel_store
    }
}
