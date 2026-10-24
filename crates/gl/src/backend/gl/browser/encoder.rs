//! Immediate WebGL2 encoder verbs.
//!
//! These methods issue browser GL calls while the owner scope is active. The
//! module retains only live pass bindings and never stores a command stream.

use std::rc::Rc;

use fluxel_rhi_core::api::{ClearColor, Error, Rect, Result, Viewport};
use web_sys::{WebGl2RenderingContext as Gl, WebGlBuffer, WebGlProgram, WebGlVertexArrayObject};

use super::completion::BrowserCompletion;
use super::owner::{
    BrowserBuffer, BrowserEncoder, BrowserOwner, BrowserRasterProgram, BrowserSampler,
    BrowserTexture, BrowserVertexArray,
};

mod attachments;
mod blit;
mod clear_texture;
mod compressed_upload;
mod depth_upload;
pub(super) mod multiview;
pub(super) mod normalized_readback;
mod packed_depth_upload;
mod pbo_readback;
mod raster;
mod resolve;
mod stencil_transfer;
mod surface_color;
mod surface_raster;
pub(crate) use attachments::{
    BrowserColorAttachment, BrowserDepthAttachment, BrowserResolveAttachment,
};

mod upload;

impl BrowserEncoder {
    /// Immediately copies bytes between two buffers. Transfer target bindings
    /// are private scratch state; other domains re-establish their GL state.
    pub(crate) fn copy_buffer(
        &mut self,
        source: &WebGlBuffer,
        source_offset: u64,
        destination: &WebGlBuffer,
        destination_offset: u64,
        bytes: u64,
    ) -> Result<()> {
        self.with_context("copy_buffer", |raw| {
            raw.bind_buffer(Gl::COPY_READ_BUFFER, Some(source));
            raw.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(destination));
            raw.copy_buffer_sub_data_with_f64_and_f64_and_f64(
                Gl::COPY_READ_BUFFER,
                Gl::COPY_WRITE_BUFFER,
                source_offset as f64,
                destination_offset as f64,
                bytes as f64,
            );
        })
    }

    /// Immediately uploads bytes. The borrowed slice is not retained after the
    /// browser call returns.
    pub(crate) fn write_buffer(
        &mut self,
        destination: &WebGlBuffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()> {
        self.with_context("write_buffer", |raw| {
            raw.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(destination));
            raw.buffer_sub_data_with_f64_and_u8_array(Gl::COPY_WRITE_BUFFER, offset as f64, bytes);
        })
    }

    /// Same-owner typed buffer copy for the Core command backend.
    pub(crate) fn copy_browser_buffer(
        &mut self,
        source: &BrowserBuffer,
        source_offset: u64,
        destination: &BrowserBuffer,
        destination_offset: u64,
        bytes: u64,
    ) -> Result<()> {
        self.same_owner(&source.owner)?;
        self.same_owner(&destination.owner)?;
        if source_offset
            .checked_add(bytes)
            .is_none_or(|end| end > source.size)
            || destination_offset
                .checked_add(bytes)
                .is_none_or(|end| end > destination.size)
        {
            return Err(Error::InvalidRange);
        }
        self.copy_buffer(
            &source.raw,
            source_offset,
            &destination.raw,
            destination_offset,
            bytes,
        )
    }

    /// Same-owner typed buffer upload used by static resource carriers.
    pub(crate) fn write_browser_buffer(
        &mut self,
        destination: &BrowserBuffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()> {
        self.same_owner(&destination.owner)?;
        self.write_buffer(&destination.raw, offset, bytes)
    }

    /// Uploads tightly packed RGBA8 texels immediately. `UNPACK_FLIP_Y_WEBGL`
    /// is the native-boundary conversion from the public top-left row order.
    pub(crate) fn write_rgba8_texture(
        &mut self,
        destination: &BrowserTexture,
        mip: u32,
        origin: [u32; 2],
        extent: [u32; 2],
        bytes: &[u8],
    ) -> Result<()> {
        self.same_owner(&destination.owner)?;
        let destination_raw = destination.texture()?;
        let mip_extent = [
            destination
                .desc
                .extent
                .width
                .checked_shr(mip)
                .unwrap_or(0)
                .max(1),
            destination
                .desc
                .extent
                .height
                .checked_shr(mip)
                .unwrap_or(0)
                .max(1),
        ];
        if mip >= destination.desc.mip_level_count
            || extent.contains(&0)
            || origin[0]
                .checked_add(extent[0])
                .is_none_or(|end| end > mip_extent[0])
            || origin[1]
                .checked_add(extent[1])
                .is_none_or(|end| end > mip_extent[1])
            || bytes.len()
                != (extent[0] as usize)
                    .saturating_mul(extent[1] as usize)
                    .saturating_mul(4)
        {
            return Err(Error::InvalidRange);
        }
        let native_y = origin[1];
        self.with_context("write_rgba8_texture", |raw| {
            raw.bind_texture(Gl::TEXTURE_2D, Some(destination_raw));
            // The typed path uploads an ArrayBufferView to TEXTURE_2D only.
            // UNPACK_FLIP_Y_WEBGL is invalid with a PBO source, so this owner
            // clears any private transfer binding before the WebGL call.
            raw.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, None);
            raw.pixel_storei(Gl::UNPACK_ALIGNMENT, 1);
            raw.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, 1);
            let _ = raw.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
                Gl::TEXTURE_2D,
                mip as i32,
                origin[0] as i32,
                native_y as i32,
                extent[0] as i32,
                extent[1] as i32,
                Gl::RGBA,
                Gl::UNSIGNED_BYTE,
                Some(bytes),
            );
            // Restore pixel-store state before later calls can observe it.
            raw.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, 0);
            raw.pixel_storei(Gl::UNPACK_ALIGNMENT, 4);
        })
    }

    /// Reads RGBA8 pixels through a temporary framebuffer for texture-to-buffer
    /// lowering and screenshots. The returned rows are converted back to the
    /// public top-left convention before this synchronous owner call returns.
    pub(crate) fn read_rgba8_texture(
        &mut self,
        source: &BrowserTexture,
        mip: u32,
        origin: [u32; 2],
        extent: [u32; 2],
    ) -> Result<Vec<u8>> {
        self.same_owner(&source.owner)?;
        let source_raw = source.texture()?;
        let mip_extent = [
            source
                .desc
                .extent
                .width
                .checked_shr(mip)
                .unwrap_or(0)
                .max(1),
            source
                .desc
                .extent
                .height
                .checked_shr(mip)
                .unwrap_or(0)
                .max(1),
        ];
        if mip >= source.desc.mip_level_count
            || extent.contains(&0)
            || origin[0]
                .checked_add(extent[0])
                .is_none_or(|end| end > mip_extent[0])
            || origin[1]
                .checked_add(extent[1])
                .is_none_or(|end| end > mip_extent[1])
        {
            return Err(Error::InvalidRange);
        }
        let native_y = mip_extent[1]
            .checked_sub(
                origin[1]
                    .checked_add(extent[1])
                    .ok_or(Error::InvalidRange)?,
            )
            .ok_or(Error::InvalidRange)?;
        let mut bytes = vec![
            0;
            (extent[0] as usize)
                .saturating_mul(extent[1] as usize)
                .saturating_mul(4)
        ];
        let mut state = self.owner.state.borrow_mut();
        state.ensure_encoder(self.epoch)?;
        state.ensure_active()?;
        let framebuffer = state
            .context
            .raw
            .create_framebuffer()
            .ok_or(Error::OutOfMemory)?;
        state
            .context
            .raw
            .bind_framebuffer(Gl::FRAMEBUFFER, Some(&framebuffer));
        state.context.raw.framebuffer_texture_2d(
            Gl::FRAMEBUFFER,
            Gl::COLOR_ATTACHMENT0,
            Gl::TEXTURE_2D,
            Some(source_raw),
            mip as i32,
        );
        let read = state.context.raw.read_pixels_with_opt_u8_array(
            origin[0] as i32,
            native_y as i32,
            extent[0] as i32,
            extent[1] as i32,
            Gl::RGBA,
            Gl::UNSIGNED_BYTE,
            Some(&mut bytes),
        );
        state.context.raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
        state.context.raw.delete_framebuffer(Some(&framebuffer));
        self.issued = true;
        if state.context.raw.is_context_lost() {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        if read.is_err() || state.context.raw.get_error() != Gl::NO_ERROR {
            return Err(Error::Backend(
                "WebGL2 RGBA8 texture readback failed".into(),
            ));
        }
        Ok(bytes)
    }

    /// Starts a raster pass targeting the browser default framebuffer.  The
    /// extent is the acquired drawing-buffer extent, expressed in the public
    /// upper-left coordinate convention.
    pub(crate) fn begin_default_raster(
        &mut self,
        extent: [u32; 2],
        clear: Option<ClearColor>,
        render_area: Rect,
    ) -> Result<()> {
        if render_area
            .x
            .checked_add(render_area.width)
            .is_none_or(|x| x > extent[0])
            || render_area
                .y
                .checked_add(render_area.height)
                .is_none_or(|y| y > extent[1])
            || render_area.width == 0
            || render_area.height == 0
        {
            return Err(Error::InvalidRange);
        }
        {
            let mut state = self.owner.state.borrow_mut();
            state.ensure_encoder(self.epoch)?;
            if state.raster_extent.is_some() {
                return Err(Error::InvalidState);
            }
        }
        self.with_context("begin_default_raster", |raw| {
            raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
            raw.viewport(0, 0, extent[0] as i32, extent[1] as i32);
            // WebGL clears respect scissor and color write masks. A prior pass
            // may have left either state narrowed, while Core clear ops cover
            // the new pass render area independently of prior draw state.
            raw.enable(Gl::SCISSOR_TEST);
            raw.scissor(
                render_area.x as i32,
                (extent[1] - render_area.y - render_area.height) as i32,
                render_area.width as i32,
                render_area.height as i32,
            );
            raw.color_mask(true, true, true, true);
            if let Some(clear) = clear {
                match clear {
                    ClearColor::Float(value) => {
                        raw.clear_color(value[0], value[1], value[2], value[3])
                    }
                    ClearColor::Uint(_) | ClearColor::Sint(_) => {
                        // WebGL2 integer attachment clearing needs a typed
                        // attachment route. The static raster lowering rejects
                        // that unsupported default-framebuffer combination.
                    }
                }
                if matches!(clear, ClearColor::Float(_)) {
                    raw.clear(Gl::COLOR_BUFFER_BIT);
                }
            }
            raw.disable(Gl::SCISSOR_TEST);
        })?;
        self.owner.state.borrow_mut().raster_extent = Some(extent);
        Ok(())
    }

    /// Installs a linked program immediately for the active raster pass.
    pub(crate) fn set_program(&mut self, program: &WebGlProgram) -> Result<()> {
        self.require_raster()?;
        self.with_context("set_program", |raw| raw.use_program(Some(program)))
    }

    /// Same-owner typed program binding for the static raster pipeline carrier.
    pub(crate) fn set_raster_program(&mut self, program: &BrowserRasterProgram) -> Result<()> {
        self.same_owner(&program.owner)?;
        self.set_program(&program.raw)
    }

    /// Installs a vertex array immediately for the active raster pass.
    pub(crate) fn set_vertex_array(&mut self, array: &WebGlVertexArrayObject) -> Result<()> {
        self.require_raster()?;
        self.with_context("set_vertex_array", |raw| raw.bind_vertex_array(Some(array)))
    }

    /// Same-owner typed VAO binding for static vertex-input state.
    pub(crate) fn set_browser_vertex_array(&mut self, array: &BrowserVertexArray) -> Result<()> {
        self.same_owner(&array.owner)?;
        self.set_vertex_array(&array.raw)
    }

    /// Applies a viewport to the upper-first texture storage convention.
    pub(crate) fn set_viewport(&mut self, viewport: Viewport) -> Result<()> {
        let extent = self.require_raster()?;
        let right = viewport.x + viewport.width;
        let bottom = viewport.y + viewport.height;
        if !viewport.x.is_finite()
            || !viewport.y.is_finite()
            || !viewport.width.is_finite()
            || !viewport.height.is_finite()
            || !viewport.min_depth.is_finite()
            || !viewport.max_depth.is_finite()
            || viewport.x < 0.0
            || viewport.y < 0.0
            || viewport.width < 0.0
            || viewport.height < 0.0
            || right > extent[0] as f32
            || bottom > extent[1] as f32
            || !(0.0..=1.0).contains(&viewport.min_depth)
            || !(0.0..=1.0).contains(&viewport.max_depth)
            || viewport.min_depth > viewport.max_depth
        {
            return Err(Error::InvalidRange);
        }
        let y = viewport.y;
        self.with_context("set_viewport", |raw| {
            raw.viewport(
                viewport.x as i32,
                y as i32,
                viewport.width as i32,
                viewport.height as i32,
            );
            raw.depth_range(viewport.min_depth, viewport.max_depth);
        })
    }

    /// Applies a scissor to the upper-first texture storage convention.
    pub(crate) fn set_scissor(&mut self, rect: Rect) -> Result<()> {
        let extent = self.require_raster()?;
        let right = rect.x.checked_add(rect.width).ok_or(Error::InvalidRange)?;
        let bottom = rect.y.checked_add(rect.height).ok_or(Error::InvalidRange)?;
        if right > extent[0]
            || bottom > extent[1]
            || rect.x > i32::MAX as u32
            || rect.width > i32::MAX as u32
            || rect.height > i32::MAX as u32
        {
            return Err(Error::InvalidRange);
        }
        let y = rect.y;
        self.with_context("set_scissor", |raw| {
            raw.enable(Gl::SCISSOR_TEST);
            raw.scissor(
                rect.x as i32,
                y as i32,
                rect.width as i32,
                rect.height as i32,
            );
        })
    }

    /// Directly issues a non-indexed instanced draw; the caller has already
    /// installed topology, program, bindings, and vertex-array state.
    pub(crate) fn draw_arrays(
        &mut self,
        mode: u32,
        first: i32,
        count: i32,
        instances: i32,
    ) -> Result<()> {
        self.require_raster()?;
        self.with_context("draw_arrays", |raw| {
            if instances == 1 {
                raw.draw_arrays(mode, first, count);
            } else {
                raw.draw_arrays_instanced(mode, first, count, instances);
            }
        })
    }

    /// Directly issues an indexed instanced draw. `index_type` is the WebGL
    /// enum selected by static Core `IndexFormat` lowering.
    pub(crate) fn draw_elements(
        &mut self,
        mode: u32,
        count: i32,
        index_type: u32,
        byte_offset: i32,
        instances: i32,
    ) -> Result<()> {
        self.require_raster()?;
        if count < 0 || byte_offset < 0 || instances < 0 {
            return Err(Error::InvalidRange);
        }
        self.with_context("draw_elements", |raw| {
            if instances == 1 {
                raw.draw_elements_with_i32(mode, count, index_type, byte_offset);
            } else {
                raw.draw_elements_instanced_with_i32(
                    mode,
                    count,
                    index_type,
                    byte_offset,
                    instances,
                );
            }
        })
    }

    /// Uses the optional WebGL extension for an indexed draw with base vertex.
    pub(crate) fn draw_elements_base_vertex(
        &mut self,
        mode: u32,
        count: i32,
        index_type: u32,
        byte_offset: i32,
        instances: i32,
        base_vertex: i32,
    ) -> Result<()> {
        use wasm_bindgen::{JsCast, JsValue};
        self.require_raster()?;
        if count < 0 || byte_offset < 0 || instances < 0 {
            return Err(Error::InvalidRange);
        }
        let mut result = Ok(());
        self.with_context("draw_elements_base_vertex", |raw| {
            let extension =
                match raw.get_extension("WEBGL_draw_instanced_base_vertex_base_instance") {
                    Ok(Some(value)) => value,
                    Ok(None) => {
                        result = Err(Error::Unsupported("WebGL base vertex extension"));
                        return;
                    }
                    Err(_) => {
                        result = Err(Error::Backend(
                            "WebGL base vertex extension lookup failed".into(),
                        ));
                        return;
                    }
                };
            let func = js_sys::Reflect::get(
                &extension,
                &JsValue::from_str("drawElementsInstancedBaseVertexBaseInstanceWEBGL"),
            )
            .ok()
            .and_then(|value| value.dyn_into::<js_sys::Function>().ok());
            let Some(func) = func else {
                result = Err(Error::Backend(
                    "WebGL base vertex extension entry point missing".into(),
                ));
                return;
            };
            let args = js_sys::Array::new();
            for value in [
                f64::from(mode),
                f64::from(count),
                f64::from(index_type),
                f64::from(byte_offset),
                f64::from(instances),
                f64::from(base_vertex),
                0.0,
            ] {
                args.push(&JsValue::from_f64(value));
            }
            if func.apply(&extension, &args).is_err() {
                result = Err(Error::Backend(
                    "WebGL base vertex draw threw an exception".into(),
                ));
            }
        })?;
        result
    }

    /// Ends the current raster pass. No command snapshot survives the pass.
    pub(crate) fn end_raster(&mut self) -> Result<()> {
        self.require_raster()?;
        self.owner.state.borrow_mut().raster_extent = None;
        Ok(())
    }

    /// Releases a normally closed immediate scope without inserting hidden
    /// completion work. Submission checkpoints are installed separately by
    /// the SPI only when the caller requested them.
    pub(crate) fn close_scope(mut self) -> Result<()> {
        if self.surface_depth_pass.is_some() || self.offscreen_pass.is_some() {
            return Err(Error::InvalidState);
        }
        let mut state = self.owner.state.borrow_mut();
        state.ensure_encoder(self.epoch)?;
        if state.raster_extent.is_some() {
            return Err(Error::InvalidState);
        }
        state.active_encoder = None;
        self.closed = true;
        Ok(())
    }

    /// Aborts an unclosed scope. Work already issued stays ordered by WebGL.
    pub(crate) fn abort(mut self) {
        // Raster-pass FBOs are encoder-owned. They must be retired before the
        // active epoch is released; otherwise a dropped immediate scope could
        // strand a browser FBO even though its caller-owned attachments remain
        // valid resources.
        if self.surface_depth_pass.is_some() {
            self.abort_surface_depth_raster();
        } else if self.offscreen_pass.is_some() {
            self.abort_offscreen_raster();
        }
        if let Ok(mut state) = self.owner.state.try_borrow_mut() {
            if state.active_encoder == Some(self.epoch) {
                state.active_encoder = None;
            }
        }
        if self.issued {
            self.owner.quarantine_issued();
        }
        self.closed = true;
    }

    // Classify native allocation/attachment failures after releasing their handles.
    pub(super) fn allocation_error(
        state: &mut super::owner::BrowserOwnerState,
        fallback: Error,
    ) -> Error {
        if state.context.raw.is_context_lost() {
            state.mark_lost();
            Error::DeviceLost
        } else {
            fallback
        }
    }

    pub(super) fn with_context(
        &mut self,
        operation: &'static str,
        issue: impl FnOnce(&Gl),
    ) -> Result<()> {
        let mut state = self.owner.state.borrow_mut();
        state.ensure_encoder(self.epoch)?;
        state.ensure_active()?;
        issue(&state.context.raw);
        self.issued = true;
        if state.context.raw.is_context_lost() {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        let error = state.context.raw.get_error();
        if error == Gl::NO_ERROR {
            Ok(())
        } else {
            Err(Error::Backend(format!(
                "WebGL2 {operation} failed with GL error {error}"
            )))
        }
    }

    pub(super) fn require_raster(&self) -> Result<[u32; 2]> {
        let state = self.owner.state.borrow();
        state.ensure_encoder(self.epoch)?;
        state.raster_extent.ok_or(Error::InvalidState)
    }

    pub(super) fn same_owner(&self, other: &BrowserOwner) -> Result<()> {
        if Rc::ptr_eq(&self.owner.state, &other.state) {
            Ok(())
        } else {
            Err(Error::WrongDevice)
        }
    }
}

impl Drop for BrowserEncoder {
    fn drop(&mut self) {
        if !self.closed {
            // `Drop` covers a failed immediate command path that Core did not
            // get to close explicitly. Context loss invalidates every WebGL
            // object, so dropping the JS handles is the only safe cleanup then;
            // otherwise retire our FBO before releasing the active epoch.
            let lost = self.owner.state.borrow().lost;
            if !lost {
                if self.surface_depth_pass.is_some() {
                    self.abort_surface_depth_raster();
                } else if self.offscreen_pass.is_some() {
                    self.abort_offscreen_raster();
                }
            }
            if let Ok(mut state) = self.owner.state.try_borrow_mut() {
                if state.active_encoder == Some(self.epoch) {
                    state.active_encoder = None;
                }
            }
            if self.issued {
                self.owner.quarantine_issued();
            }
        }
    }
}
