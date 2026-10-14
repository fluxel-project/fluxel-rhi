//! Immediate WebGL2 encoder verbs.
//!
//! These methods issue browser GL calls while the owner scope is active. The
//! module retains only live pass bindings and never stores a command stream.

use std::rc::Rc;

use fluxel_rhi_core::api::{ClearColor, Error, Rect, Result, Viewport};
use web_sys::{WebGl2RenderingContext as Gl, WebGlBuffer, WebGlProgram, WebGlVertexArrayObject};

use super::completion::BrowserCompletion;
use super::owner::{
    BrowserBuffer, BrowserEncoder, BrowserFence, BrowserOwner, BrowserRasterProgram,
    BrowserSampler, BrowserTexture, BrowserVertexArray,
};

mod raster;
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
        let native_y = mip_extent[1]
            .checked_sub(
                origin[1]
                    .checked_add(extent[1])
                    .ok_or(Error::InvalidRange)?,
            )
            .ok_or(Error::InvalidRange)?;
        self.with_context("write_rgba8_texture", |raw| {
            raw.bind_texture(Gl::TEXTURE_2D, Some(&destination.raw));
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
            Some(&source.raw),
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
        let row = extent[0] as usize * 4;
        for top in 0..extent[1] as usize / 2 {
            let bottom = extent[1] as usize - 1 - top;
            let (head, tail) = bytes.split_at_mut(bottom * row);
            head[top * row..(top + 1) * row].swap_with_slice(&mut tail[..row]);
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
    ) -> Result<()> {
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

    /// Applies a public upper-left viewport through the GL lower-left native
    /// convention. This conversion is deliberately at the owner boundary so
    /// all Core-facing backends share WebGPU's framebuffer coordinates.
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
        let y = extent[1] as f32 - bottom;
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

    /// Applies a public upper-left scissor through GL's lower-left convention.
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
        let y = extent[1].checked_sub(bottom).ok_or(Error::InvalidRange)?;
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

    /// Ends the current raster pass. No command snapshot survives the pass.
    pub(crate) fn end_raster(&mut self) -> Result<()> {
        self.require_raster()?;
        self.owner.state.borrow_mut().raster_extent = None;
        Ok(())
    }

    /// Closes the scope and returns conservative fence evidence for all issued
    /// work. A fence is never created for an empty encoder.
    pub(crate) fn finish(mut self) -> Result<Option<BrowserCompletion>> {
        let mut state = self.owner.state.borrow_mut();
        state.ensure_encoder(self.epoch)?;
        state.active_encoder = None;
        self.closed = true;
        if !self.issued {
            return Ok(None);
        }
        state.ensure_active()?;
        let Some(raw) = state
            .context
            .raw
            .fence_sync(Gl::SYNC_GPU_COMMANDS_COMPLETE, 0)
        else {
            // Commands were already issued. Without a fence, the owner cannot
            // prove their completion to Core, so keep no usable device rather
            // than allowing later resource retirement to race that prefix.
            state.mark_lost();
            return Err(Error::DeviceLost);
        };
        state.context.raw.flush();
        if state.context.raw.is_context_lost() || state.context.raw.get_error() != Gl::NO_ERROR {
            state.context.raw.delete_sync(Some(&raw));
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        let serial = state.allocate_completion()?;
        state.completions.insert(
            serial,
            BrowserFence {
                raw,
                wakers: Default::default(),
                poll_scheduled: false,
            },
        );
        Ok(Some(BrowserCompletion::pending(self.owner.clone(), serial)))
    }

    /// Aborts an unclosed scope. Work already issued stays ordered by WebGL.
    pub(crate) fn abort(mut self) {
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
