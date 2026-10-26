//! Context-bound WebGL object wrappers and their final-release deletion.
//!
//! The last shared lifetime token triggers deletion on the browser owner;
//! context loss makes deletion unnecessary because WebGL invalidates names.

use super::*;
use crate::backend::api::{GlTextureDesc, ProgramId};

/// A WebGL buffer tied to exactly one browser owner.
#[derive(Clone)]
pub(crate) struct BrowserBuffer {
    pub(in crate::backend::browser) owner: BrowserOwner,
    pub(in crate::backend::browser) raw: WebGlBuffer,
    pub(in crate::backend::browser) size: u64,
    pub(in crate::backend::browser) lifetime: Rc<()>,
}

/// RGBA8 texture storage tied to exactly one browser owner.
#[derive(Clone)]
pub(crate) struct BrowserTexture {
    pub(in crate::backend::browser) owner: BrowserOwner,
    pub(in crate::backend::browser) raw: Option<WebGlTexture>,
    pub(in crate::backend::browser) renderbuffer: Option<WebGlRenderbuffer>,
    pub(in crate::backend::browser) desc: GlTextureDesc,
    pub(in crate::backend::browser) lifetime: Rc<()>,
}

impl BrowserTexture {
    pub(in crate::backend::browser) fn texture(&self) -> Result<&WebGlTexture> {
        self.raw.as_ref().ok_or(Error::Unsupported(
            "WebGL2 renderbuffer cannot be sampled or copied as a texture",
        ))
    }
}

/// Linked ESSL 3.00 raster program owned by one browser context.
#[derive(Clone)]
pub(crate) struct BrowserRasterProgram {
    pub(in crate::backend::browser) owner: BrowserOwner,
    pub(in crate::backend::browser) id: ProgramId,
    pub(in crate::backend::browser) raw: WebGlProgram,
    pub(in crate::backend::browser) lifetime: Rc<()>,
}

impl BrowserRasterProgram {
    /// Queries the linked program on its owning context, including context loss.
    pub(crate) fn has_uniform(&self, name: &str) -> Result<bool> {
        let mut state = self.owner.state.borrow_mut();
        state.ensure_active()?;
        let found = state
            .context
            .raw
            .get_uniform_location(&self.raw, name)
            .is_some();
        if state.context.raw.is_context_lost() {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        let error = state.context.raw.get_error();
        if error != Gl::NO_ERROR {
            return Err(Error::Backend(format!(
                "WebGL2 get_uniform_location failed with GL error {error}"
            )));
        }
        Ok(found)
    }
}

/// Vertex array state owned by one browser context.
#[derive(Clone)]
pub(crate) struct BrowserVertexArray {
    pub(in crate::backend::browser) owner: BrowserOwner,
    pub(in crate::backend::browser) raw: WebGlVertexArrayObject,
    pub(in crate::backend::browser) lifetime: Rc<()>,
}

/// Sampler state tied to one browser context.
#[derive(Clone)]
pub(crate) struct BrowserSampler {
    pub(in crate::backend::browser) owner: BrowserOwner,
    pub(in crate::backend::browser) raw: WebGlSampler,
    pub(in crate::backend::browser) lifetime: Rc<()>,
}

impl Drop for BrowserBuffer {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            self.owner.delete_buffer(&self.raw);
        }
    }
}

impl Drop for BrowserTexture {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            if let Some(raw) = &self.raw {
                self.owner.delete_texture(raw);
            }
            if let Some(raw) = &self.renderbuffer {
                self.owner.delete_renderbuffer(raw);
            }
        }
    }
}

impl Drop for BrowserRasterProgram {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            self.owner.delete_program(self.id, &self.raw);
        }
    }
}

impl Drop for BrowserVertexArray {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            self.owner.delete_vertex_array(&self.raw);
        }
    }
}

impl Drop for BrowserSampler {
    fn drop(&mut self) {
        if Rc::strong_count(&self.lifetime) == 1 {
            self.owner.delete_sampler(&self.raw);
        }
    }
}
