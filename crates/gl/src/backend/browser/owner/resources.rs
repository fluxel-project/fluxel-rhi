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
    /// Linked-program metadata is immutable. Cache both present and optimized-
    /// out locations so recurring binding and push writes never query WebGL.
    pub(in crate::backend::browser) uniforms:
        Rc<RefCell<BTreeMap<String, Option<WebGlUniformLocation>>>>,
    pub(in crate::backend::browser) uniform_blocks: Rc<RefCell<BTreeMap<u32, u32>>>,
    pub(in crate::backend::browser) lifetime: Rc<()>,
}

impl BrowserRasterProgram {
    pub(crate) fn uniform_location(&self, name: &str) -> Result<Option<WebGlUniformLocation>> {
        if let Some(location) = self.uniforms.borrow().get(name) {
            return Ok(location.clone());
        }
        let mut state = self.owner.state.borrow_mut();
        state.ensure_active()?;
        let location = state.context.raw.get_uniform_location(&self.raw, name);
        if state.context.raw.is_context_lost() {
            state.mark_lost();
            return Err(Error::DeviceLost);
        }
        if state.context.raw.get_error() != Gl::NO_ERROR {
            return Err(Error::Backend(format!(
                "WebGL2 get_uniform_location failed for {name}"
            )));
        }
        self.uniforms
            .borrow_mut()
            .insert(name.into(), location.clone());
        Ok(location)
    }

    pub(crate) fn uniform_block_size(&self, point: u32) -> Result<u32> {
        if let Some(size) = self.uniform_blocks.borrow().get(&point) {
            return Ok(*size);
        }
        let mut state = self.owner.state.borrow_mut();
        state.ensure_active()?;
        let raw = &state.context.raw;
        let count = raw
            .get_program_parameter(&self.raw, Gl::ACTIVE_UNIFORM_BLOCKS)
            .as_f64()
            .ok_or_else(|| Error::Backend("WebGL2 uniform block query failed".into()))?
            as u32;
        let mut found = None;
        for index in 0..count {
            if raw
                .get_active_uniform_block_parameter(&self.raw, index, Gl::UNIFORM_BLOCK_BINDING)
                .ok()
                .and_then(|v| v.as_f64())
                == Some(f64::from(point))
            {
                found = raw
                    .get_active_uniform_block_parameter(
                        &self.raw,
                        index,
                        Gl::UNIFORM_BLOCK_DATA_SIZE,
                    )
                    .ok()
                    .and_then(|v| v.as_f64())
                    .and_then(|v| {
                        (v > 0. && v <= i32::MAX as f64 && v.fract() == 0.).then_some(v as u32)
                    });
                break;
            }
        }
        let size = found.ok_or(Error::InvalidState)?;
        self.uniform_blocks.borrow_mut().insert(point, size);
        Ok(size)
    }

    /// Queries the linked program on its owning context, including context loss.
    pub(crate) fn has_uniform(&self, name: &str) -> Result<bool> {
        Ok(self.uniform_location(name)?.is_some())
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
