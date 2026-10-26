//! Query records retained by the WebGL2 query provider.

use web_sys::{WebGlProgram, WebGlQuery, WebGlShader};

use super::super::api::{GlError, GlFamilyApi as _, ProgramId, QueryId, ShaderId};
use super::discovery::WebGl2BrowserDiscovery;

pub(super) struct BrowserShader {
    pub(super) generation: u32,
    pub(super) raw: WebGlShader,
}

pub(super) struct BrowserProgram {
    pub(super) generation: u32,
    pub(super) raw: WebGlProgram,
}

pub(super) struct BrowserQuery {
    pub(super) generation: u32,
    pub(super) raw: WebGlQuery,
    /// The target the query last recorded a measurement for.
    pub(super) target: Option<u32>,
    /// A disjoint observation invalidates all outstanding timer slots.
    pub(super) disjoint: bool,
}

impl WebGl2BrowserDiscovery {
    pub(super) fn shader(
        &self,
        operation: &'static str,
        id: ShaderId,
    ) -> Result<&BrowserShader, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.shaders.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(operation, "shader is not live")),
        }
    }

    pub(super) fn program(
        &self,
        operation: &'static str,
        id: ProgramId,
    ) -> Result<&BrowserProgram, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.programs.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(operation, "program is not live")),
        }
    }

    pub(super) fn query(
        &self,
        operation: &'static str,
        id: QueryId,
    ) -> Result<&BrowserQuery, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.queries.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(operation, "query is not live")),
        }
    }

    /// Context loss invalidates query handles; dropping these Rust records
    /// does not issue calls through the lost browser context.
    pub(super) fn clear_executable_state(&mut self) {
        self.programs.clear();
        self.shaders.clear();
        self.queries.clear();
        self.active_query = None;
        self.next_program_slot = 0;
        self.next_query_slot = 0;
        self.next_shader_slot = 0;
        self.timer = None;
    }
}
