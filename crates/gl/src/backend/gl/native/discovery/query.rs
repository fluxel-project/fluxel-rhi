//! Query vocabulary shared by native discovery and its probes.

/// Small mockable subset of native GL used by discovery.
///
/// Implementations must return `None` for a failed query, including a GL
/// error. This is deliberately not a general command interface.
pub(crate) trait NativeGlQuery {
    /// Consumes exactly one pending driver error, returning true when it was
    /// not `GL_NO_ERROR`.
    fn take_error(&self) -> bool;
    fn string(&self, name: u32) -> Option<String>;
    fn integer(&self, name: u32) -> Option<i64>;
    fn integer_pair(&self, name: u32) -> Option<[i64; 2]>;
    fn indexed_integer(&self, name: u32, index: u32) -> Option<i64>;
    fn float(&self, name: u32) -> Option<f32>;
    fn indexed_string(&self, name: u32, index: u32) -> Option<String>;
    /// A parameter of the draw framebuffer's attachment, or `None` when the
    /// driver refused the query.
    ///
    /// A separate entry point rather than an argument to `integer`, because it
    /// is a different GL command and not a different parameter: this is the one
    /// place a core profile still answers what the default framebuffer's
    /// component widths are. The target is always the draw framebuffer, so it
    /// is not a parameter -- `DRAW_FRAMEBUFFER_BINDING` above is the query that
    /// says which object that is.
    fn drawable_attachment(&self, attachment: u32, name: u32) -> Option<i64>;
}
