//! GL test-only evidence vocabulary.
//!
//! Context creation remains private to the WGL, EGL, or WebGL2 adapter. This
//! module stores only evidence gathered after a typed `Device<Gl>` exists.

/// Profile facts captured by one real GL context fixture.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlContextEvidence {
    /// API family and observed version, such as `OpenGL 4.6` or `WebGL 2.0`.
    pub profile: String,
    /// Driver or browser renderer identity.
    pub renderer: String,
    /// Enabled extensions relevant to the executed workload.
    pub extensions: Vec<String>,
    /// Framebuffer extent used for the evidence capture.
    pub extent: [u32; 2],
}

/// CPU-visible evidence emitted by a completed typed immediate workload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlWorkloadEvidence {
    /// Context facts captured before recording.
    pub context: GlContextEvidence,
    /// Workload name, for example `offscreen-rgba8`.
    pub workload: String,
    /// Completion was observed before readback or resource retirement.
    pub completion_observed: bool,
    /// Optional upper-left-row-zero RGBA8 output after GL boundary conversion.
    pub rgba8: Option<Vec<u8>>,
}

impl GlWorkloadEvidence {
    /// Creates evidence before a workload has produced an optional readback.
    pub fn new(context: GlContextEvidence, workload: impl Into<String>) -> Self {
        Self {
            context,
            workload: workload.into(),
            completion_observed: false,
            rgba8: None,
        }
    }
}

#[cfg(test)]
#[path = "test_support__tests.rs"]
mod tests;
