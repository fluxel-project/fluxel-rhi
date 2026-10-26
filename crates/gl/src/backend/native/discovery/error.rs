//! Structured errors for native current-context discovery.

use super::*;

/// Failure to obtain a complete native discovery record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NativeDiscoveryError {
    /// The current context already carried an error before discovery began.
    ///
    /// Discovery never clears an unknown driver error and continues, because
    /// that would make later observations impossible to attribute.
    PreExistingGlError,
    /// A required driver string was absent or malformed.
    InvalidContextString(&'static str),
    /// The driver reported a profile outside Fluxel's native GL-family scope.
    UnsupportedProfile(String),
    /// A required numeric observation failed.
    QueryFailed(&'static str),
    /// The common snapshot contract rejected otherwise collected facts.
    Snapshot(GlDiscoveryError),
}
