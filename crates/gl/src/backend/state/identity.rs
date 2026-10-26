//! Private, generation-safe keys used by the GL state mirror.
//!
//! These keys identify Fluxel allocation-table entries, never raw GL names or
//! browser handles.  A state-cache hit is valid only within the same context
//! generation; context loss discards the entire mirror.

use crate::backend::api::ContextStamp;

/// One backend-private object allocation within a context generation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GlObjectKey {
    pub(crate) context: ContextStamp,
    pub(crate) slot: u32,
    pub(crate) generation: u32,
}

impl GlObjectKey {
    /// Creates a key from backend-owned allocation metadata.
    pub(crate) const fn new(context: ContextStamp, slot: u32, generation: u32) -> Self {
        Self {
            context,
            slot,
            generation,
        }
    }
}

/// Identity of a bind group packet staged in the GL binding mirror.
///
/// GL has no descriptor-set object, so a native object name is insufficient:
/// the allocation slot and generation distinguish a retired packet from a
/// later packet that reuses the same native bindings.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct BindGroupKey {
    pub(crate) context: ContextStamp,
    pub(crate) slot: u32,
    pub(crate) generation: u32,
}

impl BindGroupKey {
    /// Creates a key from backend-owned bind-group allocation metadata.
    pub(crate) const fn new(context: ContextStamp, slot: u32, generation: u32) -> Self {
        Self {
            context,
            slot,
            generation,
        }
    }
}
