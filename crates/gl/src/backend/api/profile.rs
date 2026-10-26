//! Compatibility import for private GL vocabulary during owner migration.
//!
//! The canonical definitions live in `backend::profile`; native and browser
//! owners should import them there as their old descriptor seams disappear.

pub(crate) use crate::backend::profile::{GlProfile as GlFamilyProfile, GlVersion};
