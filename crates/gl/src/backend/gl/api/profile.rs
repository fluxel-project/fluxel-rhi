//! Compatibility import for private GL vocabulary during owner migration.
//!
//! The canonical definitions live in `backend::gl::profile`; native and browser
//! owners should import them there as their old descriptor seams disappear.

pub(crate) use crate::backend::gl::profile::{GlProfile as GlFamilyProfile, GlVersion};
