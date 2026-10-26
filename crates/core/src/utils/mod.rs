//! Hidden implementation support shared with backend crates.
//!
//! This module is a backend-only implementation bridge. The renderer-facing
//! Core contract remains in `api`; these helpers do not schedule GPU work and
//! are not a stable application-facing API.

#[doc(hidden)]
pub mod cold_worker;
