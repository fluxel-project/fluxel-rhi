//! Portable GPU execution API and backend implementation contract.
//!
//! The root [`api`] module is the statically typed renderer-facing RHI
//! contract. Scheduling, lifetime policy, hazard analysis, and frame policy
//! remain outside Core.
#![deny(missing_docs)]

/// Statically typed public RHI contract for renderers and hosts.
#[allow(missing_docs, dead_code, unused_variables)]
pub mod api;

/// Backend implementation contracts shared with backend crates.
#[doc(hidden)]
pub mod backend;

/// Shared internal utilities used by backend crates.
#[doc(hidden)]
pub mod utils;
