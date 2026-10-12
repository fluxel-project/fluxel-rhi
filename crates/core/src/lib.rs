//! Portable GPU execution API and backend implementation contract.
//!
//! The root [`api`] module is the statically typed renderer-facing RHI
//! contract. Scheduling, lifetime policy, hazard analysis, and frame policy
//! remain outside Core.
#![deny(missing_docs)]

/// Statically typed public RHI contract and backend SPI.
#[allow(missing_docs, dead_code, unused_variables)]
pub mod api;
