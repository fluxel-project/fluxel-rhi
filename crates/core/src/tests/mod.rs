//! Contract tests for device provenance, submission, and immediate execution.
//!
//! Test backends are local witnesses for Core invariants; they do not model a
//! native GPU implementation or replace backend conformance testing.
#![allow(missing_docs, dead_code, unused_variables, unused_imports)]
use super::*;
use std::{
    future::Future,
    marker::PhantomData,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[path = "api/binding_arrays.rs"]
mod binding_arrays;
#[path = "api/ergonomics.rs"]
mod ergonomics;
#[path = "api/execution_contract.rs"]
mod execution_contract;
#[path = "api/mesh_validation.rs"]
mod mesh_validation;
#[path = "api/multiqueue_validation.rs"]
mod multiqueue_validation;
#[path = "api/query_validation.rs"]
mod query_validation;
#[path = "api/ray_validation.rs"]
pub(crate) mod ray_validation;
#[path = "api/type_contract_lane.rs"]
mod type_contract_lane;

#[path = "contracts/mod.rs"]
mod contracts;
#[path = "support/mod.rs"]
pub(crate) mod support;

/// Compatibility re-exports for existing Core test modules.
pub(crate) mod type_contract_tests {
    pub(crate) use super::support::*;
}
