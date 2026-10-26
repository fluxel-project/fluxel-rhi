//! Fluxel RHI's typed public execution contract.
//!
//! This namespace is the sole Core API. RenderGraph supplies scheduling,
//! transitions, hazards, and lifetime policy; backends execute the supplied
//! operations and report native facts.
#![allow(missing_docs, dead_code, unused_variables, unused_imports)]

// API facade implementations use the SPI internally; only `backend` exposes
// these contracts to backend crates.
use crate::backend::*;

mod foundation;
pub use foundation::*;
mod handles;
pub use handles::*;
mod resource;
pub use resource::*;
mod creation;
pub use creation::*;
mod shader;
pub use shader::*;
mod pipeline;
pub use pipeline::*;
mod synchronization;
pub use synchronization::*;
mod pass;
pub use pass::*;
mod encoder;
pub use encoder::*;
mod compute;
pub use compute::*;
mod mesh;
pub use mesh::*;
mod ray;
pub use ray::*;
mod raster_extras;
pub use raster_extras::*;
mod submission;
pub use submission::*;
mod execution_facts;
pub use execution_facts::*;
mod execution;
pub use execution::*;
mod immediate;
pub use immediate::*;
mod sparse;
pub use sparse::*;
mod surface;
pub use surface::*;
mod mapping;
pub use mapping::*;
mod query;
pub use query::*;
mod memory;
pub use memory::*;
mod external;
pub use external::*;
mod secondary;
pub use secondary::*;
mod interop;
pub use interop::*;

#[cfg(test)]
#[path = "../tests/mod.rs"]
mod tests;
