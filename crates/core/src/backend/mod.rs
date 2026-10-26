//! Implementation contracts for backend crates.
//!
//! This module contains the Core-to-backend SPI. It is separate from the
//! renderer-facing values and facade methods in [`crate::api`].

mod foundation;
pub use foundation::*;
mod encoder;
pub use encoder::*;
mod submission;
pub use submission::*;
mod resources;
pub use resources::*;
mod operations;
pub use operations::*;
