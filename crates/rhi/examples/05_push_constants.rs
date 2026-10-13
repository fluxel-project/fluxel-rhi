//! Entry point for the push-constant sphere-ring example.
//!
//! The private scene module owns the portable workload and backend-specific
//! fallback; this target only preserves Cargo's stable example path.
#[path = "05_push_constants/scene.rs"]
mod scene;

#[allow(unused_imports)]
pub use scene::*;

#[cfg(windows)]
fn main() {
    scene::run();
}

#[cfg(not(windows))]
fn main() {
    scene::run();
}
