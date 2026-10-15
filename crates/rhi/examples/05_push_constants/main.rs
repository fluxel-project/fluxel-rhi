//! Entry point for the push-constant sphere-ring example.
//!
//! The private scene module owns the portable workload and backend-specific
//! fallback; this target only preserves Cargo's stable example path.
#[path = "scene.rs"]
mod scene;

#[allow(unused_imports)]
pub use scene::*;

#[cfg(windows)]
fn main() {
    scene::run();
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    scene::run_metal();
}

#[cfg(not(any(windows, all(target_os = "macos", feature = "metal"))))]
fn main() {
    scene::run();
}
