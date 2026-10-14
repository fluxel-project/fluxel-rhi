//! Entry point for the textured-quad example.
//!
//! Scene construction, KTX parsing, and platform hosts live in the private
//! module so this file remains the stable Cargo example target.
#[path = "scene.rs"]
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
