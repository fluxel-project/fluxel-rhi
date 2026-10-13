//! Entry point for the secondary-command UFO example.
//!
//! Its scene, worker recording, and math live in a private module while this
//! file keeps the stable Cargo example target small.
#[path = "13_multithreaded_recording/scene.rs"]
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
