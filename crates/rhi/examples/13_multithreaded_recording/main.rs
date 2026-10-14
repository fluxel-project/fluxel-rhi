//! Entry point for the secondary-command UFO example.
//!
//! Its scene, worker recording, and math live in a private module while this
//! file keeps the stable Cargo example target small.
mod scene;

#[allow(unused_imports)]
pub use scene::*;

#[cfg(windows)]
fn main() {
    #[cfg(feature = "native-gl-wgl")]
    if std::env::args()
        .skip_while(|arg| arg != "--backend")
        .nth(1)
        .is_some_and(|backend| backend == "gl")
    {
        if let Err(error) = scene::run_gl_multithreaded_recording() {
            eprintln!("13_multithreaded_recording: {error:?}");
            std::process::exit(1);
        }
        return;
    }
    scene::run();
}

#[cfg(not(windows))]
fn main() {
    scene::run();
}
