//! Desktop OVR multiview must execute on a context advertising the route.
use super::*;

#[test]
#[ignore = "requires a desktop WGL context with GL_OVR_multiview2"]
fn wgl_core_desktop_multiview() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("Desktop GL multiview", 1024, 768).unwrap()).unwrap();
    // SAFETY: window and context remain on their owner thread until all device work completes.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let d = block_on(open(native));
    assert!(
        d.capabilities().multiview,
        "reported and probed desktop OVR multiview2 must be exposed"
    );
    block_on(super::multiview::multiview(&d));
}
