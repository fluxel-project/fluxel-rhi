//! Desktop OVR multiview must execute on a context advertising the route.
use super::*;

#[test]
#[ignore = "requires a desktop WGL context with GL_OVR_multiview2"]
fn wgl_core_desktop_multiview() {
    use fluxel_host::{Window, WindowConfig};
    use std::{
        future::Future,
        task::{Context, Poll, Wake, Waker},
    };
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    fn block_on<F: Future>(f: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }
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
