
use super::*;
use fluxel_host::{Window, WindowConfig};
use glow::HasContext as _;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
#[test]
#[ignore = "requires an explicit WGL hardware run"]
fn wgl_native_currentness_restores_foreign_context_and_skips_repeat_bind() {
    let wa = Window::new(WindowConfig::new("WGL current A", 32, 32).unwrap()).unwrap();
    let wb = Window::new(WindowConfig::new("WGL current B", 32, 32).unwrap()).unwrap();
    let a = WglContextSurface::open(
        crate::next_context_stamp().unwrap(),
        wa.window_handle().unwrap(),
        wa.display_handle().unwrap(),
        [32, 32],
    )
    .unwrap();
    let b = WglContextSurface::open(
        crate::next_context_stamp().unwrap(),
        wb.window_handle().unwrap(),
        wb.display_handle().unwrap(),
        [32, 32],
    )
    .unwrap();
    b.bind_current("fixture foreign current").unwrap();
    reset_make_current_calls();
    a.with_current("restore A", |gl| {
        // SAFETY: with_current established A before these native commands.
        unsafe {
            gl.clear_color(0.25, 0.5, 0.75, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            let mut pixel = [0u8; 4];
            gl.read_pixels(
                0,
                0,
                1,
                1,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixel)),
            );
            assert_eq!(pixel, [64, 128, 191, 255]);
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(make_current_calls(), 1);
    // SAFETY: thread-local WGL queries do not dereference foreign handles.
    assert_eq!(unsafe { wgl::GetCurrentContext() }, a.hglrc);
    assert_eq!(unsafe { wgl::GetCurrentDC() }, a.hdc);
    for _ in 0..16 {
        a.bind_current("repeat owned current").unwrap();
    }
    assert_eq!(
        make_current_calls(),
        1,
        "actual current native pair needs no repeated bind"
    );
    let tick = std::time::Instant::now();
    for _ in 0..20_000 {
        a.bind_current("currentness diagnostic").unwrap();
    }
    let guard_ns = tick.elapsed().as_nanos();
    let tick = std::time::Instant::now();
    for _ in 0..20_000 {
        // SAFETY: this fixture owns the exact current HDC/HGLRC pair.
        assert_ne!(unsafe { wgl::MakeCurrent(a.hdc, a.hglrc) }, 0);
    }
    let bind_ns = tick.elapsed().as_nanos();
    a.with_current("error-query diagnostic", |gl| {
            let tick=std::time::Instant::now();
            for _ in 0..20_000 {
                // SAFETY: this diagnostic reads only this current context's error flag.
                assert_eq!(unsafe {gl.get_error()},glow::NO_ERROR);
            }
            println!("native-call-diagnostic samples=20000 actual_current_guard_ns={guard_ns} raw_make_current_ns={bind_ns} empty_get_error_ns={} no_renderer_fps_claim=true",tick.elapsed().as_nanos());
            Ok(())
        }).unwrap();
}
