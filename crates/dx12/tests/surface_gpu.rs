//! DXGI presentation smoke covers FIFO/immediate selection and reconfiguration.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::{Dx12, Dx12Instance};
use raw_window_handle::{RawWindowHandle, Win32WindowHandle};
use std::{future::Future, num::NonZeroIsize};
use windows::{
    Win32::{
        Foundation::HWND,
        UI::WindowsAndMessaging::{
            CreateWindowExW, DestroyWindow, WINDOW_EX_STYLE, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
        },
    },
    core::w,
};

fn block<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

struct Window(HWND);
impl Drop for Window {
    fn drop(&mut self) {
        // SAFETY: the fixture owns this HWND and drops its surface first.
        unsafe { DestroyWindow(self.0) }.unwrap();
    }
}

#[test]
fn fifo_and_device_gated_immediate_present() {
    // SAFETY: STATIC is a predefined Win32 class. This test owns the returned
    // window on this thread and keeps it alive until all surface leases drop.
    let window = Window(
        unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE(0),
                w!("STATIC"),
                w!("Fluxel DX12 surface smoke"),
                WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                40,
                40,
                256,
                192,
                None,
                None,
                None,
                None,
            )
        }
        .unwrap(),
    );
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let handle = RawWindowHandle::Win32(Win32WindowHandle::new(
        NonZeroIsize::new(window.0.0 as isize).unwrap(),
    ));
    // SAFETY: window outlives the surface and every acquired image.
    let mut surface = unsafe { Dx12::surface_from_raw_window_handle(&device, handle) }.unwrap();
    let modes = surface.capabilities().unwrap().present_modes;
    assert!(modes.contains(&PresentMode::Fifo));
    for mode in modes {
        block(surface.configure(&SurfaceConfig {
            extent: [128, 96],
            format: TextureFormat::Rgba8Unorm,
            present_mode: mode,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 2,
        }))
        .unwrap();
        for _ in 0..2 {
            let image = block(surface.acquire()).unwrap();
            let generation = image.generation();
            // SAFETY: this backend fixture consumes its exclusive image once;
            // no GPU commands reference the swapchain, whose initial state is
            // PRESENT/COMMON. Native presentation validates each mode's flags.
            let mut native = unsafe { image.into_native() };
            native.try_handoff(generation).unwrap();
            // The fixture calls the native lease directly, so it must also
            // drain presentation before releasing that lease or resizing.
            block(device.wait_idle()).unwrap();
        }
    }
}
