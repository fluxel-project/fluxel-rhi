//! Linux Xlib presentation smoke; the X11 display and window outlive the surface.
#![cfg(all(target_os = "linux", feature = "hardware-tests"))]
#[path = "support/mod.rs"]
mod support;
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::{Vulkan, VulkanInstance};
use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, XcbDisplayHandle, XcbWindowHandle, XlibDisplayHandle,
    XlibWindowHandle,
};
use std::{
    ffi::{c_char, c_int, c_ulong, c_void},
    num::NonZeroU32,
    ptr::NonNull,
    sync::Arc,
};
use support::block;

#[link(name = "libX11.so.6", kind = "dylib", modifiers = "+verbatim")]
unsafe extern "C" {
    fn XOpenDisplay(name: *const c_char) -> *mut c_void;
    fn XDefaultScreen(display: *mut c_void) -> c_int;
    fn XRootWindow(display: *mut c_void, screen: c_int) -> c_ulong;
    fn XCreateSimpleWindow(
        display: *mut c_void,
        parent: c_ulong,
        x: c_int,
        y: c_int,
        width: u32,
        height: u32,
        border: u32,
        border_color: c_ulong,
        background: c_ulong,
    ) -> c_ulong;
    fn XMapWindow(display: *mut c_void, window: c_ulong) -> c_int;
    fn XFlush(display: *mut c_void) -> c_int;
    fn XDestroyWindow(display: *mut c_void, window: c_ulong) -> c_int;
    fn XCloseDisplay(display: *mut c_void) -> c_int;
}
#[link(name = "libX11-xcb.so.1", kind = "dylib", modifiers = "+verbatim")]
unsafe extern "C" {
    fn XGetXCBConnection(display: *mut c_void) -> *mut c_void;
}
#[test]
fn xlib_and_xcb_present() {
    // SAFETY: all Xlib calls use the live display and own window on this thread.
    unsafe {
        let display = XOpenDisplay(std::ptr::null());
        let display_ptr = NonNull::new(display).expect("X11 display required");
        let screen = XDefaultScreen(display);
        let window = XCreateSimpleWindow(
            display,
            XRootWindow(display, screen),
            0,
            0,
            128,
            96,
            0,
            0,
            0,
        );
        assert_ne!(window, 0);
        XMapWindow(display, window);
        XFlush(display);
        let instance = VulkanInstance::new().unwrap();
        let adapters = instance.instance().unwrap().enumerate().unwrap();
        eprintln!("WSI test adapter: {}", adapters[0].name);
        let device = block(instance.open_default()).unwrap();
        let connection = NonNull::new(XGetXCBConnection(display)).unwrap();
        let handles = [
            (
                RawWindowHandle::Xlib(XlibWindowHandle::new(window)),
                RawDisplayHandle::Xlib(XlibDisplayHandle::new(Some(display_ptr), screen)),
            ),
            (
                RawWindowHandle::Xcb(XcbWindowHandle::new(
                    NonZeroU32::new(window as u32).unwrap(),
                )),
                RawDisplayHandle::Xcb(XcbDisplayHandle::new(Some(connection), screen)),
            ),
        ];
        for (raw_window, raw_display) in handles {
            let mut surface =
                Vulkan::surface_from_raw_window_handle(&device, raw_window, raw_display).unwrap();
            let caps = surface.capabilities().unwrap();
            let format = caps.formats[0];
            block(surface.configure(&SurfaceConfig {
                extent: [128, 96],
                format,
                present_mode: PresentMode::Fifo,
                alpha_mode: CompositeAlphaMode::Opaque,
                color_space: ColorSpace::Srgb,
                usage: SurfaceUsage::RENDER_ATTACHMENT,
                maximum_frame_latency: 2,
            }))
            .unwrap();
            let image = block(surface.acquire()).unwrap();
            let lane = device
                .lanes()
                .iter()
                .find(|l| l.class == LaneClass::Graphics)
                .unwrap()
                .id;
            let plan = SubmissionPlan::from_compiled(
                PlanId {
                    device: device.id(),
                    ordinal: 4242,
                },
                Arc::from([StepDesc {
                    lane,
                    kind: StepKind::Present,
                    waits: vec![],
                }]),
            );
            let step = plan.step(0);
            let result = device.queue().submit(SubmitPacket {
                plan,
                steps: vec![BoundStep {
                    step,
                    payload: StepPayload::Present(PresentBinding::new(image)),
                }],
                completion: CompletionRequest::End,
            });
            let receipt = match result {
                SubmitOutcome::Accepted(receipt) => receipt,
                _ => panic!("present submission failed"),
            };
            for checkpoint in &receipt.checkpoints {
                block(device.wait_completion(&checkpoint.point)).unwrap();
            }
            block(device.wait_idle()).unwrap();
            drop(surface);
        }
        drop(device);
        XDestroyWindow(display, window);
        XCloseDisplay(display);
    }
}
