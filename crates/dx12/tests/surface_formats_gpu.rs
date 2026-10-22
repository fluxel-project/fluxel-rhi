//! DXGI presentation smoke covers FIFO/immediate selection and reconfiguration.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::{Dx12, Dx12Instance};
use raw_window_handle::{RawWindowHandle, Win32WindowHandle};
use std::{future::Future, num::NonZeroIsize, sync::Arc};
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
fn flip_formats_configure_acquire_and_present() {
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
    for format in [TextureFormat::Rgb10a2Unorm, TextureFormat::Rgba16Float] {
        assert!(surface.capabilities().unwrap().formats.contains(&format));
        for mode in modes.iter().copied() {
            block(surface.configure(&SurfaceConfig {
                extent: [128, 96],
                format,
                present_mode: mode,
                alpha_mode: CompositeAlphaMode::Opaque,
                color_space: ColorSpace::Srgb,
                usage: SurfaceUsage::RENDER_ATTACHMENT,
                maximum_frame_latency: 2,
            }))
            .unwrap();
            for _ in 0..2 {
                let image = block(surface.acquire()).unwrap();
                let lane = device
                    .lanes()
                    .iter()
                    .find(|lane| lane.class == LaneClass::Graphics)
                    .unwrap()
                    .id;
                let id = PlanId {
                    device: device.id(),
                    ordinal: 831,
                };
                // SAFETY: both steps use this device's graphics lane and present waits for the clear.
                let plan = unsafe {
                    SubmissionPlan::from_compiled(
                        id,
                        Arc::from([
                            StepDesc {
                                lane,
                                kind: StepKind::Commands,
                                waits: vec![],
                            },
                            StepDesc {
                                lane,
                                kind: StepKind::Present,
                                waits: vec![StepWait {
                                    previous: StepId { plan: id, index: 0 },
                                    destination: AccessScope {
                                        stages: StageMask::COLOR_OUTPUT,
                                        accesses: AccessMask::COLOR_WRITE,
                                    },
                                }],
                            },
                        ]),
                    )
                };
                // SAFETY: the surface and image remain alive through completion.
                let mut execution =
                    unsafe { device.begin_execution(plan.clone(), CompletionRequest::End) }
                        .unwrap();
                execution
                    .commands(plan.step(0), |encoder| {
                        let colors = [Some(ColorAttachment {
                            view: AttachmentView::Surface(&image),
                            resolve_target: None,
                            load: LoadOp::Clear(ClearColor::Float([0.1, 0.3, 0.6, 1.0])),
                            store: StoreOp::Store,
                        })];
                        encoder
                            .begin_raster(&RasterPassDesc {
                                label: None,
                                colors: &colors,
                                depth_stencil: None,
                                render_area: Rect {
                                    x: 0,
                                    y: 0,
                                    width: 128,
                                    height: 96,
                                },
                                multiview_mask: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                secondary_inheritance: None,
                            })?
                            .finish()
                    })
                    .unwrap();
                execution
                    .present(plan.step(1), PresentBinding::new(image))
                    .map_err(|e| e.error)
                    .unwrap();
                // SAFETY: the submitted resources stay alive through the wait below.
                let result = unsafe { execution.finish() };
                assert_eq!(result.facts().status, ExecutionStatus::Accepted);
                block(device.wait_idle()).unwrap();
            }
        }
    }
}
