//! Real WSI presentation modes and multiple presents in one packet.
#![cfg(all(windows, feature = "hardware-tests"))]
#[path = "support/mod.rs"]
mod support;
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::{Vulkan, VulkanInstance};
use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowsDisplayHandle,
};
use std::{num::NonZeroIsize, sync::Arc};
use support::block;

#[repr(C)]
struct RectNative {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}
#[link(name = "user32")]
unsafe extern "system" {
    fn CreateWindowExW(
        ex: u32,
        class: *const u16,
        name: *const u16,
        style: u32,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        parent: isize,
        menu: isize,
        instance: isize,
        param: *const (),
    ) -> isize;
    fn DestroyWindow(window: isize) -> i32;
    fn GetClientRect(window: isize, rect: *mut RectNative) -> i32;
}
struct Window(isize);
impl Drop for Window {
    fn drop(&mut self) {
        unsafe {
            DestroyWindow(self.0);
        }
    }
}
fn window() -> Window {
    let class: Vec<u16> = "STATIC\0".encode_utf16().collect();
    let title: Vec<u16> = "Fluxel Vulkan surface\0".encode_utf16().collect();
    // SAFETY: STATIC is a system class; the returned window is kept alive by Window.
    let handle = unsafe {
        CreateWindowExW(
            0,
            class.as_ptr(),
            title.as_ptr(),
            0x10cf0000,
            30,
            30,
            256,
            192,
            0,
            0,
            0,
            std::ptr::null(),
        )
    };
    assert_ne!(handle, 0);
    Window(handle)
}
#[test]
fn runtime_modes_and_multiple_present_steps() {
    let windows = [window(), window()];
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let mut surfaces: Vec<_> = windows
        .iter()
        .map(|window| {
            let raw = RawWindowHandle::Win32(Win32WindowHandle::new(
                NonZeroIsize::new(window.0).unwrap(),
            ));
            // SAFETY: both owned windows outlive their surfaces and image leases.
            unsafe {
                Vulkan::surface_from_raw_window_handle(
                    &device,
                    raw,
                    RawDisplayHandle::Windows(WindowsDisplayHandle::new()),
                )
            }
            .unwrap()
        })
        .collect();
    let mut extents = Vec::new();
    for window in &windows {
        let mut rect = RectNative {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        // SAFETY: window is live and rect is a valid writable structure.
        assert_ne!(unsafe { GetClientRect(window.0, &mut rect) }, 0);
        extents.push([
            (rect.right - rect.left) as u32,
            (rect.bottom - rect.top) as u32,
        ]);
    }
    let caps = surfaces[0].capabilities().unwrap();
    eprintln!("Vulkan WSI present modes: {:?}", caps.present_modes);
    assert!(caps.present_modes.contains(&PresentMode::Fifo));
    let format = *caps
        .formats
        .iter()
        .find(|f| surfaces[1].capabilities().unwrap().formats.contains(f))
        .unwrap();
    let msaa_desc = |extent: [u32; 2]| TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 2,
        format,
        mip_levels: 1,
        samples: 4,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    };
    if extents.iter().any(|&extent| {
        matches!(
            device.texture_support(&msaa_desc(extent)),
            Support::Unsupported(_)
        )
    }) {
        eprintln!("SKIP: 4x MSAA array attachment unavailable for this surface format");
        return;
    }
    for (iteration, mode) in caps.present_modes.iter().enumerate() {
        if !surfaces[1]
            .capabilities()
            .unwrap()
            .present_modes
            .contains(mode)
        {
            continue;
        }
        let mut images = Vec::new();
        for (surface, extent) in surfaces.iter_mut().zip(&extents) {
            block(surface.configure(&SurfaceConfig {
                extent: *extent,
                format,
                present_mode: *mode,
                alpha_mode: CompositeAlphaMode::Opaque,
                color_space: ColorSpace::Srgb,
                usage: SurfaceUsage::RENDER_ATTACHMENT,
                maximum_frame_latency: 2,
            }))
            .unwrap();
            images.push(block(surface.acquire()).unwrap());
        }
        let mut work = Vec::new();
        for (image, extent) in images.iter().zip(&extents) {
            let source = device.create_texture(&msaa_desc(*extent)).unwrap();
            let source_view = device
                .create_view(
                    &source,
                    &TextureViewDesc {
                        format,
                        dimension: TextureViewDimension::D2,
                        aspects: TextureAspects::COLOR,
                        mips: 0..1,
                        layers: 1..2,
                    },
                )
                .unwrap();
            let mut encoder = device.encoder(lane).unwrap();
            encoder
                .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
                    target: Subresources {
                        texture: &source,
                        aspects: TextureAspects::COLOR,
                        mips: 0..1,
                        layers: 1..2,
                    },
                    usage: StateTransition {
                        from: TextureUses {
                            stages: StageMask::EMPTY,
                            accesses: AccessMask::EMPTY,
                        },
                        to: TextureUses {
                            stages: StageMask::COLOR_OUTPUT,
                            accesses: AccessMask::COLOR_WRITE,
                        },
                    },
                    layout: Some((ImageLayout::Undefined, ImageLayout::ColorAttachment)),
                    ownership: None,
                })])
                .unwrap();
            let colors = [Some(ColorAttachment {
                view: AttachmentView::Texture(&source_view),
                resolve_target: Some(AttachmentView::Surface(image)),
                load: LoadOp::Clear(ClearColor::Float([0.2, 0.4, 0.8, 1.0])),
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
                        width: extent[0],
                        height: extent[1],
                    },
                    multiview_mask: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    secondary_inheritance: None,
                })
                .unwrap()
                .finish()
                .unwrap();
            let loaded = [Some(ColorAttachment {
                view: AttachmentView::Surface(image),
                resolve_target: None,
                load: LoadOp::Load,
                store: StoreOp::Store,
            })];
            encoder
                .begin_raster(&RasterPassDesc {
                    label: None,
                    colors: &loaded,
                    depth_stencil: None,
                    render_area: Rect {
                        x: 0,
                        y: 0,
                        width: extent[0],
                        height: extent[1],
                    },
                    multiview_mask: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    secondary_inheritance: None,
                })
                .unwrap()
                .finish()
                .unwrap();
            work.push(encoder.finish().unwrap());
        }
        let plan_id = PlanId {
            device: device.id(),
            ordinal: 104000 + iteration as u64,
        };
        let present_lane = device
            .lanes()
            .iter()
            .find(|info| info.class == LaneClass::Compute)
            .map_or(lane, |info| info.id);
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                plan_id,
                Arc::from([
                    StepDesc {
                        lane,
                        kind: StepKind::Commands,
                        waits: vec![],
                    },
                    StepDesc {
                        lane,
                        kind: StepKind::Present,
                        waits: vec![],
                    },
                    StepDesc {
                        lane,
                        kind: StepKind::Commands,
                        waits: vec![],
                    },
                    StepDesc {
                        lane: present_lane,
                        kind: StepKind::Present,
                        waits: vec![StepWait {
                            previous: StepId {
                                plan: plan_id,
                                index: 2,
                            },
                            destination: AccessScope {
                                stages: StageMask::ALL_COMMANDS,
                                accesses: AccessMask::ALL,
                            },
                        }],
                    },
                ]),
            )
        };
        let ids = [plan.step(0), plan.step(1), plan.step(2), plan.step(3)];
        let mut work = work.into_iter();
        let mut images = images.into_iter();
        // SAFETY: queue order and the explicit cross-lane wait cover each image's render work.
        let outcome = unsafe {
            device.queue().submit(SubmitPacket {
                plan,
                steps: vec![
                    BoundStep {
                        step: ids[0],
                        payload: StepPayload::Commands(vec![work.next().unwrap()]),
                    },
                    BoundStep {
                        step: ids[1],
                        payload: StepPayload::Present(PresentBinding::new(images.next().unwrap())),
                    },
                    BoundStep {
                        step: ids[2],
                        payload: StepPayload::Commands(vec![work.next().unwrap()]),
                    },
                    BoundStep {
                        step: ids[3],
                        payload: StepPayload::Present(PresentBinding::new(images.next().unwrap())),
                    },
                ],
                completion: CompletionRequest::End,
            })
        };
        let receipt = match outcome {
            SubmitOutcome::Accepted(receipt) => receipt,
            _ => panic!("present packet rejected"),
        };
        assert_eq!(receipt.presents.len(), 2);
        block(device.wait_completion(&receipt.checkpoints[0].point)).unwrap();
        block(device.wait_idle()).unwrap();
    }
}
