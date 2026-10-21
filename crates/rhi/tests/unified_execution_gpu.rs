//! Hardware conformance for one reusable plan through the common execution API.
//!
//! This test supplies a fixed plan; it does not implement graph scheduling.
#![cfg(windows)]

use fluxel_rhi::api::*;
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

struct WakeThread(std::thread::Thread);
impl Wake for WakeThread {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

// On panic, wait before the resource owners below are dropped. No step waits
// for idle during the actual execution; this guard also handles test failure.
struct Drain<B: CompletionBackend>(Device<B>);
impl<B: CompletionBackend> Drop for Drain<B> {
    fn drop(&mut self) {
        if let Err(error) = block_on(self.0.wait_idle()) {
            eprintln!("GPU drain failed: {error:?}");
            std::process::abort();
        }
    }
}

fn exercise<B: ExecutionBackend + ReadbackBackend + ResourceBackend>(device: Device<B>) {
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let id = PlanId {
        device: device.id(),
        ordinal: 987655,
    };
    let wait = |index| StepWait {
        previous: StepId { plan: id, index },
        destination: AccessScope {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ.union(AccessMask::COPY_WRITE),
        },
    };
    let steps = Arc::from([
        StepDesc {
            lane,
            kind: StepKind::Commands,
            waits: vec![],
        },
        StepDesc {
            lane,
            kind: StepKind::Commands,
            waits: vec![wait(0)],
        },
        StepDesc {
            lane,
            kind: StepKind::Commands,
            waits: vec![wait(0), wait(1)],
        },
        StepDesc {
            lane,
            kind: StepKind::Commands,
            waits: vec![wait(0), wait(1), wait(2)],
        },
    ]);
    // SAFETY: the fixed topology orders all steps on the same device lane.
    let plan = unsafe { SubmissionPlan::<B>::from_compiled(id, steps) };
    // Declare owners before the guard so even a test panic drains first.
    let mut owners = Vec::new();
    let mut tickets = Vec::new();
    let _drain = Drain(device.clone());
    for run in 0..4u32 {
        let buffer = device
            .create_buffer(&BufferDesc {
                label: Some("unified execution test".into()),
                size: 12,
                usage: BufferUsage::COPY_DST
                    .union(BufferUsage::COPY_SRC)
                    .union(BufferUsage::QUERY_RESOLVE),
                memory: MemoryClass::Device,
            })
            .unwrap();
        owners.push(buffer);
        let buffer = owners.last().unwrap();
        let values = [run * 10 + 1, run * 10 + 2, run * 10 + 3];
        // SAFETY: `owners` retains all buffers until after the drain guard.
        let mut execution =
            unsafe { device.begin_execution(plan.clone(), CompletionRequest::End) }.unwrap();
        for (index, value) in values.iter().enumerate() {
            execution
                .commands(plan.step(index), |encoder| {
                    encoder.write_buffer(buffer, (index * 4) as u64, &value.to_le_bytes())
                })
                .unwrap();
        }
        tickets.push(None);
        execution
            .commands(plan.step(3), |encoder| {
                *tickets.last_mut().unwrap() = Some(encoder.readback_buffer(buffer, 0..12)?);
                Ok(())
            })
            .unwrap();
        // SAFETY: same lifetime and synchronization obligations as begin.
        let outcome = unsafe { execution.finish() };
        let facts = outcome.facts();
        assert_eq!(facts.status, ExecutionStatus::Accepted);
        assert_eq!(
            facts.accepted.unwrap(),
            &(0..4).map(|i| plan.step(i)).collect::<Vec<_>>()
        );
        let checkpoints = facts.checkpoints.unwrap();
        assert_eq!(checkpoints.len(), 1);
        let checkpoint = &checkpoints[0];
        assert_eq!(checkpoint.covers.lane_frontiers, [(lane, plan.step(3))]);
        block_on(device.wait_completion(&checkpoint.point)).unwrap();
        // SAFETY: the observed checkpoint covers the fourth readback step.
        let data = block_on(unsafe {
            device.read_after(
                tickets.last_mut().unwrap().take().unwrap(),
                &checkpoint.point,
            )
        })
        .unwrap();
        let expected: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(data.bytes, expected, "execution {run}");
    }
}

#[cfg(feature = "dx12")]
#[test]
#[ignore = "requires a Windows DX12 GPU"]
fn unified_dx12() {
    use fluxel_rhi::dx12::Dx12Instance;
    exercise(block_on(Dx12Instance::new().unwrap().open_default()).unwrap());
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires a Windows Vulkan GPU"]
fn unified_vulkan() {
    use fluxel_rhi::vulkan::VulkanInstance;
    exercise(block_on(VulkanInstance::new().unwrap().open_default()).unwrap());
}

#[cfg(feature = "native-gl-wgl")]
#[test]
#[ignore = "requires a Windows WGL GPU and window"]
fn unified_wgl() {
    use fluxel_host::{Window, WindowConfig};
    use fluxel_rhi::gl::{Gl, GlInstance};
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    #[repr(C)]
    struct WinRect {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }
    unsafe extern "system" {
        fn GetClientRect(hwnd: *mut core::ffi::c_void, rect: *mut WinRect) -> i32;
    }
    let window =
        Window::new(WindowConfig::new("Unified execution WGL fixture", 32, 32).unwrap()).unwrap();
    let RawWindowHandle::Win32(handle) = window.window_handle().unwrap().as_raw() else {
        panic!("expected Win32 window")
    };
    let mut rect = WinRect {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    };
    // SAFETY: the handle belongs to the live window and rect is writable.
    assert_ne!(
        unsafe { GetClientRect(handle.hwnd.get() as *mut _, &mut rect) },
        0
    );
    let extent = [
        (rect.right - rect.left) as u32,
        (rect.bottom - rect.top) as u32,
    ];
    // SAFETY: window outlives instance, device, resources; all are used and
    // dropped on this thread, which owns the current WGL context.
    let native = unsafe { GlInstance::from_wgl_window(&window, extent) }.unwrap();
    let instance_id = InstanceId::allocate().unwrap();
    // SAFETY: the Core instance owns `native` while opening its device.
    let instance = unsafe { Instance::<Gl>::from_native(instance_id, native) };
    let device = block_on(instance.open(DeviceRequest {
        adapter: AdapterId {
            instance: instance_id,
            ordinal: 0,
        },
        required_features: Vec::new(),
    }))
    .unwrap();
    exercise(device.clone());
    let mut surface = Gl::surface(&device).unwrap();
    let format = *surface.capabilities().unwrap().formats.first().unwrap();
    block_on(surface.configure(&SurfaceConfig {
        extent,
        format,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    }))
    .unwrap();
    let image = block_on(surface.acquire()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let id = PlanId {
        device: device.id(),
        ordinal: 987656,
    };
    // SAFETY: both ordered steps use the same current GL context and the
    // explicit dependency identifies the preceding commands step.
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
    let _drain = Drain(device.clone());
    // SAFETY: the surface and device outlive the test's drain guard.
    let mut execution =
        unsafe { device.begin_execution(plan.clone(), CompletionRequest::End) }.unwrap();
    execution
        .commands(plan.step(0), |encoder| {
            let colors = [Some(ColorAttachment {
                view: AttachmentView::Surface(&image),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0.1, 0.3, 0.6, 1.0])),
                store: StoreOp::Store,
            })];
            let pass = encoder.begin_raster(&RasterPassDesc {
                label: Some("unified present smoke"),
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
            })?;
            pass.finish()
        })
        .unwrap();
    execution
        .present(plan.step(1), PresentBinding::new(image))
        .map_err(|failure| failure.error)
        .unwrap();
    // SAFETY: surface ownership and its issued work are retained through idle.
    let outcome = unsafe { execution.finish() };
    let facts = outcome.facts();
    assert_eq!(facts.status, ExecutionStatus::Accepted);
    assert_eq!(facts.accepted.unwrap(), &[plan.step(0), plan.step(1)]);
    let checkpoints = facts.checkpoints.unwrap();
    assert_eq!(checkpoints.len(), 1);
    block_on(device.wait_completion(&checkpoints[0].point)).unwrap();
}
