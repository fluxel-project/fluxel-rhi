//! WGL hardware fixture for the typed immediate GL contract.
//!
//! This is opt-in because it creates a real Win32 window and executes native
//! GL calls. It verifies typed instance/device/surface construction and an
//! empty direct commands step; raster, copy, and readback coverage belong in
//! focused workloads added after their lowerers exist.

#[path = "../common/mod.rs"]
mod common;

use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

use fluxel_host::{Window, WindowConfig};
use fluxel_rhi_core::api::{
    AdapterId, CompletionRequest, Device, DeviceRequest, ImmediateFinishOutcome,
    ImmediateRunOutcome, Instance, InstanceId,
};

use crate::{Gl, GlInstance};

/// Wakes the test owner thread when a future makes completion progress.
struct ThreadWake(std::thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

/// Drives Core's cold device-open future without introducing a test runtime.
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

/// Opens an adopted WGL context as the current typed Core device.
fn device(window: &Window) -> Device<Gl> {
    // SAFETY: `window` is declared before the returned device in the fixture,
    // so reverse drop order destroys every GL object before its HWND/handles.
    // The test opens and drops it on this same owner thread.
    let native = unsafe { GlInstance::from_wgl_window(window, [32, 32]) }
        .expect("open WGL context for typed hardware fixture");
    let instance_id = InstanceId::allocate().expect("allocate GL instance id");
    // SAFETY: `native` remains owned by this Core instance through `open`; the
    // returned device retains its adopted context state after the instance is
    // consumed by this fixture's normal drop order.
    let instance = unsafe { Instance::<Gl>::from_native(instance_id, native) };
    block_on(instance.open(DeviceRequest {
        adapter: AdapterId {
            instance: instance_id,
            ordinal: 0,
        },
        required_features: Vec::new(),
    }))
    .expect("open typed GL device from adopted WGL context")
}

#[test]
#[ignore = "requires an explicit WGL hardware-test run"]
fn wgl_typed_device_surface_and_empty_immediate_step() {
    let window = Window::new(WindowConfig::new("Fluxel RHI GL WGL fixture", 32, 32).unwrap())
        .expect("create WGL fixture window");
    let device = device(&window);
    let fixture = common::ImmediateFixture {
        device: &device,
        lane: device.lanes()[0].id,
    };

    let _surface = common::adopted_surface(&device).expect("adopt typed GL surface");
    let plan = common::single_commands_plan(&fixture, 1).expect("compile typed commands plan");
    // SAFETY: this fixture has one same-device graphics lane and retains the
    // device, surface, and window until `finish` returns its issued prefix.
    let mut session = unsafe { common::begin_commands(&fixture, plan, CompletionRequest::None) }
        .expect("begin GL immediate session");
    let run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin direct GL commands"))
        .close();
    assert!(matches!(run, ImmediateRunOutcome::Issued(_)));
    assert!(matches!(
        session.finish(),
        ImmediateFinishOutcome::Issued(_)
    ));
}
