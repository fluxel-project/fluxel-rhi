//! Real GPU proof that an independent Direct checkpoint need not wait for Compute.
//!
//! A native shared fence deliberately holds the Compute queue.  The Direct
//! list writes and copies unrelated data.  The observation is the backend's
//! own Direct completion point, not a CPU delay or a synthetic readiness flag.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::{Dx12ExternalHandle, Dx12Instance};
use std::{
    future::Future,
    os::windows::io::{FromRawHandle, OwnedHandle},
    sync::{Arc, mpsc},
    thread,
    time::{Duration, Instant},
};
use windows::{
    Win32::{
        Foundation::GENERIC_ALL,
        Graphics::{Direct3D12::*, Dxgi::Common::*},
    },
    core::{Interface, PCWSTR},
};

fn block<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            // This is test-only completion observation. DX12's future itself
            // uses a fence event; this loop never drives GPU execution.
            std::task::Poll::Pending => thread::sleep(Duration::from_millis(1)),
        }
    }
}

fn native_device(device: &Device<fluxel_rhi_dx12::Dx12>) -> ID3D12Device {
    let domain = device.native_interop().expect("DX12 native interop");
    // SAFETY: the guard keeps the RHI-owned device interface alive while this
    // fixture takes its own COM reference.
    unsafe {
        let guard = domain.acquire(NativeInteropRequest::Device).unwrap();
        ID3D12Device::from_raw_borrowed(&guard.raw_pointer())
            .unwrap()
            .clone()
    }
}

unsafe fn owned(handle: windows::Win32::Foundation::HANDLE) -> OwnedHandle {
    // SAFETY: CreateSharedHandle transfers one owned NT handle to this process.
    unsafe { OwnedHandle::from_raw_handle(handle.0) }
}

/// Ensures a deliberately held test fence cannot strand a native queue if an
/// assertion or setup call unwinds before the normal release point.
struct GateRelease {
    fence: ID3D12Fence,
    released: bool,
}

impl GateRelease {
    fn new(fence: ID3D12Fence) -> Self {
        Self {
            fence,
            released: false,
        }
    }

    fn release(&mut self) {
        if !std::mem::replace(&mut self.released, true) {
            // SAFETY: this fixture owns the one producer signal for value 1.
            unsafe { self.fence.Signal(1) }.expect("release held Compute gate");
        }
    }
}

impl Drop for GateRelease {
    fn drop(&mut self) {
        if !std::mem::replace(&mut self.released, true) {
            // SAFETY: this best-effort cleanup owns the same single signal. Do
            // not panic during unwinding if a device has already been lost.
            let _ = unsafe { self.fence.Signal(1) };
        }
    }
}

fn shared_buffer(native: &ID3D12Device) -> (ID3D12Resource, OwnedHandle) {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 16,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut resource = None;
    unsafe {
        native.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_SHARED,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut resource,
        )
    }
    .unwrap();
    let resource = resource.expect("shared buffer allocation");
    let handle =
        unsafe { native.CreateSharedHandle(&resource, None, GENERIC_ALL.0, PCWSTR::null()) }
            .unwrap();
    // SAFETY: CreateSharedHandle returned the sole owned handle for this test.
    (resource, unsafe { owned(handle) })
}

#[test]
fn independent_direct_checkpoint_escapes_a_held_compute_queue() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let direct = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .expect("DX12 direct lane")
        .id;
    let compute = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Compute)
        .expect("DX12 compute lane")
        .id;
    let native = native_device(&device);
    let external = device.external().expect("DX12 external bridge");

    // The Compute encoder receives a real native Queue::Wait for value 1.
    // Keep the owning COM fence/resource alive until it is released below.
    let gate = unsafe { native.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_SHARED) }.unwrap();
    let mut gate_release = GateRelease::new(gate.clone());
    let gate_handle =
        unsafe { native.CreateSharedHandle(&gate, None, GENERIC_ALL.0, PCWSTR::null()) }.unwrap();
    let (_gate_resource, resource_handle) = shared_buffer(&native);
    let memory = unsafe {
        external.import_memory(
            Dx12ExternalHandle::Resource(resource_handle),
            &ExternalResourceDesc {
                size: 16,
                offset: 0,
                usage: BufferUsage::COPY_DST,
            },
        )
    }
    .unwrap();
    let wait = unsafe {
        external.import_sync(Dx12ExternalHandle::Fence {
            // SAFETY: the native fence and this transferred handle remain valid
            // until the test signals value 1 after the bounded observation.
            handle: owned(gate_handle),
            value: 1,
        })
    }
    .unwrap();

    let mut held_compute = device.encoder(compute).unwrap();
    external.acquire(&mut held_compute, &memory, wait).unwrap();

    let expected = [0x51, 0xA7, 0x3C, 0xE1];
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: expected.len() as u64,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: expected.len() as u64,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut independent_direct = device.encoder(direct).unwrap();
    independent_direct
        .write_buffer(&output, 0, &expected)
        .unwrap();
    independent_direct
        .copy_buffer(&output, 0, &readback, 0, expected.len() as u64)
        .unwrap();
    let ticket = independent_direct
        .readback_buffer(&readback, 0..expected.len() as u64)
        .unwrap();

    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 99_930,
            },
            Arc::from([
                StepDesc {
                    lane: compute,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane: direct,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
            ]),
        )
    };
    let compute_step = plan.step(0);
    let direct_step = plan.step(1);
    // SAFETY: both lists are submitted once to their declared independent
    // lanes. There is intentionally no StepWait between them.
    let receipt = match unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![
                BoundStep {
                    step: compute_step,
                    payload: StepPayload::Commands(vec![held_compute.finish().unwrap()]),
                },
                BoundStep {
                    step: direct_step,
                    payload: StepPayload::Commands(vec![independent_direct.finish().unwrap()]),
                },
            ],
            completion: CompletionRequest::Steps(Arc::from([direct_step])),
        })
    } {
        SubmitOutcome::Accepted(receipt) => receipt,
        _ => panic!("independent cross-queue submit rejected"),
    };
    let checkpoint = receipt
        .checkpoints
        .into_iter()
        .find(|checkpoint| {
            checkpoint
                .covers
                .lane_frontiers
                .contains(&(direct, direct_step))
        })
        .expect("requested Direct checkpoint");
    let point = checkpoint.point;

    let (sender, receiver) = mpsc::sync_channel(1);
    let observer_device = device.clone();
    let observer_point = point.clone();
    let observer = thread::spawn(move || {
        let _ = sender.send(block(observer_device.wait_completion(&observer_point)));
    });
    let observation_started = Instant::now();
    let before_release = receiver.recv_timeout(Duration::from_millis(500));
    let observation_elapsed = observation_started.elapsed();
    match &before_release {
        Ok(Ok(())) => println!(
            "queue_overlap_before_gate=unblocked observation_ms={}",
            observation_elapsed.as_millis()
        ),
        Ok(Err(error)) => panic!("Direct checkpoint failed before gate release: {error}"),
        Err(mpsc::RecvTimeoutError::Timeout) => println!(
            "queue_overlap_before_gate=blocked observation_ms={}",
            observation_elapsed.as_millis()
        ),
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("Direct checkpoint observer stopped"),
    }
    let unblocked_before_release = before_release.is_ok();

    // Always release the native GPU gate before asserting, including on the
    // old implicit-join baseline, so no queue or observer remains stuck.
    gate_release.release();
    let completion = match before_release {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("Direct checkpoint did not complete after gate release"),
        Err(mpsc::RecvTimeoutError::Disconnected) => unreachable!(),
    };
    completion.unwrap();
    observer.join().unwrap();

    // SAFETY: the requested Direct checkpoint covers this Direct-only copy.
    let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(bytes, expected);
    block(device.wait_idle()).unwrap();

    assert!(
        unblocked_before_release,
        "Direct completion remained blocked behind an unrelated held Compute queue"
    );
}
