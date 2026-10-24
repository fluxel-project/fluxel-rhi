//! Scoped native COM access records a copy into the actual RHI command list.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};
use windows::{
    Win32::Graphics::{Direct3D12::*, Dxgi::Common::*},
    core::Interface,
};

fn block<F: Future>(future: F) -> F::Output {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
fn scoped_native_copy_on_both_recording_lanes() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let domain = device.native_interop().unwrap();
    assert!(matches!(
        unsafe { domain.acquire(NativeInteropRequest::Encoder) },
        Err(Error::InvalidState)
    ));
    // SAFETY: the guard owns the device interface and we only borrow its COM
    // pointer. A clone adds its own COM reference before the guard is dropped.
    let native = unsafe {
        let guard = domain.acquire(NativeInteropRequest::Device).unwrap();
        ID3D12Device::from_raw_borrowed(&guard.raw_pointer())
            .unwrap()
            .clone()
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
    let allocate = |kind, state| {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: initialized descriptors and an owned resource output.
        unsafe {
            native.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut resource,
            )
        }
        .unwrap();
        resource.unwrap()
    };
    let upload = allocate(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    let readback = allocate(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    let expected = [
        0x13, 0x37, 0x42, 0x99, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    ];
    // SAFETY: only this test accesses the upload resource and its range is 16 bytes.
    unsafe {
        let mut pointer = std::ptr::null_mut();
        upload
            .Map(
                0,
                Some(&D3D12_RANGE { Begin: 0, End: 0 }),
                Some(&mut pointer),
            )
            .unwrap();
        std::ptr::copy_nonoverlapping(expected.as_ptr(), pointer.cast::<u8>(), 16);
        upload.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }));
    }
    for (ordinal, lane) in device.lanes().iter().enumerate() {
        let mut encoder = device.encoder(lane.id).unwrap();
        {
            // SAFETY: foreign recording uses only copy commands with fixed
            // upload/readback states; both resources survive GPU completion.
            let guard = unsafe { encoder.native_interop() }.unwrap();
            unsafe {
                let pointer = guard.raw_pointer();
                let list = ID3D12GraphicsCommandList::from_raw_borrowed(&pointer).unwrap();
                assert_eq!(
                    list.GetType(),
                    if lane.class == LaneClass::Compute {
                        D3D12_COMMAND_LIST_TYPE_COMPUTE
                    } else {
                        D3D12_COMMAND_LIST_TYPE_DIRECT
                    }
                );
                list.CopyBufferRegion(&readback, 0, &upload, 0, 16);
            }
        }
        let work = encoder.finish().unwrap();
        // SAFETY: this is a single independent command step on its exact lane.
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: device.id(),
                    ordinal: 99100 + ordinal as u64,
                },
                Arc::from([StepDesc {
                    lane: lane.id,
                    kind: StepKind::Commands,
                    waits: vec![],
                }]),
            )
        };
        let step = plan.step(0);
        // SAFETY: packet binds the sole step exactly once with compatible work.
        let outcome = unsafe {
            device.queue().submit(SubmitPacket {
                plan,
                steps: vec![BoundStep {
                    step,
                    payload: StepPayload::Commands(vec![work]),
                }],
                completion: CompletionRequest::End,
            })
        };
        let point = match outcome {
            SubmitOutcome::Accepted(receipt) => {
                receipt.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("native copy submission rejected"),
        };
        block(device.wait_completion(&point)).unwrap();
        // SAFETY: completion proves the native copy finished and resources live.
        unsafe {
            let mut pointer = std::ptr::null_mut();
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE { Begin: 0, End: 16 }),
                    Some(&mut pointer),
                )
                .unwrap();
            assert_eq!(
                std::slice::from_raw_parts(pointer.cast::<u8>(), 16),
                expected
            );
            readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
        }
    }
}
