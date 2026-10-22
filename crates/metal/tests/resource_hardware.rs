//! Apple GPU resource allocation and host mapping checks.
#![cfg(all(target_vendor = "apple", feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_metal::{Metal, MetalInstance};
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
fn run_copy(device: &Device<Metal>, src: &Buffer<Metal>, bytes: usize) -> Vec<u8> {
    let lane = device.lanes()[0].id;
    let dst = device
        .create_buffer(&BufferDesc {
            label: None,
            size: bytes as u64,
            usage: BufferUsage::COPY_DST.union(BufferUsage::COPY_SRC),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    encoder.copy_buffer(src, 0, &dst, 0, bytes as u64).unwrap();
    let ticket = encoder.readback_buffer(&dst, 0..bytes as u64).unwrap();
    let work = encoder.finish().unwrap();
    submit_readback(device, lane, work, ticket)
}

fn submit_readback(
    device: &Device<Metal>,
    lane: LaneId,
    work: RecordedWork<Metal>,
    ticket: ReadbackTicket<Metal>,
) -> Vec<u8> {
    // SAFETY: the work is recorded on this device and its matching graphics lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 1,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: the bound work belongs to this plan step and device.
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
    let receipt = match outcome {
        SubmitOutcome::Accepted(r) => r,
        SubmitOutcome::Rejected { error, .. }
        | SubmitOutcome::PartiallyAccepted { error, .. }
        | SubmitOutcome::Indeterminate { error, .. } => panic!("submit: {error:?}"),
    };
    let point = &receipt.checkpoints[0].point;
    block_on(device.wait_completion(point)).unwrap();
    // SAFETY: the waited end point covers the encoded readback.
    block_on(unsafe { device.read_after(ticket, point) })
        .unwrap()
        .bytes
}

#[test]
fn mapped_placement_upload_survives_heap_handle_drop_and_gpu_copy() {
    let device = block_on(MetalInstance::new().unwrap().open_default()).unwrap();
    let Some(memory) = device.memory() else {
        return;
    };
    let desc = BufferDesc {
        label: None,
        size: 64,
        usage: BufferUsage::MAP_WRITE.union(BufferUsage::COPY_SRC),
        memory: MemoryClass::Upload,
    };
    let req = memory.buffer_requirements(&desc).unwrap();
    let class = req.native_type_bits.trailing_zeros();
    assert!(class < 64 && !req.dedicated_only);
    let heap = memory
        .heap(&HeapDesc {
            size: req.size,
            native_memory_class: class,
        })
        .unwrap();
    let buffer = memory.place_buffer(&heap, 0, &desc).unwrap();
    let expected: Vec<u8> = (0..64).map(|i| i ^ 0xa5).collect();
    // SAFETY: no GPU work or other host mapping accesses this newly created allocation.
    let mut lease = block_on(unsafe { device.map_buffer(&buffer, 0..64, MapMode::Write) }).unwrap();
    lease.bytes_mut().unwrap().copy_from_slice(&expected);
    lease.flush(0..64).unwrap();
    drop(lease);
    drop(heap);
    assert_eq!(run_copy(&device, &buffer, expected.len()), expected);
}

#[test]
fn placed_buffers_serialize_host_mapping() {
    let device = block_on(MetalInstance::new().unwrap().open_default()).unwrap();
    let Some(memory) = device.memory() else {
        return;
    };
    let desc = BufferDesc {
        label: None,
        size: 64,
        usage: BufferUsage::MAP_WRITE.union(BufferUsage::COPY_SRC),
        memory: MemoryClass::Upload,
    };
    let req = memory.buffer_requirements(&desc).unwrap();
    let second_offset = req.size.div_ceil(req.alignment) * req.alignment;
    let heap = memory
        .heap(&HeapDesc {
            size: second_offset + req.size,
            native_memory_class: req.native_type_bits.trailing_zeros(),
        })
        .unwrap();
    let first = memory.place_buffer(&heap, 0, &desc).unwrap();
    let second = memory.place_buffer(&heap, second_offset, &desc).unwrap();
    // SAFETY: no GPU operations access this heap; the backend serializes host leases.
    let lease = block_on(unsafe { device.map_buffer(&first, 0..64, MapMode::Write) }).unwrap();
    // SAFETY: the active lease is held, so a successful second lease would be a bug.
    let second_result = block_on(unsafe { device.map_buffer(&second, 0..64, MapMode::Write) });
    assert!(matches!(second_result, Err(Error::InvalidState)));
    drop(lease);
    // SAFETY: the previous lease is dropped and no GPU operations access the heap.
    assert!(block_on(unsafe { device.map_buffer(&second, 0..64, MapMode::Write) }).is_ok());
}

#[test]
fn compressed_texture_upload_and_readback_preserve_block_bytes() {
    let device = block_on(MetalInstance::new().unwrap().open_default()).unwrap();
    let lane = device.lanes()[0].id;
    for format in [TextureFormat::Bc1RgbaUnorm, TextureFormat::Etc2Rgb8Unorm] {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_DST.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        };
        if !matches!(device.texture_support(&desc), Support::Supported(_)) {
            continue;
        }
        let texture = device.create_texture(&desc).unwrap();
        let buffer = device
            .create_buffer(&BufferDesc {
                label: None,
                size: 256,
                usage: BufferUsage::COPY_DST.union(BufferUsage::COPY_SRC),
                memory: MemoryClass::Device,
            })
            .unwrap();
        let expected = [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
        let mut encoder = device.encoder(lane).unwrap();
        let region = || TextureRegion {
            texture: &texture,
            mip: 0,
            origin: [0, 0, 0],
            extent: [4, 4, 1],
            aspect: TextureAspects::COLOR,
        };
        encoder
            .write_texture(
                region(),
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: 8,
                    rows_per_image: 1,
                },
                &expected,
            )
            .unwrap();
        encoder
            .texture_to_buffer(
                region(),
                &buffer,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: 256,
                    rows_per_image: 1,
                },
            )
            .unwrap();
        let ticket = encoder.readback_buffer(&buffer, 0..8).unwrap();
        let work = encoder.finish().unwrap();
        assert_eq!(submit_readback(&device, lane, work, ticket), expected);
    }
}
