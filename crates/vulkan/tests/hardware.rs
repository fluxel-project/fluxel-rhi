//! Opt-in Windows Vulkan hardware regressions for submission and transfer paths.
//!
//! These tests use Core unsafe constructors only after building work whose
//! resource and lane identities all originate from the same tested device.
//! They intentionally exercise a real driver when explicitly enabled; normal
//! crate tests do not gain a hardware requirement or claim visual conformance.
#![cfg(all(windows, feature = "hardware-tests"))]

use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

#[path = "hardware/texture_extended.rs"]
mod texture_extended;
#[path = "hardware/texture_planes.rs"]
mod texture_planes;

use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::{Vulkan, VulkanInstance};

struct ThreadWake(std::thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
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
fn device() -> Device<Vulkan> {
    block_on(VulkanInstance::new().unwrap().open_default()).unwrap()
}
fn texture(device: &Device<Vulkan>) -> Texture<Vulkan> {
    device
        .create_texture(&TextureDesc {
            label: Some("Vulkan copy hardware test".into()),
            dimension: TextureDimension::D2,
            extent: [2, 2, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST),
            view_formats: vec![],
        })
        .unwrap()
}
fn region(texture: &Texture<Vulkan>) -> TextureRegion<'_, Vulkan> {
    TextureRegion {
        texture,
        mip: 0,
        origin: [0, 0, 0],
        extent: [2, 2, 1],
        aspect: TextureAspects::COLOR,
    }
}
fn region4(texture: &Texture<Vulkan>) -> TextureRegion<'_, Vulkan> {
    TextureRegion {
        texture,
        mip: 0,
        origin: [0, 0, 0],
        extent: [4, 4, 1],
        aspect: TextureAspects::COLOR,
    }
}
fn layout() -> TexelCopyBufferLayout {
    TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: 8,
        rows_per_image: 2,
    }
}
fn readback(device: &Device<Vulkan>) -> Buffer<Vulkan> {
    device
        .create_buffer(&BufferDesc {
            label: Some("Vulkan copy readback".into()),
            size: 16,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap()
}
fn texture_transition<'a>(
    texture: &'a Texture<Vulkan>,
    from: ImageLayout,
    to: ImageLayout,
    from_uses: TextureUses,
    to_uses: TextureUses,
) -> ResourceTransition<'a, Vulkan> {
    ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition {
            from: from_uses,
            to: to_uses,
        },
        layout: Some((from, to)),
        ownership: None,
    })
}
fn copy_write() -> TextureUses {
    TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    }
}
fn copy_read() -> TextureUses {
    TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    }
}
fn no_access() -> TextureUses {
    TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    }
}
fn submit(
    device: &Device<Vulkan>,
    work: Vec<RecordedWork<Vulkan>>,
    steps: bool,
) -> SubmissionReceipt<Vulkan> {
    let lane = device.lanes()[0].id;
    let id = PlanId {
        device: device.id(),
        ordinal: 1,
    };
    let descriptions = (0..work.len())
        .map(|_| StepDesc {
            lane,
            kind: StepKind::Commands,
            waits: vec![],
        })
        .collect::<Vec<_>>();
    // SAFETY: each description targets this device's sole lane and later
    // payload construction binds exactly one Commands payload to every step.
    let plan = unsafe { SubmissionPlan::from_compiled(id, Arc::from(descriptions)) };
    let completion = if steps {
        CompletionRequest::Steps(Arc::from(
            (0..work.len())
                .map(|index| plan.step(index))
                .collect::<Vec<_>>(),
        ))
    } else {
        CompletionRequest::End
    };
    let steps = work
        .into_iter()
        .enumerate()
        .map(|(index, work)| BoundStep {
            step: plan.step(index),
            payload: StepPayload::Commands(vec![work]),
        })
        .collect();
    // SAFETY: `plan` and every bound step use this device identity, matching
    // lane and the exact compiled step index constructed immediately above.
    match unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps,
            completion,
        })
    } {
        SubmitOutcome::Accepted(receipt) => receipt,
        SubmitOutcome::Rejected { error, .. }
        | SubmitOutcome::PartiallyAccepted { error, .. }
        | SubmitOutcome::Indeterminate { error, .. } => panic!("Vulkan submit failed: {error:?}"),
    }
}

#[test]
fn abandoned_recording_and_reverse_submission_keep_upload_correct() {
    let device = device();
    let lane = device.lanes()[0].id;
    let image = texture(&device);
    let output = readback(&device);
    let red = [255, 0, 0, 255].repeat(4);
    let blue = [0, 0, 255, 255].repeat(4);

    let mut abandoned = device.encoder(lane).unwrap();
    abandoned
        .write_texture(region(&image), layout(), &blue)
        .unwrap();
    drop(abandoned);

    let mut recorded_first = device.encoder(lane).unwrap();
    recorded_first
        .write_texture(region(&image), layout(), &red)
        .unwrap();
    let first = recorded_first.finish().unwrap();
    let mut recorded_second = device.encoder(lane).unwrap();
    recorded_second
        .write_texture(region(&image), layout(), &blue)
        .unwrap();
    let second = recorded_second.finish().unwrap();
    let mut read = device.encoder(lane).unwrap();
    read.transition_resources(&[texture_transition(
        &image,
        ImageLayout::Common,
        ImageLayout::CopySrc,
        copy_write(),
        copy_read(),
    )])
    .unwrap();
    read.texture_to_buffer(region(&image), &output, layout())
        .unwrap();
    let ticket = read.readback_buffer(&output, 0..16).unwrap();
    let read = read.finish().unwrap();

    let receipt = submit(&device, vec![second, first, read], true);
    assert_eq!(receipt.checkpoints.len(), 3);
    drop(device.wait_completion(&receipt.checkpoints[0].point));
    let final_point = &receipt.checkpoints[2].point;
    block_on(device.wait_completion(final_point)).unwrap();
    // SAFETY: the final checkpoint is awaited before mapping this ticket, and
    // both ticket and completion point were produced by this device.
    let data = block_on(unsafe { device.read_after(ticket, final_point) }).unwrap();
    assert_eq!(data.bytes, red);
}

#[test]
fn explicit_buffer_and_texture_copies_round_trip() {
    let device = device();
    let lane = device.lanes()[0].id;
    let source = device
        .create_buffer(&BufferDesc {
            label: Some("Vulkan copy source".into()),
            size: 16,
            usage: BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let intermediate = device
        .create_buffer(&BufferDesc {
            label: Some("Vulkan copy intermediate".into()),
            size: 16,
            usage: BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let first = texture(&device);
    let second = texture(&device);
    let output = readback(&device);
    let expected = [7, 11, 13, 255].repeat(4);
    let mut encoder = device.encoder(lane).unwrap();
    encoder.write_buffer(&source, 0, &expected).unwrap();
    encoder
        .transition_resources(&[
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &source,
                    range: 0..16,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_WRITE,
                    },
                    to: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                },
                ownership: None,
            }),
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &intermediate,
                    range: 0..16,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::ALL_COMMANDS,
                        accesses: AccessMask::EMPTY,
                    },
                    to: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_WRITE,
                    },
                },
                ownership: None,
            }),
            texture_transition(
                &first,
                ImageLayout::Undefined,
                ImageLayout::CopyDst,
                no_access(),
                copy_write(),
            ),
            texture_transition(
                &second,
                ImageLayout::Undefined,
                ImageLayout::CopyDst,
                no_access(),
                copy_write(),
            ),
        ])
        .unwrap();
    encoder
        .copy_buffer(&source, 0, &intermediate, 0, 16)
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &intermediate,
                range: 0..16,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                to: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            ownership: None,
        })])
        .unwrap();
    encoder
        .buffer_to_texture(&intermediate, layout(), region(&first))
        .unwrap();
    encoder
        .transition_resources(&[texture_transition(
            &first,
            ImageLayout::CopyDst,
            ImageLayout::CopySrc,
            copy_write(),
            copy_read(),
        )])
        .unwrap();
    encoder
        .copy_texture(region(&first), region(&second))
        .unwrap();
    encoder
        .transition_resources(&[texture_transition(
            &second,
            ImageLayout::CopyDst,
            ImageLayout::CopySrc,
            copy_write(),
            copy_read(),
        )])
        .unwrap();
    encoder
        .texture_to_buffer(region(&second), &output, layout())
        .unwrap();
    let ticket = encoder.readback_buffer(&output, 0..16).unwrap();
    let work = encoder.finish().unwrap();
    let receipt = submit(&device, vec![work], false);
    let point = &receipt.checkpoints[0].point;
    block_on(device.wait_completion(point)).unwrap();
    // SAFETY: the receipt's checkpoint was awaited and belongs to the same
    // device that created this readback ticket.
    let data = block_on(unsafe { device.read_after(ticket, point) }).unwrap();
    assert_eq!(data.bytes, expected);
}

#[test]
fn partial_buffer_clear_preserves_other_bytes() {
    let device = device();
    let buffer = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 32,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 32,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut expected: Vec<u8> = (0..32).collect();
    let mut encoder = device.encoder(device.lanes()[0].id).unwrap();
    encoder.write_buffer(&buffer, 0, &expected).unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..32,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                to: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
            },
            ownership: None,
        })])
        .unwrap();
    encoder.clear_buffer(&buffer, 8..24, 0).unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..32,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                to: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            ownership: None,
        })])
        .unwrap();
    encoder.copy_buffer(&buffer, 0, &output, 0, 32).unwrap();
    let ticket = encoder.readback_buffer(&output, 0..32).unwrap();
    let receipt = submit(&device, vec![encoder.finish().unwrap()], false);
    let point = &receipt.checkpoints[0].point;
    block_on(device.wait_completion(point)).unwrap();
    // SAFETY: the same device completed the submission owning the readback.
    let data = block_on(unsafe { device.read_after(ticket, point) }).unwrap();
    expected[8..24].fill(0);
    assert_eq!(data.bytes, expected);
}

#[test]
fn many_offscreen_submissions_without_host_checkpoints_complete() {
    let device = device();
    let lane = device.lanes()[0].id;
    for ordinal in 100..1100 {
        let work = device.encoder(lane).unwrap().finish().unwrap();
        let id = PlanId {
            device: device.id(),
            ordinal,
        };
        // SAFETY: this one-step plan names only the sole lane of `device`; the
        // following packet supplies its one Commands payload at step zero.
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                id,
                Arc::from([StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                }]),
            )
        };
        let steps = vec![BoundStep {
            step: plan.step(0),
            payload: StepPayload::Commands(vec![work]),
        }];
        // SAFETY: the work, plan and bound step share this device and lane;
        // CompletionRequest::None does not invent an unbound checkpoint.
        let outcome = unsafe {
            device.queue().submit(SubmitPacket {
                plan,
                steps,
                completion: CompletionRequest::None,
            })
        };
        assert!(matches!(outcome, SubmitOutcome::Accepted(_)));
    }
    block_on(device.wait_idle()).unwrap();
}
