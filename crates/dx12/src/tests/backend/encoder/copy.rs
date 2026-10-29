//! Real-GPU conformance checks for explicit copy commands.
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use fluxel_rhi_core::api::{
    AccessMask, BoundStep, BufferBarrier, BufferDesc, BufferSlice, BufferUsage, CompletionRequest,
    Device, Error, ImageLayout, LaneClass, LaneId, MemoryClass, PlanId, ResourceTransition,
    StageMask, StateTransition, StepDesc, StepKind, StepPayload, SubmissionPlan, SubmitOutcome,
    SubmitPacket, Subresources, TexelCopyBufferLayout, TextureAspects, TextureBarrier, TextureDesc,
    TextureDimension, TextureRegion, TextureUsage, TextureUses,
};

use crate::backend::{Dx12, Dx12Instance};

#[path = "../support.rs"]
mod support;
use support::block as block_on;

static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);

fn copy_uses(read: bool) -> fluxel_rhi_core::api::BufferUses {
    fluxel_rhi_core::api::BufferUses {
        stages: StageMask::COPY,
        accesses: if read {
            AccessMask::COPY_READ
        } else {
            AccessMask::COPY_WRITE
        },
    }
}

fn buffer_common() -> fluxel_rhi_core::api::BufferUses {
    fluxel_rhi_core::api::BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    }
}

fn texture_uses(read: bool) -> TextureUses {
    TextureUses {
        stages: StageMask::COPY,
        accesses: if read {
            AccessMask::COPY_READ
        } else {
            AccessMask::COPY_WRITE
        },
    }
}

fn texture_common() -> TextureUses {
    TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    }
}

fn submit(
    device: &Device<Dx12>,
    lane: LaneId,
    work: fluxel_rhi_core::api::RecordedWork<Dx12>,
) -> fluxel_rhi_core::api::CompletionPoint<Dx12> {
    let id = PlanId {
        device: device.id(),
        ordinal: NEXT_PLAN.fetch_add(1, Ordering::Relaxed),
    };
    // SAFETY: this helper creates a unique plan id and exactly one valid
    // commands step for the device/lane supplied by the fixture.
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
    let step = plan.step(0);
    // SAFETY: work was recorded by the same device/lane and is consumed once
    // by the plan constructed immediately above.
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
    match outcome {
        SubmitOutcome::Accepted(receipt) => {
            receipt
                .checkpoints
                .into_iter()
                .next()
                .expect("end checkpoint")
                .point
        }
        _ => panic!("copy submission failed"),
    }
}

fn device_and_lane() -> (Device<Dx12>, LaneId) {
    let device = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .expect("DX12 graphics lane")
        .id;
    (device, lane)
}

fn texture_desc(label: &str) -> TextureDesc {
    TextureDesc {
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [2, 2, 1],
        array_layers: 1,
        format: fluxel_rhi_core::api::TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST),
        view_formats: vec![],
    }
}

#[test]
fn generic_copies_round_trip_real_gpu_bytes() {
    let (device, lane) = device_and_lane();
    let source = device
        .create_buffer(&BufferDesc {
            label: Some("copy source".into()),
            size: 512,
            usage: BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let buffer_readback = device
        .create_buffer(&BufferDesc {
            label: Some("copy buffer readback".into()),
            size: 512,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let texture_a = device
        .create_texture(&texture_desc("copy texture a"))
        .unwrap();
    let texture_b = device
        .create_texture(&texture_desc("copy texture b"))
        .unwrap();
    let texture_readback = device
        .create_buffer(&BufferDesc {
            label: Some("copy texture readback".into()),
            size: 512,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut expected = [0u8; 512];
    expected[..8].copy_from_slice(&[3, 7, 11, 255, 13, 17, 19, 255]);
    expected[256..264].copy_from_slice(&[23, 29, 31, 255, 37, 41, 43, 255]);

    let mut encoder = device.encoder(lane).unwrap();
    encoder.write_buffer(&source, 0, &expected).unwrap();
    encoder
        .transition_resources(&[
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &source,
                    range: 0..512,
                },
                usage: StateTransition {
                    from: buffer_common(),
                    to: copy_uses(true),
                },
                ownership: None,
            }),
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &texture_a,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: texture_common(),
                    to: texture_uses(false),
                },
                layout: Some((ImageLayout::Common, ImageLayout::CopyDst)),
                ownership: None,
            }),
        ])
        .unwrap();
    encoder
        .copy_buffer(&source, 0, &buffer_readback, 0, 512)
        .unwrap();
    let layout = TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: 256,
        rows_per_image: 2,
    };
    encoder
        .buffer_to_texture(
            &source,
            layout,
            TextureRegion {
                texture: &texture_a,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 2, 1],
                aspect: TextureAspects::COLOR,
            },
        )
        .unwrap();
    encoder
        .transition_resources(&[
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &texture_a,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: texture_uses(false),
                    to: texture_uses(true),
                },
                layout: Some((ImageLayout::CopyDst, ImageLayout::CopySrc)),
                ownership: None,
            }),
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &texture_b,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: texture_common(),
                    to: texture_uses(false),
                },
                layout: Some((ImageLayout::Common, ImageLayout::CopyDst)),
                ownership: None,
            }),
        ])
        .unwrap();
    encoder
        .copy_texture(
            TextureRegion {
                texture: &texture_a,
                mip: 0,
                origin: [1, 0, 0],
                extent: [1, 2, 1],
                aspect: TextureAspects::COLOR,
            },
            TextureRegion {
                texture: &texture_b,
                mip: 0,
                origin: [0, 0, 0],
                extent: [1, 2, 1],
                aspect: TextureAspects::COLOR,
            },
        )
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &texture_b,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: texture_uses(false),
                to: texture_uses(true),
            },
            layout: Some((ImageLayout::CopyDst, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &texture_b,
                mip: 0,
                origin: [0, 0, 0],
                extent: [1, 2, 1],
                aspect: TextureAspects::COLOR,
            },
            &texture_readback,
            layout,
        )
        .unwrap();
    let buffer_ticket = encoder.readback_buffer(&buffer_readback, 0..512).unwrap();
    let texture_ticket = encoder.readback_buffer(&texture_readback, 0..512).unwrap();
    let point = submit(&device, lane, encoder.finish().unwrap());
    block_on(device.wait_completion(&point)).unwrap();
    assert_eq!(
        // SAFETY: completion was awaited above; ticket and completion point
        // were both produced by this device's preceding readback submission.
        unsafe { block_on(device.read_after(buffer_ticket, &point)).unwrap() }.bytes,
        expected
    );
    let mut expected_texture = [0u8; 512];
    expected_texture[..4].copy_from_slice(&expected[4..8]);
    expected_texture[256..260].copy_from_slice(&expected[260..264]);
    assert_eq!(
        // SAFETY: same completed-device/ticket invariant as the buffer readback.
        unsafe { block_on(device.read_after(texture_ticket, &point)).unwrap() }.bytes,
        expected_texture
    );
}

#[test]
fn generic_copy_rejects_missing_usage_and_invalid_range() {
    let (device, lane) = device_and_lane();
    let source = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let destination = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    assert!(matches!(
        encoder.copy_buffer(&source, 0, &destination, 0, 8),
        Err(Error::Unsupported(_))
    ));
    let source = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    assert!(matches!(
        encoder.copy_buffer(&source, 12, &destination, 0, 8),
        Err(Error::InvalidRange)
    ));
}
