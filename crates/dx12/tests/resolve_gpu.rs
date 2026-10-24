//! GPU readback coverage for standalone partial multisample resolve.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};

fn block<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

fn barrier<'a>(
    texture: &'a Texture<fluxel_rhi_dx12::Dx12>,
    from: TextureUses,
    to: TextureUses,
    before: ImageLayout,
    after: ImageLayout,
) -> ResourceTransition<'a, fluxel_rhi_dx12::Dx12> {
    ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..2,
        },
        usage: StateTransition { from, to },
        layout: Some((before, after)),
        ownership: None,
    })
}

#[test]
fn partial_array_resolve_preserves_offsets_and_layers() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let source = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 4,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let target = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 4096,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let color_write = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let color_read = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_READ,
    };
    let copy_read = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let mut encoder = device.encoder(lane).unwrap();
    // The destination deliberately has no render-target use.  Seed every
    // texel before the resolve so the readback can prove the native region
    // command left the complement unchanged.
    let sentinel = vec![0x5a; 8 * 8 * 4];
    for layer in 0..2 {
        encoder
            .write_texture(
                TextureRegion {
                    texture: &target,
                    mip: 0,
                    origin: [0, 0, layer],
                    extent: [8, 8, 1],
                    aspect: TextureAspects::COLOR,
                },
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: 32,
                    rows_per_image: 8,
                },
                &sentinel,
            )
            .unwrap();
    }
    encoder
        .transition_resources(&[barrier(
            &source,
            empty,
            color_write,
            ImageLayout::Common,
            ImageLayout::ColorAttachment,
        )])
        .unwrap();
    encoder
        .clear_texture(
            &source,
            0..1,
            0..2,
            TextureClearValue::Color(ClearColor::Float([1.0, 0.0, 0.0, 1.0])),
        )
        .unwrap();
    encoder
        .transition_resources(&[
            barrier(
                &source,
                color_write,
                color_read,
                ImageLayout::ColorAttachment,
                ImageLayout::ResolveSrc,
            ),
            barrier(
                &target,
                empty,
                color_write,
                ImageLayout::Common,
                ImageLayout::ResolveDst,
            ),
        ])
        .unwrap();
    encoder
        .resolve(
            TextureRegion {
                texture: &source,
                mip: 0,
                origin: [2, 1, 0],
                extent: [4, 3, 2],
                aspect: TextureAspects::COLOR,
            },
            TextureRegion {
                texture: &target,
                mip: 0,
                origin: [1, 2, 0],
                extent: [4, 3, 2],
                aspect: TextureAspects::COLOR,
            },
        )
        .unwrap();
    encoder
        .transition_resources(&[barrier(
            &target,
            color_write,
            copy_read,
            ImageLayout::ResolveDst,
            ImageLayout::CopySrc,
        )])
        .unwrap();
    for layer in 0..2 {
        encoder
            .texture_to_buffer(
                TextureRegion {
                    texture: &target,
                    mip: 0,
                    origin: [0, 0, layer],
                    extent: [8, 8, 1],
                    aspect: TextureAspects::COLOR,
                },
                &readback,
                TexelCopyBufferLayout {
                    offset: u64::from(layer) * 2048,
                    bytes_per_row: 256,
                    rows_per_image: 8,
                },
            )
            .unwrap();
    }
    let ticket = encoder.readback_buffer(&readback, 0..4096).unwrap();
    let work = encoder.finish().unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 6_120_022,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let outcome = unsafe {
        device.queue().submit(SubmitPacket {
            steps: vec![BoundStep {
                step: plan.step(0),
                payload: StepPayload::Commands(vec![work]),
            }],
            plan,
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("DX12 submission rejected"),
    };
    block(device.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    for layer in 0..2_usize {
        for row in 0..8_usize {
            for pixel in 0..8_usize {
                let start = layer * 2048 + row * 256 + pixel * 4;
                let resolved = (1..5).contains(&pixel) && (2..5).contains(&row);
                assert_eq!(
                    &bytes[start..start + 4],
                    if resolved {
                        &[255, 0, 0, 255]
                    } else {
                        &[0x5a; 4]
                    }
                );
            }
        }
    }
}
