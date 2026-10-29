//! BC mip uploads, buffer copies, and texture copies preserve encoded blocks.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block;

const BC_FORMATS: &[(TextureFormat, usize)] = &[
    (TextureFormat::Bc1RgbaUnorm, 8),
    (TextureFormat::Bc1RgbaUnormSrgb, 8),
    (TextureFormat::Bc2RgbaUnorm, 16),
    (TextureFormat::Bc2RgbaUnormSrgb, 16),
    (TextureFormat::Bc3RgbaUnorm, 16),
    (TextureFormat::Bc3RgbaUnormSrgb, 16),
    (TextureFormat::Bc4RUnorm, 8),
    (TextureFormat::Bc4RSnorm, 8),
    (TextureFormat::Bc5RgUnorm, 16),
    (TextureFormat::Bc5RgSnorm, 16),
    (TextureFormat::Bc6hRgbUfloat, 16),
    (TextureFormat::Bc6hRgbFloat, 16),
    (TextureFormat::Bc7RgbaUnorm, 16),
    (TextureFormat::Bc7RgbaUnormSrgb, 16),
];

#[test]
fn bc_small_mip_upload_copy_readback() {
    run_mip_upload_copy_readback(2, [2, 2, 1], 99910);
}

#[test]
fn bc_one_texel_mip_upload_copy_readback() {
    run_mip_upload_copy_readback(3, [1, 1, 1], 99930);
}

fn run_mip_upload_copy_readback(mip: u32, extent: [u32; 3], ordinal: u64) {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;

    for (index, &(format, block_size)) in BC_FORMATS.iter().enumerate() {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 2,
            format,
            mip_levels: 4,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        };
        if matches!(device.texture_support(&desc), Support::Unsupported(_)) {
            continue;
        }

        let texture = device.create_texture(&desc).unwrap();
        let first_readback = readback_buffer(&device);
        let data: Vec<_> = (0..block_size)
            .map(|byte| (byte as u8).wrapping_mul(7).wrapping_add(3))
            .collect();
        let mut encoder = device.encoder(lane).unwrap();

        encoder
            .write_texture(
                texture_region(&texture, mip, extent),
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: block_size as u32,
                    rows_per_image: 1,
                },
                &data,
            )
            .unwrap();
        transition_texture(
            &mut encoder,
            &texture,
            mip,
            TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
            ImageLayout::Common,
            ImageLayout::CopySrc,
        );
        encoder
            .texture_to_buffer(
                texture_region(&texture, mip, extent),
                &first_readback,
                readback_layout(),
            )
            .unwrap();
        let first_ticket = encoder.readback_buffer(&first_readback, 0..512).unwrap();

        let source = device
            .create_buffer(&BufferDesc {
                label: None,
                size: 512,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let mut staged = vec![0; 512];
        staged[..block_size].copy_from_slice(&data);
        encoder.write_buffer(&source, 0, &staged).unwrap();
        encoder
            .transition_resources(&[
                ResourceTransition::Buffer(BufferBarrier {
                    target: BufferSlice {
                        buffer: &source,
                        range: 0..512,
                    },
                    usage: StateTransition {
                        from: BufferUses {
                            stages: StageMask::EMPTY,
                            accesses: AccessMask::EMPTY,
                        },
                        to: BufferUses {
                            stages: StageMask::COPY,
                            accesses: AccessMask::COPY_READ,
                        },
                    },
                    ownership: None,
                }),
                texture_transition(
                    &texture,
                    mip,
                    TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                    TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_WRITE,
                    },
                    ImageLayout::CopySrc,
                    ImageLayout::CopyDst,
                ),
            ])
            .unwrap();
        encoder
            .buffer_to_texture(
                &source,
                readback_layout(),
                texture_region(&texture, mip, extent),
            )
            .unwrap();
        transition_texture(
            &mut encoder,
            &texture,
            mip,
            TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            },
            TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
            ImageLayout::CopyDst,
            ImageLayout::CopySrc,
        );
        let second_readback = readback_buffer(&device);
        encoder
            .texture_to_buffer(
                texture_region(&texture, mip, extent),
                &second_readback,
                readback_layout(),
            )
            .unwrap();
        let second_ticket = encoder.readback_buffer(&second_readback, 0..512).unwrap();

        let copied = device.create_texture(&desc).unwrap();
        transition_texture(
            &mut encoder,
            &copied,
            mip,
            TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            },
            ImageLayout::Common,
            ImageLayout::CopyDst,
        );
        encoder
            .copy_texture(
                texture_region(&texture, mip, extent),
                texture_region(&copied, mip, extent),
            )
            .unwrap();
        transition_texture(
            &mut encoder,
            &copied,
            mip,
            TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            },
            TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
            ImageLayout::CopyDst,
            ImageLayout::CopySrc,
        );
        let third_readback = readback_buffer(&device);
        encoder
            .texture_to_buffer(
                texture_region(&copied, mip, extent),
                &third_readback,
                readback_layout(),
            )
            .unwrap();
        let third_ticket = encoder.readback_buffer(&third_readback, 0..512).unwrap();

        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: device.id(),
                    ordinal: ordinal + index as u64,
                },
                Arc::from([StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                }]),
            )
        };
        let step = plan.step(0);
        let outcome = unsafe {
            device.queue().submit(SubmitPacket {
                plan,
                steps: vec![BoundStep {
                    step,
                    payload: StepPayload::Commands(vec![encoder.finish().unwrap()]),
                }],
                completion: CompletionRequest::End,
            })
        };
        let point = match outcome {
            SubmitOutcome::Accepted(receipt) => {
                receipt.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("submit {format:?}"),
        };
        block(device.wait_completion(&point)).unwrap();
        assert_readback(
            &device,
            first_ticket,
            &point,
            &data,
            block_size,
            format,
            "upload",
        );
        assert_readback(
            &device,
            second_ticket,
            &point,
            &data,
            block_size,
            format,
            "buffer copy",
        );
        assert_readback(
            &device,
            third_ticket,
            &point,
            &data,
            block_size,
            format,
            "texture copy",
        );
    }
}

fn texture_region<'a>(
    texture: &'a Texture<fluxel_rhi_dx12::Dx12>,
    mip: u32,
    extent: [u32; 3],
) -> TextureRegion<'a, fluxel_rhi_dx12::Dx12> {
    TextureRegion {
        texture,
        mip,
        origin: [0, 0, 1],
        extent,
        aspect: TextureAspects::COLOR,
    }
}

fn readback_buffer(device: &Device<fluxel_rhi_dx12::Dx12>) -> Buffer<fluxel_rhi_dx12::Dx12> {
    device
        .create_buffer(&BufferDesc {
            label: None,
            size: 512,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap()
}

fn readback_layout() -> TexelCopyBufferLayout {
    TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: 256,
        rows_per_image: 1,
    }
}

fn texture_transition<'a>(
    texture: &'a Texture<fluxel_rhi_dx12::Dx12>,
    mip: u32,
    from: TextureUses,
    to: TextureUses,
    from_layout: ImageLayout,
    to_layout: ImageLayout,
) -> ResourceTransition<'a, fluxel_rhi_dx12::Dx12> {
    ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture,
            aspects: TextureAspects::COLOR,
            mips: mip..mip + 1,
            layers: 1..2,
        },
        usage: StateTransition { from, to },
        layout: Some((from_layout, to_layout)),
        ownership: None,
    })
}

fn transition_texture(
    encoder: &mut CommandEncoder<fluxel_rhi_dx12::Dx12>,
    texture: &Texture<fluxel_rhi_dx12::Dx12>,
    mip: u32,
    from: TextureUses,
    to: TextureUses,
    from_layout: ImageLayout,
    to_layout: ImageLayout,
) {
    encoder
        .transition_resources(&[texture_transition(
            texture,
            mip,
            from,
            to,
            from_layout,
            to_layout,
        )])
        .unwrap();
}

fn assert_readback(
    device: &Device<fluxel_rhi_dx12::Dx12>,
    ticket: ReadbackTicket<fluxel_rhi_dx12::Dx12>,
    point: &CompletionPoint<fluxel_rhi_dx12::Dx12>,
    expected: &[u8],
    block_size: usize,
    format: TextureFormat,
    operation: &str,
) {
    // SAFETY: the submitted command encoder recorded this ticket before `point`.
    let bytes = unsafe { block(device.read_after(ticket, point)).unwrap().bytes };
    assert_eq!(&bytes[..block_size], expected, "{operation} {format:?}");
}
