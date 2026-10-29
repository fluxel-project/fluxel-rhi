//! Exact integer render-target clears across D1, D2, and D3 views.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block;

#[derive(Clone, Copy)]
struct DimensionCase {
    dimension: TextureDimension,
    extent: [u32; 3],
    view_dimension: TextureViewDimension,
    readback_size: u64,
    rows_per_image: u32,
    readback_rows: u32,
}

#[test]
fn integer_render_target_clear_readback() {
    run(DimensionCase {
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        view_dimension: TextureViewDimension::D2,
        readback_size: 1024,
        rows_per_image: 4,
        readback_rows: 4,
    });
}

#[test]
fn integer_d1_render_target_clear_readback() {
    run(DimensionCase {
        dimension: TextureDimension::D1,
        extent: [4, 1, 1],
        view_dimension: TextureViewDimension::D1,
        readback_size: 1024,
        rows_per_image: 1,
        readback_rows: 1,
    });
}

#[test]
fn integer_d3_render_target_clear_readback() {
    run(DimensionCase {
        dimension: TextureDimension::D3,
        extent: [4, 4, 2],
        view_dimension: TextureViewDimension::D3,
        readback_size: 2048,
        rows_per_image: 4,
        readback_rows: 8,
    });
}

fn run(case: DimensionCase) {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let unsigned = [0xffff_fffeu32, 0x8000_0001, 0x0100_0001, 0xdead_beef];
    let signed = [i32::MIN + 1, i32::MAX, -16_777_217, 0x1234_5678];
    let formats = [
        (
            TextureFormat::Rgba32Uint,
            ClearColor::Uint(unsigned),
            unsigned
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        ),
        (
            TextureFormat::Rgba32Sint,
            ClearColor::Sint(signed),
            signed
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        ),
    ];

    for (format, value, expected) in formats {
        for pass_clear in [false, true] {
            let texture = device
                .create_texture(&TextureDesc {
                    label: None,
                    dimension: case.dimension,
                    extent: case.extent,
                    array_layers: 1,
                    format,
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
                    view_formats: vec![],
                })
                .unwrap();
            let readback = device
                .create_buffer(&BufferDesc {
                    label: None,
                    size: case.readback_size,
                    usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                    memory: MemoryClass::Readback,
                })
                .unwrap();
            let mut encoder = device.encoder(lane).unwrap();
            let transition = |from, to, before, after| {
                ResourceTransition::Texture(TextureBarrier {
                    target: Subresources {
                        texture: &texture,
                        aspects: TextureAspects::COLOR,
                        mips: 0..1,
                        layers: 0..1,
                    },
                    usage: StateTransition { from, to },
                    layout: Some((before, after)),
                    ownership: None,
                })
            };
            let empty = TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            };
            let attachment_write = TextureUses {
                stages: StageMask::COLOR_OUTPUT,
                accesses: AccessMask::COLOR_WRITE,
            };
            let copy_read = TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            };
            encoder
                .transition_resources(&[transition(
                    empty,
                    attachment_write,
                    ImageLayout::Common,
                    ImageLayout::ColorAttachment,
                )])
                .unwrap();
            if pass_clear {
                let view = device
                    .create_view(
                        &texture,
                        &TextureViewDesc {
                            format,
                            dimension: case.view_dimension,
                            aspects: TextureAspects::COLOR,
                            mips: 0..1,
                            layers: 0..1,
                        },
                    )
                    .unwrap();
                let colors = [Some(ColorAttachment {
                    view: AttachmentView::Texture(&view),
                    resolve_target: None,
                    load: LoadOp::Clear(value),
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
                            width: case.extent[0],
                            height: case.extent[1],
                        },
                        multiview_mask: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        secondary_inheritance: None,
                    })
                    .unwrap()
                    .finish()
                    .unwrap();
            } else {
                encoder
                    .clear_texture(&texture, 0..1, 0..1, TextureClearValue::Color(value))
                    .unwrap();
            }
            encoder
                .transition_resources(&[transition(
                    attachment_write,
                    copy_read,
                    ImageLayout::ColorAttachment,
                    ImageLayout::CopySrc,
                )])
                .unwrap();
            encoder
                .texture_to_buffer(
                    TextureRegion {
                        texture: &texture,
                        mip: 0,
                        origin: [0, 0, 0],
                        extent: case.extent,
                        aspect: TextureAspects::COLOR,
                    },
                    &readback,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: 256,
                        rows_per_image: case.rows_per_image,
                    },
                )
                .unwrap();
            let ticket = encoder
                .readback_buffer(&readback, 0..case.readback_size)
                .unwrap();
            let plan = unsafe {
                SubmissionPlan::from_compiled(
                    PlanId {
                        device: device.id(),
                        ordinal: 99802,
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
                _ => panic!("submission rejected"),
            };
            block(device.wait_completion(&point)).unwrap();
            // SAFETY: this completion point covers the one recorded ticket.
            let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
            for y in 0..case.readback_rows as usize {
                for x in 0..4 {
                    assert_eq!(
                        &bytes[y * 256 + x * expected.len()..y * 256 + (x + 1) * expected.len()],
                        &expected,
                        "{format:?}"
                    );
                }
            }
        }
    }
}
