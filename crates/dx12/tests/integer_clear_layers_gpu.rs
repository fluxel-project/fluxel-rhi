//! Exact integer render-target clear and readback.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};
fn block<F: Future>(f: F) -> F::Output {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}
#[test]
fn integer_array_layer_and_partial_clear_readback() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|v| v.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let unsigned = [0xffff_fffeu32, 0x8000_0001, 0x0100_0001, 0xdead_beef];
    let signed = [i32::MIN + 1, i32::MAX, -16_777_217, 0x1234_5678];
    let cases = [
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
    for (format, value, expected) in cases {
        for pass_clear in [false, true] {
            let tex = d
                .create_texture(&TextureDesc {
                    label: None,
                    dimension: TextureDimension::D2,
                    extent: [4, 4, 1],
                    array_layers: 2,
                    format,
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
                    view_formats: vec![],
                })
                .unwrap();
            let dst = d
                .create_buffer(&BufferDesc {
                    label: None,
                    size: 1024,
                    usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                    memory: MemoryClass::Readback,
                })
                .unwrap();
            let mut e = d.encoder(lane).unwrap();
            let barrier = |from, to, before, after| {
                ResourceTransition::Texture(TextureBarrier {
                    target: Subresources {
                        texture: &tex,
                        aspects: TextureAspects::COLOR,
                        mips: 0..1,
                        layers: 1..2,
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
            let rt = TextureUses {
                stages: StageMask::COLOR_OUTPUT,
                accesses: AccessMask::COLOR_WRITE,
            };
            let copy = TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            };
            e.transition_resources(&[barrier(
                empty,
                rt,
                ImageLayout::Common,
                ImageLayout::ColorAttachment,
            )])
            .unwrap();
            if pass_clear {
                let zero = match value {
                    ClearColor::Uint(_) => ClearColor::Uint([0; 4]),
                    ClearColor::Sint(_) => ClearColor::Sint([0; 4]),
                    _ => unreachable!(),
                };
                e.clear_texture(&tex, 0..1, 1..2, TextureClearValue::Color(zero))
                    .unwrap();
                let view = d
                    .create_view(
                        &tex,
                        &TextureViewDesc {
                            format,
                            dimension: TextureViewDimension::D2Array,
                            aspects: TextureAspects::COLOR,
                            mips: 0..1,
                            layers: 1..2,
                        },
                    )
                    .unwrap();
                let colors = [Some(ColorAttachment {
                    view: AttachmentView::Texture(&view),
                    resolve_target: None,
                    load: LoadOp::Clear(value),
                    store: StoreOp::Store,
                })];
                e.begin_raster(&RasterPassDesc {
                    label: None,
                    colors: &colors,
                    depth_stencil: None,
                    render_area: Rect {
                        x: 1,
                        y: 1,
                        width: 2,
                        height: 2,
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
                e.clear_texture(&tex, 0..1, 1..2, TextureClearValue::Color(value))
                    .unwrap();
            }
            e.transition_resources(&[barrier(
                rt,
                copy,
                ImageLayout::ColorAttachment,
                ImageLayout::CopySrc,
            )])
            .unwrap();
            e.texture_to_buffer(
                TextureRegion {
                    texture: &tex,
                    mip: 0,
                    origin: [0, 0, 1],
                    extent: [4, 4, 1],
                    aspect: TextureAspects::COLOR,
                },
                &dst,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: 256,
                    rows_per_image: 4,
                },
            )
            .unwrap();
            let ticket = e.readback_buffer(&dst, 0..1024).unwrap();
            let plan = unsafe {
                SubmissionPlan::from_compiled(
                    PlanId {
                        device: d.id(),
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
                d.queue().submit(SubmitPacket {
                    plan,
                    steps: vec![BoundStep {
                        step,
                        payload: StepPayload::Commands(vec![e.finish().unwrap()]),
                    }],
                    completion: CompletionRequest::End,
                })
            };
            let point = match outcome {
                SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
                _ => panic!("submission rejected"),
            };
            block(d.wait_completion(&point)).unwrap();
            let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
            for y in 0..4 {
                for x in 0..4 {
                    assert_eq!(
                        &bytes[y * 256 + x * expected.len()..y * 256 + (x + 1) * expected.len()],
                        if pass_clear && !(1..3).contains(&x) || pass_clear && !(1..3).contains(&y)
                        {
                            &[0u8; 16][..]
                        } else {
                            &expected
                        },
                        "{format:?} at {x},{y}"
                    );
                }
            }
        }
    }
}
