//! Real GPU readback for multiple raster color targets.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};

fn block_on<F: Future>(f: F) -> F::Output {
    let w = std::task::Waker::noop();
    let mut c = std::task::Context::from_waker(w);
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut c) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

#[test]
fn sparse_multiview_resolve_preserves_masked_layer() {
    resolve_targets(true, true);
}

fn resolve_targets(array: bool, multiview: bool) {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let texture = |samples, usage| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: if array { 2 } else { 1 },
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples,
            usage,
            view_formats: vec![],
        })
        .unwrap()
    };
    let sources = [
        texture(4, TextureUsage::RENDER_ATTACHMENT),
        texture(4, TextureUsage::RENDER_ATTACHMENT),
    ];
    let destinations = [
        texture(
            1,
            TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
        ),
        texture(
            1,
            TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
        ),
    ];
    let view = |texture: &Texture<_>| {
        d.create_view(
            texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: if array {
                    TextureViewDimension::D2Array
                } else {
                    TextureViewDimension::D2
                },
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: if multiview {
                    0..2
                } else if array {
                    1..2
                } else {
                    0..1
                },
            },
        )
        .unwrap()
    };
    let source_views = [view(&sources[0]), view(&sources[1])];
    let destination_views = [view(&destinations[0]), view(&destinations[1])];
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: if array { 16384 } else { 8192 },
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let write = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let copy = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let barrier = |texture, from, to, layouts| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: if multiview {
                    0..2
                } else if array {
                    1..2
                } else {
                    0..1
                },
            },
            usage: StateTransition { from, to },
            layout: Some(layouts),
            ownership: None,
        })
    };
    for source in &sources {
        e.transition_resources(&[barrier(
            source,
            empty,
            write,
            (ImageLayout::Common, ImageLayout::ColorAttachment),
        )])
        .unwrap();
    }
    for destination in &destinations {
        e.transition_resources(&[barrier(
            destination,
            empty,
            write,
            (ImageLayout::Common, ImageLayout::ColorAttachment),
        )])
        .unwrap();
        e.clear_texture(
            destination,
            0..1,
            0..2,
            TextureClearValue::Color(ClearColor::Float([0., 0., 1., 1.])),
        )
        .unwrap();
        e.transition_resources(&[barrier(
            destination,
            write,
            write,
            (ImageLayout::ColorAttachment, ImageLayout::ResolveDst),
        )])
        .unwrap();
    }
    let attachments = [
        Some(ColorAttachment {
            view: AttachmentView::Texture(&source_views[0]),
            resolve_target: Some(AttachmentView::Texture(&destination_views[0])),
            load: LoadOp::Clear(ClearColor::Float([1.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        }),
        Some(ColorAttachment {
            view: AttachmentView::Texture(&source_views[1]),
            resolve_target: Some(AttachmentView::Texture(&destination_views[1])),
            load: LoadOp::Clear(ClearColor::Float([0.0, 1.0, 0.0, 1.0])),
            store: StoreOp::Store,
        }),
    ];
    e.begin_raster(&RasterPassDesc {
        label: None,
        colors: &attachments,
        depth_stencil: None,
        render_area: Rect {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
        },
        multiview_mask: if multiview { Some(2) } else { None },
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    })
    .unwrap()
    .finish()
    .unwrap();
    for (i, destination) in destinations.iter().enumerate() {
        e.transition_resources(&[barrier(
            destination,
            write,
            copy,
            (ImageLayout::ResolveDst, ImageLayout::CopySrc),
        )])
        .unwrap();
        e.texture_to_buffer(
            TextureRegion {
                texture: destination,
                mip: 0,
                origin: [0, 0, if array { 1 } else { 0 }],
                extent: [16, 16, 1],
                aspect: TextureAspects::COLOR,
            },
            &dst,
            TexelCopyBufferLayout {
                offset: i as u64 * 4096,
                bytes_per_row: 256,
                rows_per_image: 16,
            },
        )
        .unwrap();
    }
    if array {
        for (i, destination) in destinations.iter().enumerate() {
            e.texture_to_buffer(
                TextureRegion {
                    texture: destination,
                    mip: 0,
                    origin: [0, 0, 0],
                    extent: [16, 16, 1],
                    aspect: TextureAspects::COLOR,
                },
                &dst,
                TexelCopyBufferLayout {
                    offset: 8192 + i as u64 * 4096,
                    bytes_per_row: 256,
                    rows_per_image: 16,
                },
            )
            .unwrap();
        }
    }
    let ticket = e
        .readback_buffer(&dst, 0..if array { 16384 } else { 8192 })
        .unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755322,
    };
    // SAFETY: one command step on this device's graphics lane.
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
    // SAFETY: this recorded work is submitted exactly once on its lane.
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
    block_on(d.wait_completion(&point)).unwrap();
    // SAFETY: the completion point covers this readback ticket.
    let bytes = unsafe { block_on(d.read_after(ticket, &point)).unwrap().bytes };
    for y in 0..16 {
        for x in 0..16 {
            let offset = y * 256 + x * 4;
            assert_eq!(&bytes[offset..offset + 4], &[255, 0, 0, 255]);
            assert_eq!(&bytes[4096 + offset..4096 + offset + 4], &[0, 255, 0, 255]);
            if array {
                let expected0 = [0, 0, 255, 255];
                let expected1 = [0, 0, 255, 255];
                assert_eq!(&bytes[8192 + offset..8192 + offset + 4], &expected0);
                assert_eq!(&bytes[12288 + offset..12288 + offset + 4], &expected1);
            }
        }
    }
}
