//! Active-pass rectangle clear preserves pixels outside the rectangle and other attachments.
use super::*;
#[test]
fn active_pass_rect_clear_preserves_outside_and_other_attachment() {
    let d = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    assert!(d.capabilities().rect_attachment_clear);
    let lane = d
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let texture = || {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap()
    };
    let first = texture();
    let second = texture();
    let view = |texture: &Texture<_>| {
        d.create_view(
            texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap()
    };
    let first_view = view(&first);
    let second_view = view(&second);
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 4096,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let queries = d.queries(QueryKind::Occlusion).unwrap();
    let occlusion = queries
        .create(&QueryDesc {
            kind: QueryKind::Occlusion,
            count: 1,
        })
        .unwrap();
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let render = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let copy = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    macro_rules! transition {
        ($texture:expr, $from:expr, $to:expr, $layouts:expr $(,)?) => {
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: $texture,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: $from,
                    to: $to,
                },
                layout: Some($layouts),
                ownership: None,
            })
        };
    }
    let mut encoder = d.encoder(lane).unwrap();
    queries.reset(&mut encoder, &occlusion, 0..1).unwrap();
    encoder
        .transition_resources(&[
            transition!(
                &first,
                empty,
                render,
                (ImageLayout::Undefined, ImageLayout::ColorAttachment),
            ),
            transition!(
                &second,
                empty,
                render,
                (ImageLayout::Undefined, ImageLayout::ColorAttachment),
            ),
        ])
        .unwrap();
    let colors = [
        Some(ColorAttachment {
            view: AttachmentView::Texture(&first_view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([1.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        }),
        Some(ColorAttachment {
            view: AttachmentView::Texture(&second_view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 1.0, 1.0])),
            store: StoreOp::Store,
        }),
    ];
    let mut pass = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: Some(&occlusion),
            secondary_inheritance: None,
        })
        .unwrap();
    queries.begin_occlusion(&mut pass, &occlusion, 0).unwrap();
    pass.clear_rect(
        Rect {
            x: 2,
            y: 2,
            width: 4,
            height: 4,
        },
        &[ColorAttachmentClear {
            attachment: 0,
            value: ClearColor::Float([0.0, 1.0, 0.0, 1.0]),
        }],
        None,
        None,
    )
    .unwrap();
    queries.end_occlusion(&mut pass).unwrap();
    pass.finish().unwrap();
    encoder
        .transition_resources(&[
            transition!(
                &first,
                render,
                copy,
                (ImageLayout::ColorAttachment, ImageLayout::CopySrc),
            ),
            transition!(
                &second,
                render,
                copy,
                (ImageLayout::ColorAttachment, ImageLayout::CopySrc),
            ),
        ])
        .unwrap();
    for (texture, offset) in [(&first, 0), (&second, 2048)] {
        encoder
            .texture_to_buffer(
                TextureRegion {
                    texture,
                    mip: 0,
                    origin: [0, 0, 0],
                    extent: [8, 8, 1],
                    aspect: TextureAspects::COLOR,
                },
                &dst,
                TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: 256,
                    rows_per_image: 8,
                },
            )
            .unwrap();
    }
    let ticket = encoder.readback_buffer(&dst, 0..4096).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 755322,
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
                payload: StepPayload::Commands(vec![encoder.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block_on(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block_on(d.read_after(ticket, &point)).unwrap().bytes };
    let pixel = |base: usize, x: usize, y: usize| {
        let start = base + y * 256 + x * 4;
        &bytes[start..start + 4]
    };
    assert_eq!(pixel(0, 0, 0), &[255, 0, 0, 255]);
    assert_eq!(pixel(0, 3, 3), &[0, 255, 0, 255]);
    assert_eq!(pixel(0, 7, 7), &[255, 0, 0, 255]);
    assert_eq!(pixel(2048, 0, 0), &[0, 0, 255, 255]);
    assert_eq!(pixel(2048, 3, 3), &[0, 0, 255, 255]);
}
