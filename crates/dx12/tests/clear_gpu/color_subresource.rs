//! Color clear scopes exactly to the selected mip and array layer.

use super::*;

#[test]
fn clears_only_selected_color_mip_and_layer() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 4096,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
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
    let barrier = |from, to, layouts| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &texture,
                aspects: TextureAspects::COLOR,
                mips: 0..2,
                layers: 0..2,
            },
            usage: StateTransition { from, to },
            layout: Some(layouts),
            ownership: None,
        })
    };
    e.transition_resources(&[barrier(
        empty,
        render,
        (ImageLayout::Common, ImageLayout::ColorAttachment),
    )])
    .unwrap();
    e.clear_texture(
        &texture,
        0..2,
        0..2,
        TextureClearValue::Color(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
    )
    .unwrap();
    e.clear_texture(
        &texture,
        1..2,
        1..2,
        TextureClearValue::Color(ClearColor::Float([1.0, 0.0, 0.0, 1.0])),
    )
    .unwrap();
    e.transition_resources(&[barrier(
        render,
        copy,
        (ImageLayout::ColorAttachment, ImageLayout::CopySrc),
    )])
    .unwrap();
    for layer in 0..2 {
        for mip in 0..2 {
            let offset = u64::from(layer * 2 + mip) * 1024;
            e.texture_to_buffer(
                TextureRegion {
                    texture: &texture,
                    mip,
                    origin: [0, 0, layer],
                    extent: [4 >> mip, 4 >> mip, 1],
                    aspect: TextureAspects::COLOR,
                },
                &dst,
                TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: 256,
                    rows_per_image: 4 >> mip,
                },
            )
            .unwrap();
        }
    }
    let ticket = e.readback_buffer(&dst, 0..4096).unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755320,
    };
    // SAFETY: one command step uses this device's graphics lane.
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
    // SAFETY: this work is submitted exactly once on its recording lane.
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
    // SAFETY: this completion point covers the readback ticket.
    let bytes = unsafe { block_on(d.read_after(ticket, &point)).unwrap().bytes };
    for layer in 0..2usize {
        for mip in 0..2usize {
            let expected = if (layer, mip) == (1, 1) {
                [255, 0, 0, 255]
            } else {
                [0, 0, 0, 255]
            };
            let width = 4 >> mip;
            for y in 0..width {
                for x in 0..width {
                    let offset = (layer * 2 + mip) * 1024 + y * 256 + x * 4;
                    assert_eq!(&bytes[offset..offset + 4], &expected);
                }
            }
        }
    }
}
