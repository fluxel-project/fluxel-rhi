//! Transient multisample attachment resolve and native allocation fact.
use super::*;

fn barrier<'a>(
    texture: &'a Texture<fluxel_rhi_vulkan::Vulkan>,
    from: TextureUses,
    to: TextureUses,
    layouts: (ImageLayout, ImageLayout),
) -> ResourceTransition<'a, fluxel_rhi_vulkan::Vulkan> {
    ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition { from, to },
        layout: Some(layouts),
        ownership: None,
    })
}

#[test]
fn transient_msaa_attachment_resolves_and_discards() {
    let instance = VulkanInstance::new().unwrap();
    let native_instance = instance.instance().unwrap();
    let adapter = native_instance
        .enumerate()
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    eprintln!("Vulkan transient attachment adapter: {}", adapter.name);
    let device = block_on(instance.open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let source = device
        .create_texture(&TextureDesc {
            label: Some("transient MSAA source".into()),
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 4,
            usage: TextureUsage::TRANSIENT_ATTACHMENT | TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let target = device
        .create_texture(&TextureDesc {
            label: Some("persistent resolve target".into()),
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    eprintln!(
        "Vulkan transient attachment native lazy allocation: {}",
        source.uses_native_transient_attachment()
    );
    let view = |texture: &Texture<_>| {
        device
            .create_view(
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
    let source_view = view(&source);
    let target_view = view(&target);
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 1024,
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
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .transition_resources(&[
            barrier(
                &source,
                empty,
                color_write,
                (ImageLayout::Undefined, ImageLayout::ColorAttachment),
            ),
            barrier(
                &target,
                empty,
                color_write,
                (ImageLayout::Undefined, ImageLayout::ResolveDst),
            ),
        ])
        .unwrap();
    let attachments = [Some(ColorAttachment {
        view: AttachmentView::Texture(&source_view),
        resolve_target: Some(AttachmentView::Texture(&target_view)),
        load: LoadOp::Clear(ClearColor::Float([1.0, 0.0, 0.0, 1.0])),
        store: StoreOp::Discard,
    })];
    encoder
        .begin_raster(&RasterPassDesc {
            label: Some("transient clear and resolve"),
            colors: &attachments,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap()
        .finish()
        .unwrap();
    encoder
        .transition_resources(&[barrier(
            &target,
            color_write,
            TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
            (ImageLayout::ResolveDst, ImageLayout::CopySrc),
        )])
        .unwrap();
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &target,
                mip: 0,
                origin: [0, 0, 0],
                extent: [16, 16, 1],
                aspect: TextureAspects::COLOR,
            },
            &readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 64,
                rows_per_image: 16,
            },
        )
        .unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..1024).unwrap();
    let work = encoder.finish().unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 755_323,
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
            plan: plan.clone(),
            steps: vec![BoundStep {
                step: plan.step(0),
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("transient attachment submission was rejected"),
    };
    block_on(device.wait_completion(&point)).unwrap();
    let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };
    for texel in bytes.chunks_exact(4) {
        assert_eq!(texel, [255, 0, 0, 255]);
    }
}
