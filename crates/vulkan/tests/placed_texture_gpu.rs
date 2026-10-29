//! GPU round trip through an explicitly placed image after dropping its heap.
#![cfg(all(windows, feature = "hardware-tests"))]
#[path = "support/mod.rs"]
mod support;
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use support::block_on;

#[test]
fn placed_texture_roundtrip_retains_heap() {
    let d = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    let memory = d.memory().expect("placed memory capability");
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let extent = [32, 32, 1];
    let mip = 0;
    let target_layer = 0;
    let pitch = extent[0] * 4;
    let size = u64::from(pitch) * u64::from(extent[1]);
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent,
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    let req = memory.texture_requirements(&desc).unwrap();
    assert!(!req.dedicated_only && req.native_type_bits != 0);
    let heap = memory
        .heap(&HeapDesc {
            size: req.size * 2,
            native_memory_class: req.native_type_bits.trailing_zeros(),
        })
        .unwrap();
    let texture = memory.place_texture(&heap, req.size, &desc).unwrap();
    drop(heap);
    let source_desc = BufferDesc {
        label: None,
        size,
        usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
        memory: MemoryClass::Upload,
    };
    let output_desc = BufferDesc {
        label: None,
        size,
        usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
        memory: MemoryClass::Readback,
    };
    let place = |desc: &BufferDesc| {
        let req = memory.buffer_requirements(desc).unwrap();
        assert!(!req.dedicated_only && req.native_type_bits != 0);
        let heap = memory
            .heap(&HeapDesc {
                size: req.size * 2,
                native_memory_class: req.native_type_bits.trailing_zeros(),
            })
            .unwrap();
        let buffer = memory.place_buffer(&heap, req.size, desc).unwrap();
        drop(heap);
        buffer
    };
    let source = place(&source_desc);
    let output = place(&output_desc);
    let alias_desc = BufferDesc {
        label: None,
        size,
        usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        memory: MemoryClass::Device,
    };
    let alias_req = memory.buffer_requirements(&alias_desc).unwrap();
    let alias_heap = memory
        .heap(&HeapDesc {
            size: alias_req.size,
            native_memory_class: alias_req.native_type_bits.trailing_zeros(),
        })
        .unwrap();
    let alias_a = memory.place_buffer(&alias_heap, 0, &alias_desc).unwrap();
    let alias_b = memory.place_buffer(&alias_heap, 0, &alias_desc).unwrap();
    drop(alias_heap);
    let bytes: Vec<u8> = (0..size).map(|i| ((i * 17 + 31) % 251) as u8).collect();
    // SAFETY: no GPU submission has used this upload buffer yet.
    let mut mapped = block_on(unsafe { d.map_buffer(&source, 0..size, MapMode::Write) }).unwrap();
    mapped.bytes_mut().unwrap().copy_from_slice(&bytes);
    mapped.flush(0..size).unwrap();
    drop(mapped);
    let mut e = d.encoder(lane).unwrap();
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let write = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    };
    let read = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let barrier = |from, to, layout| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &texture,
                aspects: TextureAspects::COLOR,
                mips: mip..mip + 1,
                layers: target_layer..target_layer + 1,
            },
            usage: StateTransition { from, to },
            layout: Some(layout),
            ownership: None,
        })
    };
    e.transition_resources(&[
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &source,
                range: 0..size,
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
        barrier(empty, write, (ImageLayout::Undefined, ImageLayout::CopyDst)),
    ])
    .unwrap();
    let layout = TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: pitch,
        rows_per_image: extent[1],
    };
    let region = || TextureRegion {
        texture: &texture,
        mip,
        origin: [0, 0, target_layer],
        extent,
        aspect: TextureAspects::COLOR,
    };
    e.copy_buffer(&source, 0, &alias_a, 0, size).unwrap();
    e.transition_resources(&[ResourceTransition::Aliasing {
        before: Some(AliasedResource::Buffer(&alias_a)),
        after: Some(AliasedResource::Buffer(&alias_b)),
    }])
    .unwrap();
    e.buffer_to_texture(&alias_b, layout, region()).unwrap();
    e.transition_resources(&[barrier(
        write,
        read,
        (ImageLayout::CopyDst, ImageLayout::CopySrc),
    )])
    .unwrap();
    e.texture_to_buffer(region(), &output, layout).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 889000,
            },
            vec![StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]
            .into(),
        )
    };
    let steps = vec![BoundStep {
        step: plan.step(0),
        payload: StepPayload::Commands(vec![e.finish().unwrap()]),
    }];
    let receipt = match unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps,
            completion: CompletionRequest::End,
        })
    } {
        SubmitOutcome::Accepted(r) => r,
        _ => panic!("placed texture submit failed"),
    };
    let point = &receipt.checkpoints[0].point;
    block_on(d.wait_completion(point)).unwrap();
    // SAFETY: the completed checkpoint covers the copy into output.
    let mut actual = block_on(unsafe { d.map_buffer(&output, 0..size, MapMode::Read) }).unwrap();
    actual.invalidate(0..size).unwrap();
    assert_eq!(actual.bytes(), bytes, "placed texture");
}
