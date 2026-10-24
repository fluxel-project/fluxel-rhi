//! Explicit DX12 heap placement keeps resources valid after heap wrappers drop.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};

fn block<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn submit(
    device: &Device<fluxel_rhi_dx12::Dx12>,
    lane: LaneId,
    work: RecordedWork<fluxel_rhi_dx12::Dx12>,
) -> CompletionPoint<fluxel_rhi_dx12::Dx12> {
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 73_040,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    match unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    } {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("placed-memory submission was not accepted"),
    }
}

#[test]
fn placed_upload_and_device_buffers_survive_heap_drop_and_roundtrip() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let memory = device.memory().expect("DX12 placed-memory capability");
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let upload_desc = BufferDesc {
        label: None,
        size: 64,
        usage: BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC,
        memory: MemoryClass::Upload,
    };
    let device_desc = BufferDesc {
        label: None,
        size: 64,
        usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        memory: MemoryClass::Device,
    };
    let upload_req = memory.buffer_requirements(&upload_desc).unwrap();
    let device_req = memory.buffer_requirements(&device_desc).unwrap();
    assert_ne!(upload_req.native_type_bits & (1 << 1), 0);
    assert_ne!(device_req.native_type_bits & ((1 << 0) | (1 << 3)), 0);
    let upload_heap = memory
        .heap(&HeapDesc {
            size: upload_req.size,
            native_memory_class: upload_req.native_type_bits.trailing_zeros(),
        })
        .unwrap();
    let device_heap = memory
        .heap(&HeapDesc {
            size: device_req.size,
            native_memory_class: if device_req.native_type_bits & (1 << 3) != 0 {
                3
            } else {
                device_req.native_type_bits.trailing_zeros()
            },
        })
        .unwrap();
    assert!(memory.place_buffer(&upload_heap, 0, &device_desc).is_err());
    assert!(memory.place_buffer(&device_heap, 1, &device_desc).is_err());
    let upload = memory.place_buffer(&upload_heap, 0, &upload_desc).unwrap();
    let alias_a = memory.place_buffer(&device_heap, 0, &device_desc).unwrap();
    let alias_b = memory.place_buffer(&device_heap, 0, &device_desc).unwrap();
    drop(upload_heap);
    drop(device_heap);
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage: BufferUsage::MAP_READ | BufferUsage::COPY_DST,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let expected: Vec<u8> = (0..64).map(|i| (i * 13 + 7) as u8).collect();
    {
        let mut mapped =
            unsafe { block(device.map_buffer(&upload, 0..64, MapMode::Write)) }.unwrap();
        mapped.bytes_mut().unwrap().copy_from_slice(&expected);
        mapped.flush(0..64).unwrap();
    }
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let copy_read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let copy_write = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    };
    let transition = |buffer, from, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer,
                range: 0..64,
            },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .transition_resources(&[transition(&upload, empty, copy_read)])
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Aliasing {
            before: None,
            after: Some(AliasedResource::Buffer(&alias_a)),
        }])
        .unwrap();
    encoder
        .transition_resources(&[transition(&alias_a, empty, copy_write)])
        .unwrap();
    encoder.copy_buffer(&upload, 0, &alias_a, 0, 64).unwrap();
    encoder
        .transition_resources(&[
            ResourceTransition::Aliasing {
                before: Some(AliasedResource::Buffer(&alias_a)),
                after: Some(AliasedResource::Buffer(&alias_b)),
            },
            transition(&alias_b, empty, copy_read),
        ])
        .unwrap();
    encoder.copy_buffer(&alias_b, 0, &readback, 0, 64).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..64).unwrap();
    let point = submit(&device, lane, encoder.finish().unwrap());
    block(device.wait_completion(&point)).unwrap();
    assert_eq!(
        unsafe { block(device.read_after(ticket, &point)).unwrap().bytes },
        expected
    );
}

#[test]
fn placed_texture_aliasing_survives_heap_drop_and_copies_pixels() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let memory = device.memory().expect("DX12 placed-memory capability");
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_DST | TextureUsage::COPY_SRC | TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    };
    let req = memory.texture_requirements(&desc).unwrap();
    assert_ne!(req.native_type_bits & ((1 << 0) | (1 << 5)), 0);
    let heap = memory
        .heap(&HeapDesc {
            size: req.size,
            native_memory_class: if req.native_type_bits & (1 << 5) != 0 {
                5
            } else {
                req.native_type_bits.trailing_zeros()
            },
        })
        .unwrap();
    let alias_a = memory.place_texture(&heap, 0, &desc).unwrap();
    let alias_b = memory.place_texture(&heap, 0, &desc).unwrap();
    drop(heap);
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 1024,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut pixels = vec![0u8; 1024];
    for row in 0..4 {
        for column in 0..4 {
            let at = row * 256 + column * 4;
            pixels[at..at + 4].copy_from_slice(&[column as u8, row as u8, 73, 255]);
        }
    }
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Aliasing {
            before: None,
            after: Some(AliasedResource::Texture(&alias_a)),
        }])
        .unwrap();
    encoder
        .write_texture(
            TextureRegion {
                texture: &alias_a,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 4,
            },
            &pixels,
        )
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Aliasing {
            before: Some(AliasedResource::Texture(&alias_a)),
            after: Some(AliasedResource::Texture(&alias_b)),
        }])
        .unwrap();
    // An aliasing barrier does not transfer initialized contents to the new
    // D3D12 resource identity. Undefined activates B through the native
    // placed-resource discard before its first attachment write.
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &alias_b,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: TextureUses {
                    stages: StageMask::COLOR_OUTPUT,
                    accesses: AccessMask::COLOR_WRITE,
                },
            },
            layout: Some((ImageLayout::Undefined, ImageLayout::ColorAttachment)),
            ownership: None,
        })])
        .unwrap();
    encoder
        .clear_texture(
            &alias_b,
            0..1,
            0..1,
            TextureClearValue::Color(ClearColor::Float([0.0, 0.0, 0.0, 0.0])),
        )
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &alias_b,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COLOR_OUTPUT,
                    accesses: AccessMask::COLOR_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::ColorAttachment, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &alias_b,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &output,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let ticket = encoder.readback_buffer(&output, 0..1024).unwrap();
    let point = submit(&device, lane, encoder.finish().unwrap());
    block(device.wait_completion(&point)).unwrap();
    let actual = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    for row in 0..4 {
        for column in 0..4 {
            let at = row * 256 + column * 4;
            assert_eq!(&actual[at..at + 4], &[0, 0, 0, 0]);
        }
    }
}
