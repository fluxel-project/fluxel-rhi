//! Rectangular nearest and linear blit readback.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};
fn block<F: Future>(f: F) -> F::Output {
    let mut c = std::task::Context::from_waker(std::task::Waker::noop());
    let mut f = std::pin::pin!(f);
    loop {
        if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut c) {
            return v;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}
#[test]
fn array_layers_and_srgb_blits_preserve_outside_pixels() {
    run(LaneClass::Graphics);
    run(LaneClass::Compute);
    // SAFETY: this fixture has one test and changes its process-local environment only.
    unsafe { std::env::set_var("FLUXEL_DX12_TEST_BUFFER_BLIT", "1") };
    run(LaneClass::Compute);
}
fn run(lane_class: LaneClass) {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d.lanes().iter().find(|l| l.class == lane_class).unwrap().id;
    for (i, (format, filter)) in [
        (TextureFormat::Rgba8Unorm, FilterMode::Nearest),
        (TextureFormat::Rgba8UnormSrgb, FilterMode::Nearest),
        (TextureFormat::Bgra8UnormSrgb, FilterMode::Nearest),
        (TextureFormat::Rgba8Uint, FilterMode::Nearest),
        (TextureFormat::Rgba8Sint, FilterMode::Nearest),
        (TextureFormat::Rgb10a2Unorm, FilterMode::Nearest),
    ]
    .into_iter()
    .enumerate()
    {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 3,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        };
        let src = d.create_texture(&desc).unwrap();
        let dst = d.create_texture(&desc).unwrap();
        let output = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 4,
                usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let group_layout = d
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: None,
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 4,
                        access: StorageAccess::ReadWrite,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                }],
            })
            .unwrap();
        let group = d
            .create_bind_group(&BindGroupDesc {
                label: None,
                layout: &group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::StorageBufferView {
                        buffer: &output,
                        range: 0..4,
                        view: StorageBufferView::Raw,
                    },
                }],
            })
            .unwrap();
        let pl = d
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: None,
                groups: &[&group_layout],
                push_constants: &[],
            })
            .unwrap();
        let shader = block(d.shader(ShaderDesc { label: None, format: ShaderFormat::Wgsl, bytes: Arc::from(&b"@group(0) @binding(0) var<storage,read_write> output: array<u32>; @compute @workgroup_size(1) fn main() { output[0] = 42; }"[..]), entries: vec![ShaderEntryPoint { name: "main".into(), stage: ShaderStage::Compute, workgroup_size: Some([1,1,1]) }], required_features: vec![] })).unwrap();
        let compute = block(d.compute().unwrap().create_pipeline(ComputePipelineDesc {
            label: None,
            layout: pl,
            compute: PipelineStage {
                module: shader,
                entry: "main".into(),
                constants: vec![],
            },
            cache: None,
            required_subgroup_size: None,
        }))
        .unwrap();
        let readback = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 2048,
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .unwrap();
        let region = |texture, origin, extent| TextureRegion {
            texture,
            mip: 0,
            origin,
            extent,
            aspect: TextureAspects::COLOR,
        };
        if matches!(format, TextureFormat::Rgba8Uint | TextureFormat::Rgba8Sint) {
            let mut invalid = d.encoder(lane).unwrap();
            assert!(matches!(
                invalid.blit(
                    region(&src, [0, 0, 0], [4, 4, 1]),
                    region(&dst, [0, 0, 0], [4, 4, 1]),
                    FilterMode::Linear
                ),
                Err(Error::Unsupported(_))
            ));
        }
        let mut pixels = [0u8; 64];
        for (n, pixel) in pixels.chunks_exact_mut(4).enumerate() {
            pixel.copy_from_slice(&[n as u8 * 10, 0, 0, 255]);
        }
        let mut second_pixels = pixels;
        for pixel in second_pixels.chunks_exact_mut(4) {
            pixel[0] += 1;
        }
        let mut e = d.encoder(lane).unwrap();
        let buffer_empty = BufferUses {
            stages: StageMask::EMPTY,
            accesses: AccessMask::EMPTY,
        };
        let buffer_write = BufferUses {
            stages: StageMask::COMPUTE,
            accesses: AccessMask::SHADER_WRITE,
        };
        let buffer_copy = BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        };
        e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &output,
                range: 0..4,
            },
            usage: StateTransition {
                from: buffer_empty,
                to: buffer_write,
            },
            ownership: None,
        })])
        .unwrap();
        {
            let mut pass = d.compute().unwrap().begin(&mut e).unwrap();
            pass.set_pipeline(&compute).unwrap();
            pass.finish().unwrap();
        }
        let layout = TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 16,
            rows_per_image: 4,
        };
        e.write_texture(region(&src, [0, 0, 0], [4, 4, 1]), layout, &pixels)
            .unwrap();
        e.write_texture(region(&src, [0, 0, 1], [4, 4, 1]), layout, &second_pixels)
            .unwrap();
        e.write_texture(region(&dst, [0, 0, 1], [4, 4, 1]), layout, &[0; 64])
            .unwrap();
        e.write_texture(region(&dst, [0, 0, 2], [4, 4, 1]), layout, &[0; 64])
            .unwrap();
        let empty = TextureUses {
            stages: StageMask::EMPTY,
            accesses: AccessMask::EMPTY,
        };
        let read = TextureUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        };
        let write = TextureUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        };
        let barrier = |texture, from, to, layouts| {
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..3,
                },
                usage: StateTransition { from, to },
                layout: Some(layouts),
                ownership: None,
            })
        };
        e.transition_resources(&[
            barrier(
                &src,
                empty,
                read,
                (ImageLayout::Common, ImageLayout::CopySrc),
            ),
            barrier(
                &dst,
                empty,
                write,
                (ImageLayout::Common, ImageLayout::CopyDst),
            ),
        ])
        .unwrap();
        e.blit(
            region(&src, [1, 1, 0], [2, 2, 2]),
            region(&dst, [1, 1, 1], [2, 2, 2]),
            filter,
        )
        .unwrap();
        {
            let mut pass = d.compute().unwrap().begin(&mut e).unwrap();
            pass.set_bind_group(0, &group, &[]).unwrap();
            pass.dispatch([1, 1, 1]).unwrap();
            pass.finish().unwrap();
        }
        e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &output,
                range: 0..4,
            },
            usage: StateTransition {
                from: buffer_write,
                to: buffer_copy,
            },
            ownership: None,
        })])
        .unwrap();
        let output_ticket = e.readback_buffer(&output, 0..4).unwrap();
        e.transition_resources(&[barrier(
            &dst,
            write,
            read,
            (ImageLayout::CopyDst, ImageLayout::CopySrc),
        )])
        .unwrap();
        for layer in 1..=2 {
            e.texture_to_buffer(
                region(&dst, [0, 0, layer], [4, 4, 1]),
                &readback,
                TexelCopyBufferLayout {
                    offset: u64::from(layer - 1) * 1024,
                    bytes_per_row: 256,
                    rows_per_image: 4,
                },
            )
            .unwrap();
        }
        let ticket = e.readback_buffer(&readback, 0..2048).unwrap();
        let work = e.finish().unwrap();
        drop(src);
        drop(dst);
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: d.id(),
                    ordinal: 99000 + i as u64,
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
                    payload: StepPayload::Commands(vec![work]),
                }],
                completion: CompletionRequest::End,
            })
        };
        let point = match outcome {
            SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
            _ => panic!("submit"),
        };
        block(d.wait_completion(&point)).unwrap();
        let data = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
        let output_data = unsafe { block(d.read_after(output_ticket, &point)).unwrap().bytes };
        assert_eq!(output_data, 42u32.to_le_bytes());
        for layer in 0..2 {
            let expected = if layer == 0 { &pixels } else { &second_pixels };
            for y in 0..4 {
                for x in 0..4 {
                    let p =
                        &data[layer * 1024 + y * 256 + x * 4..layer * 1024 + y * 256 + x * 4 + 4];
                    if (1..3).contains(&x) && (1..3).contains(&y) {
                        assert_eq!(p, &expected[(y * 4 + x) * 4..(y * 4 + x) * 4 + 4]);
                    } else {
                        assert_eq!(p, &[0; 4]);
                    }
                }
            }
        }
    }
}
