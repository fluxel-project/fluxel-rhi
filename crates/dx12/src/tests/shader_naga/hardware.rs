//! Opt-in GPU execution check for generated DXIL with raw resource views.

use std::sync::Arc;

#[cfg(feature = "hardware-tests")]
#[test]
fn generated_dxil_executes_with_raw_views_on_dx12() {
    use crate::Dx12Instance;
    use fluxel_rhi_core::api::*;
    use std::future::Future;
    fn block<F: Future>(f: F) -> F::Output {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        let mut f = std::pin::pin!(f);
        loop {
            if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
                return v;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    let source = b"@group(0) @binding(1) var samp: sampler; @group(0) @binding(2) var image: texture_2d<f32>; @group(0) @binding(3) var<storage, read_write> data: array<u32>; @group(0) @binding(4) var output_image: texture_storage_2d<rgba8unorm, write>; override extra: u32 = 0u; @compute @workgroup_size(1) fn main() { data[0] = u32(round(textureSampleLevel(image, samp, vec2<f32>(0.5), 0.0).x * 255.0)) + extra; textureStore(output_image, vec2<i32>(0), vec4<f32>(1.0, 0.0, 0.0, 1.0)); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let sampler = device
        .create_sampler(&SamplerDesc {
            label: None,
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })
        .unwrap();
    let texture = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let texture_view = device
        .create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let output_image = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::STORAGE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let output_view = device
        .create_view(
            &output_image,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let image_readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let data = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let group_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::Sampler {
                        filtering: false,
                        comparison: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 16,
                        access: StorageAccess::ReadWrite,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageTexture {
                        format: TextureFormat::Rgba8Unorm,
                        dimension: TextureViewDimension::D2,
                        access: StorageAccess::Write,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
            ],
        })
        .unwrap();
    let group = device
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::StorageBufferView {
                        buffer: &data,
                        range: 0..16,
                        view: StorageBufferView::Raw,
                    },
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&output_view),
                },
            ],
        })
        .unwrap();
    let structured_group = device
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::StorageBufferView {
                        buffer: &data,
                        range: 0..16,
                        view: StorageBufferView::Structured { stride: 4 },
                    },
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&output_view),
                },
            ],
        })
        .unwrap();
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();
    let shader = block(device.shader(desc)).unwrap();
    let pipeline = block(
        device
            .compute()
            .unwrap()
            .create_pipeline(ComputePipelineDesc {
                label: None,
                layout,
                compute: PipelineStage {
                    module: shader,
                    entry: "main".into(),
                    constants: vec![("extra".into(), SpecializationValue::U32(40))],
                },
                cache: None,
                required_subgroup_size: None,
            }),
    )
    .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .write_texture(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [1, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 4,
                rows_per_image: 1,
            },
            &[2, 0, 0, 255],
        )
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &texture,
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
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_READ,
                },
            },
            layout: Some((ImageLayout::Common, ImageLayout::ShaderReadOnly)),
            ownership: None,
        })])
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &output_image,
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
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_WRITE,
                },
            },
            layout: Some((ImageLayout::Undefined, ImageLayout::General)),
            ownership: None,
        })])
        .unwrap();
    let transition = |from, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &data,
                range: 0..16,
            },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    encoder
        .transition_resources(&[transition(empty, write)])
        .unwrap();
    {
        let mut rejected_encoder = device.encoder(lane).unwrap();
        let mut rejected = device
            .compute()
            .unwrap()
            .begin(&mut rejected_encoder)
            .unwrap();
        rejected.set_pipeline(&pipeline).unwrap();
        assert!(rejected.set_bind_group(0, &structured_group, &[]).is_err());
    }
    {
        let mut rejected_encoder = device.encoder(lane).unwrap();
        let mut rejected = device
            .compute()
            .unwrap()
            .begin(&mut rejected_encoder)
            .unwrap();
        rejected.set_pipeline(&pipeline).unwrap();
        rejected.set_bind_group(0, &group, &[]).unwrap();
        rejected.set_pipeline(&pipeline).unwrap();
        assert!(rejected.dispatch([1, 1, 1]).is_err());
    }
    let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.dispatch([1, 1, 1]).unwrap();
    pass.finish().unwrap();
    encoder
        .transition_resources(&[transition(write, read)])
        .unwrap();
    encoder.copy_buffer(&data, 0, &readback, 0, 16).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..16).unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &output_image,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::General, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &output_image,
                mip: 0,
                origin: [0, 0, 0],
                extent: [1, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            &image_readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 1,
            },
        )
        .unwrap();
    let image_ticket = encoder.readback_buffer(&image_readback, 0..256).unwrap();
    let work = encoder.finish().unwrap();
    // SAFETY: the plan and recorded work use the same device and lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 19991,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: resources and work are retained until completion below.
    let result = unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match result {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block(device.wait_completion(&point)).unwrap();
    // SAFETY: the completion point covers the readback ticket.
    let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(u32::from_ne_bytes(bytes[..4].try_into().unwrap()), 42);
    let image_bytes = unsafe {
        block(device.read_after(image_ticket, &point))
            .unwrap()
            .bytes
    };
    assert_eq!(&image_bytes[..4], &[255, 0, 0, 255]);
}
