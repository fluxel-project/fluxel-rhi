//! Direct typed-RHI proof for divergent sampled-texture array indexing.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = std::task::Waker::noop();
    let mut context = std::task::Context::from_waker(waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

#[test]
fn divergent_sampled_texture_array_reads_distinct_values() {
    let instance = Dx12Instance::new().unwrap();
    let api_instance = instance.instance().unwrap();
    let adapters = api_instance.enumerate().unwrap();
    eprintln!(
        "DXGI adapters: {:?}",
        adapters
            .iter()
            .map(|adapter| adapter.name.as_str())
            .collect::<Vec<_>>()
    );
    let adapter = adapters.first().expect("DXGI adapter list is empty");
    let device = block_on(api_instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: vec![],
    }))
    .unwrap();
    assert!(device.capabilities().nonuniform_sampled_texture_indexing);
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;

    let textures: Vec<_> = (0..2)
        .map(|_| {
            device
                .create_texture(&TextureDesc {
                    label: Some("nonuniform sampled input".into()),
                    dimension: TextureDimension::D2,
                    extent: [1, 1, 1],
                    array_layers: 1,
                    format: TextureFormat::Rgba8Unorm,
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
                    view_formats: vec![],
                })
                .unwrap()
        })
        .collect();
    let views: Vec<_> = textures
        .iter()
        .map(|texture| {
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
        })
        .collect();
    let output = device
        .create_buffer(&BufferDesc {
            label: Some("nonuniform sampled output".into()),
            size: 8,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("nonuniform sampled readback".into()),
            size: 8,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();

    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("nonuniform sampled texture layout".into()),
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: BindingCount::Fixed(2),
                    partially_bound: false,
                    nonuniform_indexing: true,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 8,
                        access: StorageAccess::ReadWrite,
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
            label: Some("nonuniform sampled texture group".into()),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Array(
                        views
                            .iter()
                            .map(|view| Some(BindingResource::TextureView(view)))
                            .collect(),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::StorageBufferView {
                        buffer: &output,
                        range: 0..8,
                        view: StorageBufferView::Raw,
                    },
                },
            ],
        })
        .unwrap();
    let pipeline_layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("nonuniform sampled pipeline layout".into()),
            groups: &[&layout],
            push_constants: &[],
        })
        .unwrap();
    let shader = block_on(device.shader(ShaderDesc {
        label: Some("divergent sampled texture array".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(
            b"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>; @group(0) @binding(1) var<storage, read_write> output: array<u32, 2>; @compute @workgroup_size(2) fn main(@builtin(local_invocation_index) i: u32) { let value = textureLoad(images[i], vec2<i32>(0), 0).r; output[i] = u32(round(value * 255.0)); }"
                .as_slice(),
        ),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([2, 1, 1]),
        }],
        required_features: vec![],
    }))
    .unwrap();
    let pipeline = block_on(
        device
            .compute()
            .unwrap()
            .create_pipeline(ComputePipelineDesc {
                label: Some("nonuniform sampled pipeline".into()),
                layout: pipeline_layout,
                compute: PipelineStage {
                    module: shader,
                    entry: "main".into(),
                    constants: vec![],
                },
                cache: None,
                required_subgroup_size: None,
            }),
    )
    .unwrap();

    let mut encoder = device.encoder(lane).unwrap();
    for (texture, red) in textures.iter().zip([17u8, 93]) {
        encoder
            .write_texture(
                TextureRegion {
                    texture,
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
                &[red, 0, 0, 255],
            )
            .unwrap();
    }
    let mut barriers = textures
        .iter()
        .map(|texture| {
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture,
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
                layout: Some((ImageLayout::Undefined, ImageLayout::ShaderReadOnly)),
                ownership: None,
            })
        })
        .collect::<Vec<_>>();
    barriers.push(ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &output,
            range: 0..8,
        },
        usage: StateTransition {
            from: BufferUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            to: BufferUses {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::SHADER_WRITE,
            },
        },
        ownership: None,
    }));
    encoder.transition_resources(&barriers).unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    encoder
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &output,
                range: 0..8,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_WRITE,
                },
                to: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            ownership: None,
        })])
        .unwrap();
    encoder.copy_buffer(&output, 0, &readback, 0, 8).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..8).unwrap();
    let work = encoder.finish().unwrap();
    // SAFETY: the submission plan and recorded work use this device-owned lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 0x5a_4d_0001,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: every object referenced by the work stays alive through completion.
    let outcome = unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("sampled texture nonuniform submission rejected"),
    };
    block_on(device.wait_completion(&point)).unwrap();
    // SAFETY: completion covers this readback ticket.
    let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 17);
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 93);
}
