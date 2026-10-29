//! Direct typed-RHI proof for divergent storage-texture array indexing.
#![cfg(all(windows, feature = "hardware-tests"))]

#[path = "support/mod.rs"]
mod support;
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::sync::Arc;
use support::block_on;

#[test]
fn divergent_storage_texture_array_writes_each_selected_texture() {
    let device = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    assert!(device.capabilities().nonuniform_storage_texture_indexing);
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
                    label: None,
                    dimension: TextureDimension::D2,
                    extent: [1, 1, 1],
                    array_layers: 1,
                    format: TextureFormat::Rgba8Unorm,
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsage::STORAGE_BINDING | TextureUsage::COPY_SRC,
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
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("storage texture array".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageTexture {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2,
                    access: StorageAccess::Write,
                },
                count: BindingCount::Fixed(2),
                partially_bound: false,
                nonuniform_indexing: true,
            }],
        })
        .unwrap();
    let group = device
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Array(vec![
                    Some(BindingResource::TextureView(&views[0])),
                    Some(BindingResource::TextureView(&views[1])),
                ]),
            }],
        })
        .unwrap();
    let pipeline_layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[],
        })
        .unwrap();
    let shader = block_on(device.shader(ShaderDesc {
        label: Some("divergent storage texture array".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(
            b"@group(0) @binding(0) var images: binding_array<texture_storage_2d<rgba8unorm, write>, 2>; @compute @workgroup_size(2) fn main(@builtin(local_invocation_index) i: u32) { let color = select(vec4f(17.0 / 255.0, 0.0, 0.0, 1.0), vec4f(93.0 / 255.0, 0.0, 0.0, 1.0), i == 1u); textureStore(images[i], vec2i(0), color); }".as_slice(),
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
                label: None,
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
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 512,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    for texture in &textures {
        encoder
            .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
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
                        accesses: AccessMask::SHADER_WRITE,
                    },
                },
                layout: Some((ImageLayout::Undefined, ImageLayout::General)),
                ownership: None,
            })])
            .unwrap();
    }
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    for (index, texture) in textures.iter().enumerate() {
        encoder
            .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture,
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
                    texture,
                    mip: 0,
                    origin: [0, 0, 0],
                    extent: [1, 1, 1],
                    aspect: TextureAspects::COLOR,
                },
                &readback,
                TexelCopyBufferLayout {
                    offset: index as u64 * 256,
                    bytes_per_row: 256,
                    rows_per_image: 1,
                },
            )
            .unwrap();
    }
    let ticket = encoder.readback_buffer(&readback, 0..512).unwrap();
    let work = encoder.finish().unwrap();
    // SAFETY: the plan and recorded work use this device's lane and remain alive to completion.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 8_421_117,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: every object used by the recorded work is retained until its completion point.
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
        SubmitOutcome::Accepted(result) => result.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block_on(device.wait_completion(&point)).unwrap();
    // SAFETY: the completion point covers the ticket returned by this encoder.
    let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(&bytes[..4], &[17, 0, 0, 255]);
    assert_eq!(&bytes[256..260], &[93, 0, 0, 255]);
}
