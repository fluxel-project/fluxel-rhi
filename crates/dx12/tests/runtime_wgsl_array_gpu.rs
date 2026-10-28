//! Public Core execution of bounded WGSL runtime descriptor arrays on DX12.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::{Dx12, Dx12Instance};
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

fn submit(device: &Device<Dx12>, lane: LaneId, work: RecordedWork<Dx12>) -> CompletionPoint<Dx12> {
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 49_162,
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
        SubmitOutcome::Accepted(mut receipt) => receipt.checkpoints.remove(0).point,
        SubmitOutcome::Rejected { error, .. } => panic!("runtime-array submit rejected: {error:?}"),
        SubmitOutcome::PartiallyAccepted { error, .. } => {
            panic!("runtime-array submit partially accepted: {error:?}")
        }
        SubmitOutcome::Indeterminate { error, .. } => {
            panic!("runtime-array submit indeterminate: {error:?}")
        }
    }
}

fn texture(device: &Device<Dx12>) -> Texture<Dx12> {
    device
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
        .unwrap()
}

fn sampled_view(device: &Device<Dx12>, texture: &Texture<Dx12>) -> TextureView<Dx12> {
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
}

#[test]
fn runtime_texture_sampler_arrays_select_nonzero_dynamic_element() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    if !device.capabilities().runtime_sized_binding_arrays
        || !device.capabilities().nonuniform_sampled_texture_indexing
        || !device.capabilities().nonuniform_sampler_indexing
    {
        return;
    }
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let a = texture(&device);
    let b = texture(&device);
    let a_view = sampled_view(&device, &a);
    let b_view = sampled_view(&device, &b);
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
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 4,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 4,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let runtime_array = |ty| BindGroupLayoutEntry {
        binding: 0,
        visibility: vec![ShaderStage::Compute],
        ty,
        count: BindingCount::RuntimeSized { max_count: 3 },
        partially_bound: true,
        nonuniform_indexing: true,
    };
    let image_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![runtime_array(BindingType::SampledTexture {
                sample: TextureSampleType::Float,
                dimension: TextureViewDimension::D2,
                multisampled: false,
            })],
        })
        .unwrap();
    let sampler_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![runtime_array(BindingType::Sampler {
                filtering: false,
                comparison: false,
            })],
        })
        .unwrap();
    let output_layout = device
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
    let image_group = device
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &image_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Array(vec![
                    Some(BindingResource::TextureView(&a_view)),
                    Some(BindingResource::TextureView(&b_view)),
                ]),
            }],
        })
        .unwrap();
    let sampler_group = device
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &sampler_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Array(vec![
                    Some(BindingResource::Sampler(&sampler)),
                    Some(BindingResource::Sampler(&sampler)),
                ]),
            }],
        })
        .unwrap();
    let output_group = device
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &output_layout,
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
    let pipeline_layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&image_layout, &sampler_layout, &output_layout],
            push_constants: &[],
        })
        .unwrap();
    let source = b"@group(0) @binding(0) var images:binding_array<texture_2d<f32>>; @group(1) @binding(0) var samplers:binding_array<sampler>; @group(2) @binding(0) var<storage,read_write> out:array<u32>; @compute @workgroup_size(2) fn main(@builtin(global_invocation_id) id:vec3<u32>) { let color=textureSampleLevel(images[id.x],samplers[id.x],vec2f(0.5),0.0); if id.x==1u { out[0]=u32(round(color.r*255.0)); } }";
    let shader = block(device.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source.as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([2, 1, 1]),
        }],
        required_features: vec![],
    }))
    .unwrap();
    let compute = device.compute().unwrap();
    let pipeline = block(compute.create_pipeline(ComputePipelineDesc {
        label: None,
        layout: pipeline_layout,
        compute: PipelineStage {
            module: shader,
            entry: "main".into(),
            constants: vec![],
        },
        required_subgroup_size: None,
        cache: None,
    }))
    .unwrap();
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let copy = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let mut encoder = device.encoder(lane).unwrap();
    for (texture, pixel) in [(&a, [2, 0, 0, 255]), (&b, [79, 0, 0, 255])] {
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
                &pixel,
            )
            .unwrap();
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
                        accesses: AccessMask::SHADER_READ,
                    },
                },
                layout: Some((ImageLayout::Undefined, ImageLayout::ShaderReadOnly)),
                ownership: None,
            })])
            .unwrap();
    }
    encoder
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &output,
                range: 0..4,
            },
            usage: StateTransition {
                from: empty,
                to: write,
            },
            ownership: None,
        })])
        .unwrap();
    let mut pass = compute.begin(&mut encoder).unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &image_group, &[]).unwrap();
    pass.set_bind_group(1, &sampler_group, &[]).unwrap();
    pass.set_bind_group(2, &output_group, &[]).unwrap();
    pass.dispatch([1, 1, 1]).unwrap();
    pass.finish().unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &output,
                range: 0..4,
            },
            usage: StateTransition {
                from: write,
                to: copy,
            },
            ownership: None,
        })])
        .unwrap();
    encoder.copy_buffer(&output, 0, &readback, 0, 4).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..4).unwrap();
    let point = submit(&device, lane, encoder.finish().unwrap());
    block(device.wait_completion(&point)).unwrap();
    assert_eq!(
        unsafe { block(device.read_after(ticket, &point)).unwrap().bytes },
        79u32.to_le_bytes()
    );
}
