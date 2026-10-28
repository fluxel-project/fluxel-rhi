//! Direct typed-RHI proof for divergent uniform-buffer array indexing.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::{future::Future, sync::Arc};

fn block_on<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

#[test]
fn divergent_uniform_buffer_array_indices_read_distinct_values() {
    let device = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    if !device.capabilities().nonuniform_uniform_buffer_indexing {
        eprintln!("SKIP: Vulkan uniform-buffer non-uniform indexing unavailable");
        return;
    }
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let inputs: Vec<_> = [11u32, 22]
        .into_iter()
        .map(|value| {
            let buffer = device
                .create_buffer(&BufferDesc {
                    label: Some("nonuniform uniform input".into()),
                    size: 4,
                    usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                    memory: MemoryClass::Device,
                })
                .unwrap();
            (buffer, value)
        })
        .collect();
    let output = device
        .create_buffer(&BufferDesc {
            label: Some("nonuniform uniform output".into()),
            size: 8,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("nonuniform uniform readback".into()),
            size: 8,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("nonuniform uniform layout".into()),
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::UniformBuffer {
                        dynamic_offset: false,
                        min_size: 4,
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
                        access: StorageAccess::Write,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
            ],
        })
        .unwrap();
    let mut entries = inputs
        .iter()
        .map(|(buffer, _)| BindingResource::Buffer {
            buffer,
            range: 0..4,
        })
        .collect::<Vec<_>>();
    let group = device
        .create_bind_group(&BindGroupDesc {
            label: Some("nonuniform uniform group".into()),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Array(entries.drain(..).map(Some).collect()),
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
            label: None,
            groups: &[&layout],
            push_constants: &[],
        })
        .unwrap();
    let shader = block_on(device.shader(ShaderDesc {
        label: Some("divergent uniform buffer array".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(b"struct Item { value: u32, }; @group(0) @binding(0) var<uniform> values: binding_array<Item, 2>; @group(0) @binding(1) var<storage, read_write> output: array<u32, 2>; @compute @workgroup_size(2) fn main(@builtin(local_invocation_index) i: u32) { output[i] = values[i].value; }".as_slice()),
        entries: vec![ShaderEntryPoint { name: "main".into(), stage: ShaderStage::Compute, workgroup_size: Some([2, 1, 1]) }],
        required_features: vec![],
    })).unwrap();
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
    let mut encoder = device.encoder(lane).unwrap();
    for (buffer, value) in &inputs {
        encoder
            .write_buffer(buffer, 0, &value.to_le_bytes())
            .unwrap();
    }
    let transition = |buffer, range, from, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice { buffer, range },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
    let copy_write = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    };
    let compute_read = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::UNIFORM_READ,
    };
    let compute_write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let copy_read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let mut barriers = inputs
        .iter()
        .map(|(buffer, _)| transition(buffer, 0..4, copy_write, compute_read))
        .collect::<Vec<_>>();
    barriers.push(transition(
        &output,
        0..8,
        BufferUses {
            stages: StageMask::EMPTY,
            accesses: AccessMask::EMPTY,
        },
        compute_write,
    ));
    encoder.transition_resources(&barriers).unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    encoder
        .transition_resources(&[transition(&output, 0..8, compute_write, copy_read)])
        .unwrap();
    encoder.copy_buffer(&output, 0, &readback, 0, 8).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..8).unwrap();
    let work = encoder.finish().unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 0x0b00_f001,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    let point = match unsafe {
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
        _ => panic!("uniform buffer array submission rejected"),
    };
    block_on(device.wait_completion(&point)).unwrap();
    let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 11);
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 22);
}
