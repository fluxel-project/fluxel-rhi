//! Real-GPU cold creation burst coverage for the bounded DX12 worker.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Waker},
    time::Instant,
};

const SOURCE: &[u8] = b"@group(0) @binding(0) var<storage, read_write> output: array<u32>; @compute @workgroup_size(1) fn main() { output[0] = 0x12345678u; }";

fn wait<T>(future: &mut std::pin::Pin<Box<impl Future<Output = Result<T>>>>) -> T {
    let mut context = Context::from_waker(Waker::noop());
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(result) => return result.unwrap(),
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

#[test]
fn cold_creation_burst_backpressures_and_returns_usable_compute_pso() {
    let instance = Dx12Instance::new().unwrap();
    let mut open = Box::pin(instance.open_default());
    let device = wait(&mut open);
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let output = device
        .create_buffer(&BufferDesc {
            label: Some("cold creation output".into()),
            size: 4,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("cold creation readback".into()),
            size: 4,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let group_layout = device
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
    let group = device
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
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();

    let construct = Instant::now();
    let mut shader_futures: Vec<_> = (0..40)
        .map(|ordinal| {
            Box::pin(device.shader(ShaderDesc {
                label: Some(format!("cold shader {ordinal}")),
                format: ShaderFormat::Wgsl,
                bytes: Arc::from(SOURCE),
                entries: vec![ShaderEntryPoint {
                    name: "main".into(),
                    stage: ShaderStage::Compute,
                    workgroup_size: Some([1, 1, 1]),
                }],
                required_features: vec![],
            }))
        })
        .collect();
    let construct_ns = construct.elapsed().as_nanos();
    let first_poll = Instant::now();
    let mut context = Context::from_waker(Waker::noop());
    let mut shader_ready = (0..shader_futures.len()).map(|_| None).collect::<Vec<_>>();
    for (future, ready) in shader_futures.iter_mut().zip(&mut shader_ready) {
        if let Poll::Ready(result) = future.as_mut().poll(&mut context) {
            *ready = Some(result.unwrap());
        }
    }
    let first_poll_ns = first_poll.elapsed().as_nanos();
    // Cancel unadmitted/queued work after its first poll. This must not leave
    // capacity permanently unavailable to the owned futures below.
    shader_futures.truncate(32);
    shader_ready.truncate(32);
    let shaders = shader_futures
        .iter_mut()
        .zip(shader_ready)
        .map(|(future, ready)| ready.unwrap_or_else(|| wait(future)))
        .collect::<Vec<_>>();

    let compute = device.compute().unwrap();
    let pipeline_construct = Instant::now();
    let mut pipeline_futures: Vec<_> = shaders
        .into_iter()
        .enumerate()
        .map(|(ordinal, shader)| {
            Box::pin(compute.create_pipeline(ComputePipelineDesc {
                label: Some(format!("cold pipeline {ordinal}")),
                layout: layout.clone(),
                compute: PipelineStage {
                    module: shader,
                    entry: "main".into(),
                    constants: vec![],
                },
                required_subgroup_size: None,
                cache: None,
            }))
        })
        .collect();
    let pipeline_construct_ns = pipeline_construct.elapsed().as_nanos();
    let pipeline_first_poll = Instant::now();
    let mut pipeline_ready = (0..pipeline_futures.len())
        .map(|_| None)
        .collect::<Vec<_>>();
    for (future, ready) in pipeline_futures.iter_mut().zip(&mut pipeline_ready) {
        if let Poll::Ready(result) = future.as_mut().poll(&mut context) {
            *ready = Some(result.unwrap());
        }
    }
    let pipeline_first_poll_ns = pipeline_first_poll.elapsed().as_nanos();
    pipeline_futures.truncate(24);
    pipeline_ready.truncate(24);
    let pipeline = pipeline_ready[0]
        .take()
        .unwrap_or_else(|| wait(&mut pipeline_futures[0]));
    eprintln!(
        "dx12 cold creation: shader_construct_ns={construct_ns} shader_first_poll_ns={first_poll_ns} pipeline_construct_ns={pipeline_construct_ns} pipeline_first_poll_ns={pipeline_first_poll_ns}"
    );

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
    let mut encoder = device.encoder(lane).unwrap();
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
                range: 0..4,
            },
            usage: StateTransition {
                from: write,
                to: read,
            },
            ownership: None,
        })])
        .unwrap();
    encoder.copy_buffer(&output, 0, &readback, 0, 4).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..4).unwrap();
    let work = encoder.finish().unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 7_711_000,
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
            steps: vec![BoundStep {
                step: plan.step(0),
                payload: StepPayload::Commands(vec![work]),
            }],
            plan,
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("DX12 cold creation submission rejected"),
    };
    wait(&mut Box::pin(device.wait_completion(&point)));
    let bytes = unsafe { wait(&mut Box::pin(device.read_after(ticket, &point))).bytes };
    assert_eq!(
        u32::from_le_bytes(bytes[..4].try_into().unwrap()),
        0x1234_5678
    );
}
