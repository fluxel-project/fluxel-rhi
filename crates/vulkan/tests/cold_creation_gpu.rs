//! Real-driver coverage for nonblocking Vulkan cold shader/PSO creation.
#![cfg(feature = "hardware-tests")]

use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    time::Instant,
};

struct ThreadWake(std::thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn block<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

fn shader_desc() -> ShaderDesc {
    ShaderDesc {
        label: Some("cold creation burst".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(b"@group(0) @binding(0) var<storage, read_write> output: array<u32>; @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) { output[id.x] = 0x51u + id.x; }".as_slice()),
        entries: vec![ShaderEntryPoint { name: "main".into(), stage: ShaderStage::Compute, workgroup_size: Some([1, 1, 1]) }],
        required_features: vec![],
    }
}

/// A burst larger than the bounded queue must remain valid, be cancellable,
/// and leave a usable real PSO. The timing is diagnostic only: scheduling and
/// driver compilation make a wall-clock pass/fail threshold meaningless.
#[test]
fn cold_shader_and_pipeline_burst_cancel_then_dispatch_readback() {
    const BURST: usize = 40;
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let wake_count = Arc::new(AtomicUsize::new(0));
    struct CountWake(Arc<AtomicUsize>);
    impl Wake for CountWake {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    let waker = Waker::from(Arc::new(CountWake(wake_count.clone())));
    let mut context = Context::from_waker(&waker);

    let mut shaders: Vec<Option<Pin<Box<_>>>> = (0..BURST)
        .map(|_| Some(Box::pin(device.shader(shader_desc()))))
        .collect();
    let started = Instant::now();
    let mut shader_ready = None;
    for index in 0..shaders.len() {
        let result =
            shaders[index]
                .as_mut()
                .and_then(|future| match future.as_mut().poll(&mut context) {
                    Poll::Ready(result) => Some(result),
                    Poll::Pending => None,
                });
        if let Some(result) = result {
            shaders[index].take();
            if index == 0 {
                shader_ready = Some(result.unwrap());
            }
        }
    }
    let shader_first_poll = started.elapsed();
    for future in shaders.iter_mut().skip(1) {
        future.take();
    }
    let shader = shader_ready.unwrap_or_else(|| block(shaders[0].take().unwrap()).unwrap());

    let output = device
        .create_buffer(&BufferDesc {
            label: Some("cold creation output".into()),
            size: 16,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("cold creation readback".into()),
            size: 16,
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
                    min_size: 16,
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
                    range: 0..16,
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
    let desc = || ComputePipelineDesc {
        label: Some("cold creation pipeline".into()),
        layout: layout.clone(),
        compute: PipelineStage {
            module: shader.clone(),
            entry: "main".into(),
            constants: vec![],
        },
        required_subgroup_size: None,
        cache: None,
    };
    let compute = device.compute().unwrap();
    let mut pipelines: Vec<Option<Pin<Box<_>>>> = (0..BURST)
        .map(|_| Some(Box::pin(compute.create_pipeline(desc()))))
        .collect();
    let started = Instant::now();
    let mut pipeline_ready = None;
    for index in 0..pipelines.len() {
        let result =
            pipelines[index]
                .as_mut()
                .and_then(|future| match future.as_mut().poll(&mut context) {
                    Poll::Ready(result) => Some(result),
                    Poll::Pending => None,
                });
        if let Some(result) = result {
            pipelines[index].take();
            if index == 0 {
                pipeline_ready = Some(result.unwrap());
            }
        }
    }
    let pipeline_first_poll = started.elapsed();
    for future in pipelines.iter_mut().skip(1) {
        future.take();
    }
    let pipeline = pipeline_ready.unwrap_or_else(|| block(pipelines[0].take().unwrap()).unwrap());
    eprintln!(
        "Vulkan cold burst: shader first-poll {:?}, pipeline first-poll {:?}, wakes {}",
        shader_first_poll,
        pipeline_first_poll,
        wake_count.load(Ordering::Relaxed),
    );

    let mut encoder = device.encoder(lane).unwrap();
    let undefined = BufferUses {
        stages: StageMask(0),
        accesses: AccessMask(0),
    };
    let shader_write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let copy_read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let barrier = |from, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &output,
                range: 0..16,
            },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
    encoder
        .transition_resources(&[barrier(undefined, shader_write)])
        .unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([4, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    encoder
        .transition_resources(&[barrier(shader_write, copy_read)])
        .unwrap();
    encoder.copy_buffer(&output, 0, &readback, 0, 16).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..16).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 8_245_001,
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
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![encoder.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(result) => result.checkpoints[0].point.clone(),
        _ => panic!("cold fixture submit rejected"),
    };
    block(device.wait_completion(&point)).unwrap();
    // SAFETY: the completion point covers the readback ticket created above.
    let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(
        bytes,
        [0x51, 0, 0, 0, 0x52, 0, 0, 0, 0x53, 0, 0, 0, 0x54, 0, 0, 0]
    );
}
