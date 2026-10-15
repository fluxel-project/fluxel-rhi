//! Opt-in Apple GPU checks for Metal command submission and readback.
//!
//! These tests execute only on Apple targets with `hardware-tests` enabled.
//! They assert bytes returned from a real GPU submission; compiling this file
//! on another host does not count as hardware evidence.
#![cfg(all(target_vendor = "apple", feature = "hardware-tests"))]

use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

use fluxel_rhi_core::api::*;
use fluxel_rhi_metal::{Metal, MetalInstance};

struct ThreadWake(std::thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
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

fn submit_and_wait(
    device: &Device<Metal>,
    lane: LaneId,
    work: RecordedWork<Metal>,
) -> CompletionPoint<Metal> {
    let id = PlanId {
        device: device.id(),
        ordinal: 1,
    };
    // SAFETY: this one-step plan is local to the supplied device and graphics
    // lane; the work item below was recorded on that same lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: the packet binds the matching device-local work to its plan step
    // and requests a completion marker after the encoded operations.
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
    let receipt = match outcome {
        SubmitOutcome::Accepted(receipt) => receipt,
        SubmitOutcome::Rejected { error, .. }
        | SubmitOutcome::PartiallyAccepted { error, .. }
        | SubmitOutcome::Indeterminate { error, .. } => panic!("Metal submit failed: {error:?}"),
    };
    let point = receipt
        .checkpoints
        .into_iter()
        .next()
        .expect("End completion checkpoint")
        .point;
    block_on(device.wait_completion(&point)).unwrap();
    point
}

#[test]
fn buffer_upload_submission_and_readback_match_gpu_bytes() {
    let device = block_on(MetalInstance::new().unwrap().open_default()).unwrap();
    let lane = device.lanes()[0].id;
    let buffer = device
        .create_buffer(&BufferDesc {
            label: Some("Metal hardware readback source".into()),
            size: 32,
            usage: BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let expected = (0..32).map(|value| value ^ 0x5a).collect::<Vec<_>>();

    let mut encoder = device.encoder(lane).unwrap();
    encoder.write_buffer(&buffer, 0, &expected).unwrap();
    let ticket = encoder
        .readback_buffer(&buffer, 0..expected.len() as u64)
        .unwrap();
    let work = encoder.finish().unwrap();

    let point = submit_and_wait(&device, lane, work);
    // SAFETY: the point is the End checkpoint from the submission that encoded
    // this ticket's readback copy, and it was awaited immediately above.
    let actual = block_on(unsafe { device.read_after(ticket, &point) }).unwrap();
    assert_eq!(actual.bytes, expected);
}

#[test]
fn texture_upload_blit_and_readback_preserve_texels() {
    let device = block_on(MetalInstance::new().unwrap().open_default()).unwrap();
    let lane = device.lanes()[0].id;
    let extent = [2, 2, 1];
    let texture = device
        .create_texture(&TextureDesc {
            label: Some("Metal texture transfer hardware source".into()),
            dimension: TextureDimension::D2,
            extent,
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_DST.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap();
    let destination = device
        .create_buffer(&BufferDesc {
            label: Some("Metal texture transfer hardware readback".into()),
            size: 16,
            usage: BufferUsage::COPY_DST.union(BufferUsage::COPY_SRC),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let expected = [
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    ];

    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .write_texture(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent,
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 8,
                rows_per_image: 2,
            },
            &expected,
        )
        .unwrap();
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent,
                aspect: TextureAspects::COLOR,
            },
            &destination,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 8,
                rows_per_image: 2,
            },
        )
        .unwrap();
    let ticket = encoder.readback_buffer(&destination, 0..16).unwrap();
    let work = encoder.finish().unwrap();

    let point = submit_and_wait(&device, lane, work);
    // SAFETY: the awaited point covers both the texture upload and the later
    // texture-to-buffer copy represented by this ticket.
    let actual = block_on(unsafe { device.read_after(ticket, &point) }).unwrap();
    assert_eq!(actual.bytes, expected);
}

#[test]
fn compute_storage_texture_output_matches_readback_bytes() {
    let device = block_on(MetalInstance::new().unwrap().open_default()).unwrap();
    let lane = device.lanes()[0].id;
    let texture = device
        .create_texture(&TextureDesc {
            label: Some("Metal compute hardware output".into()),
            dimension: TextureDimension::D2,
            extent: [2, 2, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::STORAGE_BINDING.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap();
    let view = device
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
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("Metal compute hardware layout".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageTexture {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2,
                    access: StorageAccess::Write,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let entries = [BindGroupEntry {
        binding: 0,
        resource: BindingResource::TextureView(&view),
    }];
    let group = device
        .create_bind_group(&BindGroupDesc {
            label: Some("Metal compute hardware group"),
            layout: &layout,
            entries: &entries,
        })
        .unwrap();
    let groups = [&layout];
    let pipeline_layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("Metal compute hardware pipeline layout"),
            groups: &groups,
            push_constants: &[],
        })
        .unwrap();
    let source = r#"
        @group(0) @binding(0) var output_image: texture_storage_2d<rgba8unorm, write>;
        @compute @workgroup_size(1, 1, 1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            textureStore(output_image, vec2<i32>(id.xy), vec4<f32>(1.0, 0.0, 0.0, 1.0));
        }
    "#;
    let shader = block_on(device.shader(ShaderDesc {
        label: Some("Metal compute hardware shader".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source.as_bytes()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    }))
    .unwrap();
    let pipeline = block_on(
        device
            .compute()
            .expect("Metal compute capability")
            .create_pipeline(ComputePipelineDesc {
                label: Some("Metal compute hardware pipeline".into()),
                layout: pipeline_layout,
                compute: PipelineStage {
                    module: shader,
                    entry: "main".into(),
                    constants: vec![],
                },
                required_subgroup_size: None,
                cache: None,
            }),
    )
    .unwrap();
    let destination = device
        .create_buffer(&BufferDesc {
            label: Some("Metal compute readback destination".into()),
            size: 16,
            usage: BufferUsage::COPY_DST.union(BufferUsage::COPY_SRC),
            memory: MemoryClass::Device,
        })
        .unwrap();

    let mut encoder = device.encoder(lane).unwrap();
    let mut compute = device.compute().unwrap().begin(&mut encoder).unwrap();
    compute.set_pipeline(&pipeline).unwrap();
    compute.set_bind_group(0, &group, &[]).unwrap();
    compute.dispatch([2, 2, 1]).unwrap();
    compute.finish().unwrap();
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 2, 1],
                aspect: TextureAspects::COLOR,
            },
            &destination,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 8,
                rows_per_image: 2,
            },
        )
        .unwrap();
    let ticket = encoder.readback_buffer(&destination, 0..16).unwrap();
    let work = encoder.finish().unwrap();

    let point = submit_and_wait(&device, lane, work);
    // SAFETY: the awaited point covers compute execution and both copies.
    let actual = block_on(unsafe { device.read_after(ticket, &point) }).unwrap();
    assert_eq!(actual.bytes, [255, 0, 0, 255].repeat(4));
}
