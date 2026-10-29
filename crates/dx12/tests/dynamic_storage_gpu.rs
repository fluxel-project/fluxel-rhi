//! Dynamic SRV/UAV view offsets and descriptor lifetime on DX12.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::{Dx12, Dx12Instance};
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;
fn barrier(
    buffer: &Buffer<Dx12>,
    from: BufferUses,
    to: BufferUses,
) -> ResourceTransition<'_, Dx12> {
    ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer,
            range: 0..64,
        },
        usage: StateTransition { from, to },
        ownership: None,
    })
}
#[test]
fn dynamic_storage_views_execute_and_retain_descriptors() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let buffer = |usage| {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let input = buffer(BufferUsage::STORAGE | BufferUsage::COPY_DST);
    let output = buffer(BufferUsage::STORAGE | BufferUsage::COPY_SRC);
    let readback = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let entry = |binding, access| BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Compute],
        ty: BindingType::StorageBuffer {
            dynamic_offset: true,
            min_size: 16,
            access,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry(0, StorageAccess::Read),
                entry(1, StorageAccess::ReadWrite),
            ],
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::StorageBufferView {
                        buffer: &input,
                        range: 0..16,
                        view: StorageBufferView::Raw,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::StorageBufferView {
                        buffer: &output,
                        range: 0..16,
                        view: StorageBufferView::Raw,
                    },
                },
            ],
        })
        .unwrap();
    let pl = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[],
        })
        .unwrap();
    let shader = block(d.shader(ShaderDesc { label: None, format: ShaderFormat::Wgsl, bytes: Arc::from(&b"@group(0) @binding(0) var<storage,read> input: array<u32>; @group(0) @binding(1) var<storage,read_write> output: array<u32>; @compute @workgroup_size(1) fn main() { output[0] = arrayLength(&input); output[1] = input[0]; output[2] = arrayLength(&output); }"[..]), entries: vec![ShaderEntryPoint { name: "main".into(), stage: ShaderStage::Compute, workgroup_size: Some([1,1,1]) }], required_features: vec![] })).unwrap();
    let pipeline = block(d.compute().unwrap().create_pipeline(ComputePipelineDesc {
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
    for offsets in [&[1, 16][..], &[52, 16][..], &[16, 52][..], &[16][..]] {
        let mut e = d.encoder(lane).unwrap();
        let mut p = d.compute().unwrap().begin(&mut e).unwrap();
        p.set_pipeline(&pipeline).unwrap();
        assert!(p.set_bind_group(0, &group, offsets).is_err());
    }
    let mut e = d.encoder(lane).unwrap();
    let mut data = [0u8; 64];
    data[16..20].copy_from_slice(&42u32.to_le_bytes());
    e.write_buffer(&input, 0, &data).unwrap();
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let read = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_READ,
    };
    let write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let copy = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    e.transition_resources(&[barrier(&input, empty, read), barrier(&output, empty, write)])
        .unwrap();
    {
        let mut p = d.compute().unwrap().begin(&mut e).unwrap();
        p.set_pipeline(&pipeline).unwrap();
        p.set_bind_group(0, &group, &[16, 16]).unwrap();
        p.dispatch([1, 1, 1]).unwrap();
        p.finish().unwrap();
    }
    e.transition_resources(&[barrier(&output, write, copy)])
        .unwrap();
    e.copy_buffer(&output, 16, &readback, 0, 16).unwrap();
    let ticket = e.readback_buffer(&readback, 0..16).unwrap();
    let work = e.finish().unwrap();
    drop(group);
    drop(input);
    drop(output);
    drop(pipeline);
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 97134,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    let result = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match result {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    let result: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(&result[..3], &[4, 42, 4]);
}
