//! Hardware evidence for compute indirect, dynamic storage offsets and cache.
#![cfg(feature = "hardware-tests")]
#[path = "support/mod.rs"]
mod support;
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::sync::Arc;
use support::block;
#[test]
fn indirect_dynamic_storage_specialization_and_cache() {
    run(false, 0);
    run(true, 0);
    run(true, 1);
    run(true, 2);
}
fn run(wgsl: bool, array: u8) {
    let d = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    if (array == 1 && !d.capabilities().binding_arrays)
        || (array == 2 && !d.capabilities().runtime_sized_binding_arrays)
    {
        eprintln!("descriptor array capability unavailable; skipping case {array}");
        return;
    }
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let out = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 512,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let args = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 12,
            usage: BufferUsage::INDIRECT | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 12,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: true,
                    min_size: 12,
                    access: StorageAccess::ReadWrite,
                },
                count: match array {
                    1 => BindingCount::Fixed(2),
                    2 => BindingCount::RuntimeSized { max_count: 4 },
                    _ => BindingCount::One,
                },
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let resource = BindingResource::StorageBufferView {
        buffer: &out,
        range: 0..12,
        view: StorageBufferView::Raw,
    };
    let resource = if array != 0 {
        BindingResource::Array(vec![
            Some(BindingResource::StorageBufferView {
                buffer: &out,
                range: 0..12,
                view: StorageBufferView::Raw,
            }),
            Some(resource),
        ])
    } else {
        resource
    };
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource,
            }],
        })
        .unwrap();
    let pl = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[
                PushConstantRange {
                    stages: vec![ShaderStage::Vertex],
                    range: 0..4,
                },
                PushConstantRange {
                    stages: vec![ShaderStage::Compute],
                    range: 0..4,
                },
            ],
        })
        .unwrap();
    let shader = block(d.shader(ShaderDesc {
        label: None,
        format: if wgsl { ShaderFormat::Wgsl } else { ShaderFormat::SpirV },
        bytes: if wgsl {
            Arc::from(b"override value: u32 = 10u; struct Push { addend: u32, } var<push_constant> pc: Push; @group(0) @binding(0) var<storage, read_write> output_data: array<u32>; @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) { output_data[id.x] = value + pc.addend + id.x; }".as_slice())
        } else { Arc::from(include_bytes!("assets/compute_features/compute.spv").as_slice()) },
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    }))
    .unwrap();
    let cache = d.create_pipeline_cache(None).unwrap();
    let create = |cache| {
        block(d.compute().unwrap().create_pipeline(ComputePipelineDesc {
            label: None,
            layout: pl.clone(),
            compute: PipelineStage {
                module: shader.clone(),
                entry: "main".into(),
                constants: vec![("value".into(), SpecializationValue::U32(42))],
            },
            cache: Some(cache),
            required_subgroup_size: None,
        }))
        .unwrap()
    };
    let _first = create(cache.clone());
    let serialized = d.serialize_pipeline_cache(&cache).unwrap();
    let reopened = d.create_pipeline_cache(Some(&serialized)).unwrap();
    let pipeline = create(reopened);
    assert!(d.create_pipeline_cache(Some(&[0; 32])).is_err());
    let mut e = d.encoder(lane).unwrap();
    e.write_buffer(&args, 0, &[3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        .unwrap();
    let barrier = |buffer, range, from, to| {
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
    let indirect_read = BufferUses {
        stages: StageMask::INDIRECT,
        accesses: AccessMask::INDIRECT_READ,
    };
    let compute_write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let copy_read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    e.transition_resources(&[barrier(&args, 0..12, copy_write, indirect_read)])
        .unwrap();
    {
        let mut pass = d.compute().unwrap().begin(&mut e).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[256]).unwrap();
        pass.push_constants(0, &5u32.to_le_bytes()).unwrap();
        pass.dispatch_indirect(&args, 0).unwrap();
        pass.finish().unwrap();
    }
    e.transition_resources(&[barrier(&out, 0..512, compute_write, copy_read)])
        .unwrap();
    e.copy_buffer(&out, 256, &dst, 0, 12).unwrap();
    let ticket = e.readback_buffer(&dst, 0..12).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 759123,
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
                payload: StepPayload::Commands(vec![e.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match result {
        SubmitOutcome::Accepted(r) => r.checkpoints[0].point.clone(),
        _ => panic!("submission rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(bytes, [47, 0, 0, 0, 48, 0, 0, 0, 49, 0, 0, 0]);
}
