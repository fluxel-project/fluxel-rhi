//! Private-SPI fixed SSBO array execution and dynamic offset readback.
use super::*;

async fn binding_dynamic_storage(d: &Device<Gl>) {
    let Some(compute) = d.compute() else {
        eprintln!("SKIP: device has no compute capability");
        return;
    };
    let uniform_align = d.capabilities().limits.uniform_offset_alignment.max(16);
    let storage_align = d.capabilities().limits.storage_offset_alignment.max(16);
    let uniform_offset = u32::try_from(uniform_align).unwrap();
    let storage_offset = u32::try_from(storage_align).unwrap();
    let source = br#"struct U { value: u32, pad0:u32, pad1:u32, pad2:u32 };
struct S { value: atomic<u32>, tail: array<u32> };
@group(0) @binding(0) var<uniform> u:U;
@group(0) @binding(1) var<storage, read_write> s:binding_array<S,2>;
@compute @workgroup_size(1) fn cs(@builtin(global_invocation_id) id:vec3u) {
 let index=id.x; let old=atomicAdd(&s[index].value,5u); s[index].tail[0]=9u;
 let n=arrayLength(&s[index].tail); let v=s[index].tail[0];
 atomicStore(&s[1].value,old+n+v+u.value);
}"#;
    let shader = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "cs".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: vec![],
        })
        .await
        .unwrap();
    let entry = |binding, ty| BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Compute],
        ty,
        count: if binding == 1 {
            BindingCount::Fixed(2)
        } else {
            BindingCount::One
        },
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let bgl_desc = BindGroupLayoutDesc {
        label: None,
        entries: vec![
            entry(
                0,
                BindingType::UniformBuffer {
                    dynamic_offset: true,
                    min_size: 16,
                },
            ),
            entry(
                1,
                BindingType::StorageBuffer {
                    dynamic_offset: true,
                    min_size: 16,
                    access: StorageAccess::ReadWrite,
                },
            ),
        ],
    };
    let bgl = d.create_bind_group_layout(&bgl_desc).unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&bgl],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = compute
        .create_pipeline(ComputePipelineDesc {
            label: None,
            layout,
            compute: PipelineStage {
                module: shader,
                entry: "cs".into(),
                constants: vec![],
            },
            required_subgroup_size: None,
            cache: None,
        })
        .await
        .unwrap();
    let uniform = d
        .create_buffer(&BufferDesc {
            label: None,
            size: uniform_align + 16,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let storage = d
        .create_buffer(&BufferDesc {
            label: None,
            size: storage_align + 16,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let arguments = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 28,
            usage: BufferUsage::INDIRECT | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let other = d
        .create_buffer(&BufferDesc {
            label: None,
            size: storage_align + 16,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &uniform,
                        range: 0..16,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Array(vec![
                        Some(BindingResource::Buffer {
                            buffer: &other,
                            range: 0..16,
                        }),
                        Some(BindingResource::Buffer {
                            buffer: &storage,
                            range: 0..16,
                        }),
                    ]),
                },
            ],
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 152).unwrap();
    // SAFETY: context and all resources remain alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    e.write_buffer(&uniform, 0, &[0; 16]).unwrap();
    e.write_buffer(
        &uniform,
        u64::from(uniform_offset),
        &[123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    .unwrap();
    e.write_buffer(&storage, 0, &[0; 16]).unwrap();
    e.write_buffer(&storage, u64::from(storage_offset), &[0; 16])
        .unwrap();
    e.write_buffer(&other, u64::from(storage_offset), &[17, 0, 0, 0])
        .unwrap();
    e.write_buffer(&arguments, 0, &[0; 16]).unwrap();
    e.write_buffer(&arguments, 16, &[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        .unwrap();
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &arguments,
            range: 16..28,
        },
        usage: StateTransition {
            from: BufferUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            },
            to: BufferUses {
                stages: StageMask::INDIRECT,
                accesses: AccessMask::INDIRECT_READ,
            },
        },
        ownership: None,
    })])
    .unwrap();
    let mut p = compute.begin(e).unwrap();
    p.set_pipeline(&pipeline).unwrap();
    p.set_bind_group(0, &group, &[uniform_offset, storage_offset])
        .unwrap();
    p.dispatch([1, 1, 1]).unwrap();
    p.finish().unwrap();
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &other,
            range: storage_align..storage_align + 16,
        },
        usage: StateTransition {
            from: BufferUses {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::SHADER_WRITE,
            },
            to: BufferUses {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::SHADER_READ | AccessMask::SHADER_WRITE,
            },
        },
        ownership: None,
    })])
    .unwrap();
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &storage,
            range: storage_align..storage_align + 16,
        },
        usage: StateTransition {
            from: BufferUses {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::SHADER_WRITE,
            },
            to: BufferUses {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::SHADER_READ | AccessMask::SHADER_WRITE,
            },
        },
        ownership: None,
    })])
    .unwrap();
    let mut p = compute.begin(e).unwrap();
    p.set_pipeline(&pipeline).unwrap();
    p.set_bind_group(0, &group, &[uniform_offset, storage_offset])
        .unwrap();
    p.dispatch_indirect(&arguments, 16).unwrap();
    p.finish().unwrap();
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &storage,
            range: 0..storage_align + 16,
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
    let ticket = e.readback_buffer(&storage, 0..storage_align + 16).unwrap();
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &other,
            range: storage_align..storage_align + 4,
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
    let other_ticket = e
        .readback_buffer(&other, storage_align..storage_align + 4)
        .unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: point covers the shader write and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let other_bytes = unsafe { d.read_after(other_ticket, &point).await }
        .unwrap()
        .bytes;
    assert_eq!(&other_bytes[..], &[27, 0, 0, 0]);
    assert_eq!(&bytes[..4], &[0, 0, 0, 0]);
    assert_eq!(
        &bytes[storage_offset as usize..storage_offset as usize + 4],
        &[157, 0, 0, 0]
    );
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_binding_dynamic_storage() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(binding_dynamic_storage(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_binding_dynamic_storage() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(binding_dynamic_storage(&d));
}
