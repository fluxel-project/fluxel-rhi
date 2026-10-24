//! WGSL raster shader specialization and binding ABI GPU checks.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

fn block_on<F: Future>(f: F) -> F::Output {
    let w = std::task::Waker::noop();
    let mut c = std::task::Context::from_waker(w);
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut c) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

#[test]
fn native_and_wgsl_draw_index_semantics() {
    for indexed in [false, true] {
        for indirect in 0..=2 {
            run(indexed, indirect, false, false);
            run(indexed, indirect, true, false);
        }
    }
    for indirect in 0..=2 {
        run(true, indirect, false, true);
    }
    run(false, 3, false, false);
    run(true, 3, false, true);
}

/// A diagnostic recording benchmark for the WGSL `vertex_index` indirect
/// lowering.  It is deliberately a real-GPU gate: the reported CPU intervals
/// are adjacent first/warm `multi_draw` calls on one pipeline, while the later
/// readback proves that every selected command count still produces the
/// expected pixel.
///
/// This is not a universal draw benchmark. It isolates the backend's native
/// lowering of a single non-indexed indirect batch with no user bindings.
#[test]
fn wgsl_vertex_index_indirect_recording_counts() {
    for count in [1, 3, 10_000] {
        let [cold, warm] = run_indirect_batch(count);
        eprintln!(
            "wgsl_vertex_index_indirect_recording count={count} cold_us={} warm_us={}",
            cold.as_micros(),
            warm.as_micros()
        );
    }
}

fn run(indexed: bool, indirect: u8, native: bool, negative: bool) {
    run_inner(indexed, indirect, native, negative, None);
}

#[derive(Clone, Copy)]
struct Batch {
    count: u32,
    stride: u32,
    offset: u64,
    gpu_count: bool,
    count_offset: u64,
}

#[test]
fn wgsl_packing_offset_stride_and_gpu_count_chunk_boundary() {
    for indexed in [false, true] {
        for stride in [if indexed { 20 } else { 16 }, 32] {
            run_inner(
                indexed,
                4,
                false,
                indexed,
                Some(Batch {
                    count: 257,
                    stride,
                    offset: 12,
                    gpu_count: false,
                    count_offset: 0,
                }),
            );
        }
    }
    run_inner(
        false,
        4,
        false,
        false,
        Some(Batch {
            count: 257,
            stride: 32,
            offset: 28,
            gpu_count: true,
            count_offset: 20,
        }),
    );
    // One record beyond the native dispatch-x limit. The GPU writes count=1;
    // the last record has a different color, catching an unclamped second chunk.
    run_inner(
        false,
        4,
        false,
        false,
        Some(Batch {
            count: 65_535 * 64 + 1,
            stride: 16,
            offset: 12,
            gpu_count: true,
            count_offset: 0,
        }),
    );
}

fn run_indirect_batch(count: u32) -> [Duration; 2] {
    run_inner(
        false,
        4,
        false,
        false,
        Some(Batch {
            count,
            stride: 16,
            offset: 0,
            gpu_count: false,
            count_offset: 0,
        }),
    )
    .expect("batch benchmark records a duration")
}

fn run_inner(
    indexed: bool,
    indirect: u8,
    native: bool,
    negative: bool,
    batch: Option<Batch>,
) -> Option<[Duration; 2]> {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let (vertex_shader, fragment_shader) = if native {
        let vertex = block_on(d.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Dxil,
            bytes: Arc::from(&include_bytes!("assets/native_vertex_index.vs.dxil")[..]),
            entries: vec![ShaderEntryPoint {
                name: "vs".into(),
                stage: ShaderStage::Vertex,
                workgroup_size: None,
            }],
            required_features: vec![],
        }))
        .unwrap();
        let fragment = block_on(d.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Dxil,
            bytes: Arc::from(&include_bytes!("assets/native_vertex_index.ps.dxil")[..]),
            entries: vec![ShaderEntryPoint {
                name: "ps".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: vec![],
        }))
        .unwrap();
        (vertex, fragment)
    } else {
        let source = b"struct V { @builtin(position) p: vec4<f32>, @location(0) @interpolate(flat) color: vec4<f32> }; @vertex fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) instance: u32) -> V { var o: V; let i = v - 5u; let x = f32((i << 1u) & 2u); let y = f32(i & 2u); o.p = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.5, 1.0); o.color = vec4<f32>(f32(v) / 255.0, f32(instance) / 255.0, 0.0, 1.0); return o; } @fragment fn ps_main(@location(0) @interpolate(flat) color: vec4<f32>) -> @location(0) vec4<f32> { return color; }";
        let shader = block_on(d.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(&source[..]),
            entries: vec![
                ShaderEntryPoint {
                    name: "vs_main".into(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                },
                ShaderEntryPoint {
                    name: "ps_main".into(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                },
            ],
            required_features: vec![],
        }))
        .unwrap();
        (shader.clone(), shader)
    };
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = block_on(d.raster_pipeline(RasterPipelineDesc {
        label: None,
        layout,
        vertex: PipelineStage {
            module: vertex_shader,
            entry: if native { "vs" } else { "vs_main" }.into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: fragment_shader,
            entry: if native { "ps" } else { "ps_main" }.into(),
            constants: vec![],
        }),
        vertex_buffers: vec![],
        color_targets: vec![Some(ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: 15,
        })],
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: MultisampleState {
            count: 1,
            mask: u64::from(u32::MAX),
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    }))
    .unwrap();
    let color = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap();
    let views = |texture: &Texture<_>, format, aspect| {
        (0..2)
            .map(|layer| {
                d.create_view(
                    texture,
                    &TextureViewDesc {
                        format,
                        dimension: TextureViewDimension::D2Array,
                        aspects: aspect,
                        mips: 0..1,
                        layers: layer..layer + 1,
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let colors = views(&color, TextureFormat::Rgba8Unorm, TextureAspects::COLOR);
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 8208,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ | BufferUsage::QUERY_RESOLVE,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let indices = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 8,
            usage: BufferUsage::INDEX | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let args_size = batch.map_or(52, |batch| {
        batch.offset + u64::from(batch.count) * u64::from(batch.stride)
    });
    let args = d
        .create_buffer(&BufferDesc {
            label: None,
            size: args_size,
            usage: BufferUsage::INDIRECT | BufferUsage::COPY_DST | BufferUsage::STORAGE,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    let idx: [u16; 4] = if negative { [6, 7, 8, 0] } else { [0, 1, 2, 0] };
    let idx_bytes: Vec<u8> = idx.into_iter().flat_map(u16::to_le_bytes).collect();
    e.write_buffer(&indices, 0, &idx_bytes).unwrap();
    let base = if negative { -1i32 } else { 5i32 };
    let words = if indexed {
        [3u32, 1, 0, base as u32, 9, 1]
    } else {
        [3u32, 1, 5, 9, 0, 1]
    };
    let data: Vec<u8> = if let Some(batch) = batch {
        let mut records = vec![0u8; args_size as usize];
        for index in 0..batch.count {
            let start = (batch.offset + u64::from(index) * u64::from(batch.stride)) as usize;
            let mut record = words;
            if index + 1 == batch.count {
                record[if indexed { 4 } else { 3 }] = if batch.gpu_count { 77 } else { 10 };
            }
            let length = if indexed { 5 } else { 4 };
            for (word, value) in record[..length].iter().enumerate() {
                records[start + word * 4..start + word * 4 + 4]
                    .copy_from_slice(&value.to_le_bytes());
            }
        }
        records
    } else {
        words.into_iter().flat_map(u32::to_le_bytes).collect()
    };
    if indirect != 3 {
        e.write_buffer(&args, 0, &data).unwrap();
    }
    let gpu_producer = indirect == 3 || batch.is_some_and(|batch| batch.gpu_count);
    if gpu_producer {
        let gl = d
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: None,
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 52,
                        access: StorageAccess::ReadWrite,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                }],
            })
            .unwrap();
        let group = d
            .create_bind_group(&BindGroupDesc {
                label: None,
                layout: &gl,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::StorageBufferView {
                        buffer: &args,
                        range: 0..52,
                        view: StorageBufferView::Raw,
                    },
                }],
            })
            .unwrap();
        let layout = d
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: None,
                groups: &[&gl],
                push_constants: &[],
            })
            .unwrap();
        let code = if let Some(batch) = batch.filter(|batch| batch.gpu_count) {
            format!("a[{}]=1u;", batch.count_offset / 4)
        } else if indexed {
            "a[0]=3u; a[1]=1u; a[2]=0u; a[3]=4294967295u; a[4]=9u; a[6]=3u; a[7]=1u; a[8]=0u; a[9]=4294967295u; a[10]=10u;".to_owned()
        } else {
            "a[0]=3u; a[1]=1u; a[2]=5u; a[3]=9u; a[5]=3u; a[6]=1u; a[7]=5u; a[8]=10u;".to_owned()
        };
        let count_write = if indirect == 3 { "a[12]=2u;" } else { "" };
        let source = format!(
            "@group(0) @binding(0) var<storage, read_write> a: array<u32>; @compute @workgroup_size(1) fn main() {{ {code} {count_write} }}"
        );
        let shader = block_on(d.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.into_bytes()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: vec![],
        }))
        .unwrap();
        let compute = block_on(d.compute().unwrap().create_pipeline(ComputePipelineDesc {
            label: None,
            layout,
            compute: PipelineStage {
                module: shader,
                entry: "main".into(),
                constants: vec![],
            },
            cache: None,
            required_subgroup_size: None,
        }))
        .unwrap();
        e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &args,
                range: 0..args_size,
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
        })])
        .unwrap();
        let mut pass = d.compute().unwrap().begin(&mut e).unwrap();
        pass.set_pipeline(&compute).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    e.transition_resources(&[
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &indices,
                range: 0..8,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: BufferUses {
                    stages: StageMask::VERTEX_INPUT,
                    accesses: AccessMask::INDEX_READ,
                },
            },
            ownership: None,
        }),
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &args,
                range: 0..args_size,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: if gpu_producer {
                        StageMask::COMPUTE
                    } else {
                        StageMask::EMPTY
                    },
                    accesses: if gpu_producer {
                        AccessMask::SHADER_WRITE
                    } else {
                        AccessMask::EMPTY
                    },
                },
                to: BufferUses {
                    stages: StageMask::INDIRECT,
                    accesses: AccessMask::INDIRECT_READ,
                },
            },
            ownership: None,
        }),
    ])
    .unwrap();

    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let render = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let copy = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let barrier = |from, to, layouts| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &color,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..2,
            },
            usage: StateTransition { from, to },
            layout: Some(layouts),
            ownership: None,
        })
    };
    e.transition_resources(&[barrier(
        empty,
        render,
        (ImageLayout::Common, ImageLayout::ColorAttachment),
    )])
    .unwrap();
    let attachments = [Some(ColorAttachment {
        view: AttachmentView::Texture(&colors[0]),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
        store: StoreOp::Store,
    })];
    let timestamps = batch.map(|_| {
        d.queries(QueryKind::Timestamp)
            .unwrap()
            .create(&QueryDesc {
                kind: QueryKind::Timestamp,
                count: 2,
            })
            .unwrap()
    });
    let mut pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &attachments,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
            multiview_mask: None,
            timestamp_writes: timestamps.as_ref().map(|set| PassTimestampWrites {
                set,
                begin: Some(0),
                end: Some(1),
            }),
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.viewport(Viewport {
        x: 0.0,
        y: 0.0,
        width: 16.0,
        height: 16.0,
        min_depth: 0.0,
        max_depth: 1.0,
    })
    .unwrap();
    pass.scissor(Rect {
        x: 0,
        y: 0,
        width: 16,
        height: 16,
    })
    .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    if indexed {
        pass.set_index_buffer(&indices, 0..8, IndexFormat::Uint16)
            .unwrap();
    }
    let extras = d.raster_extras().unwrap();
    let recording = if indirect == 4 {
        let batch = batch.unwrap();
        let mut record = |pass: &mut RasterPass<'_, _>| {
            if batch.gpu_count {
                extras
                    .multi_draw_count(
                        pass,
                        &args,
                        batch.offset,
                        &args,
                        batch.count_offset,
                        batch.count,
                        batch.stride,
                        indexed,
                    )
                    .unwrap();
            } else {
                extras
                    .multi_draw(
                        pass,
                        &args,
                        batch.offset,
                        batch.count,
                        batch.stride,
                        indexed,
                    )
                    .unwrap();
            }
        };
        let start = Instant::now();
        record(&mut pass);
        let cold = start.elapsed();
        let start = Instant::now();
        record(&mut pass);
        Some([cold, start.elapsed()])
    } else {
        match (indexed, indirect) {
            (false, 0) => pass.draw(5..8, 9..10).unwrap(),
            (true, 0) => pass.draw_indexed(0..3, base, 9..10).unwrap(),
            (_, 1) => extras.draw_indirect(&mut pass, &args, 0, indexed).unwrap(),
            (_, 2) => extras
                .multi_draw_count(
                    &mut pass,
                    &args,
                    0,
                    &args,
                    20,
                    1,
                    if indexed { 20 } else { 16 },
                    indexed,
                )
                .unwrap(),
            (_, 3) => extras
                .multi_draw_count(
                    &mut pass,
                    &args,
                    0,
                    &args,
                    48,
                    2,
                    if indexed { 24 } else { 20 },
                    indexed,
                )
                .unwrap(),
            _ => unreachable!(),
        }
        None
    };

    pass.finish().unwrap();
    if let Some(set) = &timestamps {
        d.queries(QueryKind::Timestamp)
            .unwrap()
            .resolve(&mut e, set, 0..2, &dst, 8192)
            .unwrap();
    }
    e.transition_resources(&[barrier(
        render,
        copy,
        (ImageLayout::ColorAttachment, ImageLayout::CopySrc),
    )])
    .unwrap();
    for layer in 0..2 {
        e.texture_to_buffer(
            TextureRegion {
                texture: &color,
                mip: 0,
                origin: [0, 0, layer],
                extent: [16, 16, 1],
                aspect: TextureAspects::COLOR,
            },
            &dst,
            TexelCopyBufferLayout {
                offset: u64::from(layer) * 4096,
                bytes_per_row: 256,
                rows_per_image: 16,
            },
        )
        .unwrap();
    }
    let ticket = e.readback_buffer(&dst, 0..8208).unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755321,
    };
    // SAFETY: the one-step plan uses this device's graphics lane.
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
    // SAFETY: the work is consumed once on its recording lane.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![e.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        SubmitOutcome::Rejected { error, .. }
        | SubmitOutcome::Indeterminate { error, .. }
        | SubmitOutcome::PartiallyAccepted { error, .. } => panic!("submission failed: {error:?}"),
    };
    block_on(d.wait_completion(&point)).unwrap();
    // SAFETY: the awaited completion point covers the readback.
    let bytes = unsafe { block_on(d.read_after(ticket, &point)).unwrap().bytes };
    if let Some(batch) = batch {
        let first = u64::from_le_bytes(bytes[8192..8200].try_into().unwrap());
        let last = u64::from_le_bytes(bytes[8200..8208].try_into().unwrap());
        assert!(
            last >= first && first != 0,
            "invalid timestamp pair {first}..{last}"
        );
        let frequency = d
            .queries(QueryKind::Timestamp)
            .unwrap()
            .timestamp_frequency(lane)
            .unwrap();
        let gpu_ns = u128::from(last - first) * 1_000_000_000 / u128::from(frequency);
        eprintln!(
            "indirect pass pack+draw count={} indexed={} stride={} offset={} gpu_count={} gpu_ns={gpu_ns}",
            batch.count, indexed, batch.stride, batch.offset, batch.gpu_count
        );
    }
    let center = 8 * 256 + 8 * 4;
    assert_eq!(
        &bytes[center..center + 4],
        if native {
            &[0, 0, 0, 255]
        } else if indirect == 3 || batch.is_some_and(|batch| !batch.gpu_count) {
            &[5, 10, 0, 255]
        } else {
            &[5, 9, 0, 255]
        }
    );
    recording
}
