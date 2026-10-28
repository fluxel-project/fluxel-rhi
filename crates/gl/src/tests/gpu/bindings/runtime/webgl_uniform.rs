//! WebGL2 bounded runtime uniform-buffer arrays through ordinary UBO points.
//!
//! WebGL2 exposes neither SSBOs nor the combined nonuniform Core feature.
//! This fixture deliberately indexes a runtime UBO array with a separate
//! uniform selector, which ESSL 3.00 may lower to ordinary fixed UBO slots.

use super::*;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
async fn runtime_webgl_uniform_arrays(d: &Device<Gl>) {
    let source = br#"
struct U { color: vec4u }
struct Pick { index: u32 }
@group(0) @binding(0) var<uniform> values: binding_array<U>;
@group(1) @binding(0) var<uniform> pick: Pick;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let points = array<vec2f, 3>(vec2f(-1., -1.), vec2f(3., -1.), vec2f(-1., 3.));
  return vec4f(points[i], 0.5, 1.);
}
@fragment fn fs() -> @location(0) vec4f {
  return vec4f(values[pick.index].color) / 255.0;
}
"#;
    let shader = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source.as_slice()),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = d.shader(shader("vs", ShaderStage::Vertex)).await.unwrap();
    let fragment = d.shader(shader("fs", ShaderStage::Fragment)).await.unwrap();
    let values_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 16,
                },
                count: BindingCount::RuntimeSized { max_count: 3 },
                partially_bound: true,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let pick_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                // Uniform-space structs retain a 16-byte struct alignment.
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 16,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&values_layout, &pick_layout],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: vertex,
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment,
                entry: "fs".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![],
            color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
            primitive: triangle_list_state(),
            depth_stencil: None,
            multisample: single_sample_state(u64::MAX),
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
    let uniform = || {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let red = uniform();
    let blue = uniform();
    let pick_two = uniform();
    let pick_zero = uniform();
    let value_group = |elements| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &values_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Array(elements),
            }],
        })
        .unwrap()
    };
    let with_hole = value_group(vec![
        Some(BindingResource::Buffer {
            buffer: &red,
            range: 0..16,
        }),
        None,
        Some(BindingResource::Buffer {
            buffer: &blue,
            range: 0..16,
        }),
    ]);
    let short = value_group(vec![Some(BindingResource::Buffer {
        buffer: &blue,
        range: 0..16,
    })]);
    let pick_group = |buffer| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &pick_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer,
                    range: 0..16,
                },
            }],
        })
        .unwrap()
    };
    let pick_two_group = pick_group(&pick_two);
    let pick_zero_group = pick_group(&pick_zero);
    let out = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [6, 1, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let out_view = d
        .create_view(
            &out,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let readback = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 151).unwrap();
    // SAFETY: the adopted context and every resource live through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = match session.begin_commands() {
        Ok(run) => run,
        Err(_) => panic!("begin commands"),
    };
    let commands = run.commands();
    let uniform_words = |words: [u32; 4]| {
        let mut bytes = [0; 16];
        for (slot, word) in bytes.chunks_exact_mut(4).zip(words) {
            slot.copy_from_slice(&word.to_ne_bytes());
        }
        bytes
    };
    commands
        .write_buffer(&red, 0, &uniform_words([255, 0, 0, 255]))
        .unwrap();
    commands
        .write_buffer(&blue, 0, &uniform_words([0, 0, 255, 255]))
        .unwrap();
    commands
        .write_buffer(
            &pick_two,
            0,
            &[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        .unwrap();
    commands.write_buffer(&pick_zero, 0, &[0; 16]).unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&out_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
        store: StoreOp::Store,
    })];
    let mut pass = commands
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 6,
                height: 1,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    for (x, values, pick) in [
        (0, &with_hole, &pick_two_group),
        (2, &with_hole, &pick_zero_group),
        (4, &short, &pick_zero_group),
    ] {
        pass.set_bind_group(0, values, &[]).unwrap();
        pass.set_bind_group(1, pick, &[]).unwrap();
        pass.scissor(Rect {
            x,
            y: 0,
            width: 2,
            height: 1,
        })
        .unwrap();
        pass.draw(0..3, 0..1).unwrap();
    }
    pass.finish().unwrap();
    commands
        .texture_to_buffer(
            TextureRegion {
                texture: &out,
                mip: 0,
                origin: [0, 0, 0],
                extent: [6, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            &readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 24,
                rows_per_image: 1,
            },
        )
        .unwrap();
    let ticket = commands.readback_buffer(&readback, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: the completion point covers the output copy.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for (x, expected) in [
        (0, [0, 0, 255, 255]),
        (2, [255, 0, 0, 255]),
        (4, [0, 0, 255, 255]),
    ] {
        assert_eq!(&bytes[x * 4..x * 4 + 4], expected);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_binding_runtime_uniform_arrays() {
    let device = webgl2_device([6, 1]).await;
    assert!(device.capabilities().runtime_sized_binding_arrays);
    assert!(
        !device
            .capabilities()
            .nonuniform_sampled_texture_and_storage_buffer_indexing
    );
    runtime_webgl_uniform_arrays(&device).await;
}
