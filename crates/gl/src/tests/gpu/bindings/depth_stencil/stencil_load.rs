//! Core comparison samplers and signed/unsigned texture readback.
use super::*;

async fn binding_stencil_load(d: &Device<Gl>, required: bool, expect_conflict: bool) {
    let source = br#"
@group(0) @binding(0) var depth: texture_2d<u32>;
@group(0) @binding(1) var cmp: sampler_comparison;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}
@fragment fn fs()->@location(0) vec4f {
 let value=textureLoad(depth,vec2i(0,0),0).x;
 return vec4f(f32(value)/255.0,0.,0.,1.);
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
    let entry = |binding, ty| BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Fragment],
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let texture_type = |sample| BindingType::SampledTexture {
        sample,
        dimension: TextureViewDimension::D2,
        multisampled: false,
    };
    let textures = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry(0, texture_type(TextureSampleType::Uint)),
                entry(
                    1,
                    BindingType::Sampler {
                        filtering: false,
                        comparison: true,
                    },
                ),
            ],
        })
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&textures],
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
    let depth_source = br#"
@group(0) @binding(0) var depth: texture_depth_2d;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}
@fragment fn fs()->@location(0) vec4f { return vec4f(textureLoad(depth,vec2i(0,0),0),0.,0.,1.); }
"#;
    let depth_module = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(depth_source.as_slice()),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let depth_vertex = d
        .shader(depth_module("vs", ShaderStage::Vertex))
        .await
        .unwrap();
    let depth_fragment = d
        .shader(depth_module("fs", ShaderStage::Fragment))
        .await
        .unwrap();
    let depth_textures = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![entry(0, texture_type(TextureSampleType::Depth))],
        })
        .unwrap();
    let depth_layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&depth_textures],
            push_constants: &[],
        })
        .unwrap();
    let depth_pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout: depth_layout,
            vertex: PipelineStage {
                module: depth_vertex,
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: depth_fragment,
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
    let dual_source = br#"
@group(0) @binding(0) var depth: texture_depth_2d;
@group(0) @binding(1) var stencil: texture_2d<u32>;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}
@fragment fn fs()->@location(0) vec4f { return vec4f(textureLoad(depth,vec2i(0,0),0),f32(textureLoad(stencil,vec2i(0,0),0).x)/255.,0.,1.); }
"#;
    let dual_module = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(dual_source.as_slice()),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let dual_vertex = d
        .shader(dual_module("vs", ShaderStage::Vertex))
        .await
        .unwrap();
    let dual_fragment = d
        .shader(dual_module("fs", ShaderStage::Fragment))
        .await
        .unwrap();
    let dual_textures = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry(0, texture_type(TextureSampleType::Depth)),
                entry(1, texture_type(TextureSampleType::Uint)),
            ],
        })
        .unwrap();
    let dual_layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&dual_textures],
            push_constants: &[],
        })
        .unwrap();
    let dual_pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout: dual_layout,
            vertex: PipelineStage {
                module: dual_vertex,
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: dual_fragment,
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
    let texture = |format, usage| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage,
            view_formats: vec![],
        })
        .unwrap_or_else(|error| panic!("create {format:?}: {error:?}"))
    };
    let depth = texture(
        TextureFormat::Depth24PlusStencil8,
        TextureUsage::TEXTURE_BINDING | TextureUsage::RENDER_ATTACHMENT,
    );
    let other_depth = texture(
        TextureFormat::Depth24PlusStencil8,
        TextureUsage::TEXTURE_BINDING | TextureUsage::RENDER_ATTACHMENT,
    );
    let out = texture(
        TextureFormat::Rgba8Unorm,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );
    let view = |t, format, aspects| {
        d.create_view(
            t,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap()
    };
    let dv = view(
        &depth,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::STENCIL,
    );
    let full = view(
        &depth,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::DEPTH | TextureAspects::STENCIL,
    );
    let other_stencil = view(
        &other_depth,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::STENCIL,
    );
    let other_full = view(
        &other_depth,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::DEPTH | TextureAspects::STENCIL,
    );
    let depth_view = view(
        &depth,
        TextureFormat::Depth24PlusStencil8,
        TextureAspects::DEPTH,
    );
    let depth_group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &depth_textures,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&depth_view),
            }],
        })
        .unwrap();
    let dual_group = d.create_bind_group(&BindGroupDesc {
        label: None,
        layout: &dual_textures,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&depth_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&dv),
            },
        ],
    });
    let separate_group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &dual_textures,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&depth_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&other_stencil),
                },
            ],
        })
        .unwrap();
    let ov = view(&out, TextureFormat::Rgba8Unorm, TextureAspects::COLOR);
    let sampler = d
        .create_sampler(&SamplerDesc {
            label: None,
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: Some(CompareFunction::Less),
            border: None,
        })
        .unwrap();
    let group = match d.create_bind_group(&BindGroupDesc {
        label: None,
        layout: &textures,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&dv),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&sampler),
            },
        ],
    }) {
        Ok(group) => group,
        Err(Error::InvalidState | Error::Unsupported(_)) => {
            assert!(
                !required,
                "stencil sampling expected on the WGL texture-view device"
            );
            eprintln!(
                "SKIP stencil sampling: backend does not provide an independent stencil sampling view"
            );
            return;
        }
        Err(error) => panic!("stencil bind group: {error:?}"),
    };
    let buffer = d
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
    let plan = common::single_commands_plan(&fixture, 153).unwrap();
    // SAFETY: resources and context remain alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    let pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&full),
                depth: Some((LoadOp::Clear(0.25), StoreOp::Store)),
                stencil: Some((LoadOp::Clear(7), StoreOp::Store)),
                depth_read_only: false,
                stencil_read_only: false,
            }),
            render_area: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.finish().unwrap();
    let pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&other_full),
                depth: None,
                stencil: Some((LoadOp::Clear(7), StoreOp::Store)),
                depth_read_only: false,
                stencil_read_only: false,
            }),
            render_area: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.finish().unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&ov),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 0.])),
        store: StoreOp::Store,
    })];
    let mut p = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    for (x, width, selected_pipeline, selected_group) in [
        (0, 3, &pipeline, &group),
        (3, 3, &depth_pipeline, &depth_group),
        (6, 2, &pipeline, &group),
    ] {
        p.set_pipeline(selected_pipeline).unwrap();
        p.set_bind_group(0, selected_group, &[]).unwrap();
        p.scissor(Rect {
            x,
            y: 0,
            width,
            height: 8,
        })
        .unwrap();
        p.draw(0..3, 0..1).unwrap();
    }
    p.set_pipeline(&dual_pipeline).unwrap();
    p.set_bind_group(0, &separate_group, &[]).unwrap();
    p.scissor(Rect {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
    })
    .unwrap();
    p.draw(0..3, 0..1).unwrap();
    p.finish().unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &out,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 1],
            aspect: TextureAspects::COLOR,
        },
        &buffer,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 32,
            rows_per_image: 8,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&buffer, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: point covers all commands including readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for (x, expected) in [
        (1, [7, 0, 0, 255]),
        (4, [64, 0, 0, 255]),
        (7, [7, 0, 0, 255]),
    ] {
        let pixel = &bytes[4 * 32 + x * 4..4 * 32 + x * 4 + 4];
        eprintln!("stencil-depth-stencil rebind pixel {x}: {pixel:?}");
        assert_eq!(pixel, &expected);
    }
    assert_eq!(&bytes[0..4], &[64, 7, 0, 255]);
    let plan = common::single_commands_plan(&fixture, 154).unwrap();
    // SAFETY: all resources and the context are retained throughout the immediate session.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin conflict commands"));
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&ov),
        resolve_target: None,
        load: LoadOp::Load,
        store: StoreOp::Store,
    })];
    let mut p = run
        .commands()
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    p.set_pipeline(&dual_pipeline).unwrap();
    p.set_bind_group(0, &dual_group.unwrap(), &[]).unwrap();
    let simultaneous = p.draw(0..3, 0..1);
    if expect_conflict {
        assert!(matches!(simultaneous, Err(Error::Unsupported(_))));
    } else {
        simultaneous.unwrap();
        p.finish().unwrap();
        let e = run.commands();
        e.texture_to_buffer(
            TextureRegion {
                texture: &out,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            &buffer,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
        let ticket = e.readback_buffer(&buffer, 0..256).unwrap();
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish simultaneous sample"),
        };
        d.wait_completion(&point).await.unwrap();
        // SAFETY: this completion covers the readback.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        assert_eq!(&bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4], &[64, 7, 0, 255]);
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_binding_stencil_load() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(binding_stencil_load(&d, true, false));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_binding_stencil_load() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(binding_stencil_load(&d, true, false));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_binding_stencil_load() {
    use wasm_bindgen::JsCast;
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(8);
    canvas.set_height(8);
    let gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::WebGl2RenderingContext>()
        .unwrap();
    let available = gl
        .get_supported_extensions()
        .map(|items| {
            items
                .iter()
                .any(|item| item.as_string().as_deref() == Some("WEBGL_stencil_texturing"))
        })
        .unwrap_or(false);
    eprintln!("WEBGL_stencil_texturing reported: {available}");
    let d = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    binding_stencil_load(&d, available, false).await;
}
