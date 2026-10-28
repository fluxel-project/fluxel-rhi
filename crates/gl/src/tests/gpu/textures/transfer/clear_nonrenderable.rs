//! Non-renderable texture clears, with mip/layer sentinels.
use super::*;

async fn clear_nonrenderable(d: &Device<Gl>, format: TextureFormat) {
    use crate::backend::api::{GlFamilyApi as _, GlFamilyProfile, GlFormat, GlFormatResourceKind};
    #[cfg(not(target_arch = "wasm32"))]
    {
        // SAFETY: device owns the active context during discovery.
        let native = unsafe { d.native_ref() };
        let available=native.0.with_native("inspect direct texture clear",|p| {let x=p.discovery(); let desktop=matches!(x.context().profile(),GlFamilyProfile::Desktop{major,minor} if major>4 || major==4 && minor>=4); let ext=x.extensions().raw_reported_names().any(|v|v=="GL_ARB_clear_texture" || v=="GL_EXT_clear_texture"); Ok(desktop||ext)}).unwrap();
        eprintln!(
            "EXECUTED non-renderable clear route: {}",
            if available {
                "native clear-texture"
            } else {
                "bounded upload"
            }
        );
        let renderable = native
            .0
            .with_native("inspect RGB9E5 render evidence", |p| {
                Ok(p.discovery()
                    .formats()
                    .get_for(GlFormatResourceKind::Texture, GlFormat::Rgb9e5Ufloat, 1)
                    .is_some_and(|f| f.renderable))
            })
            .unwrap();
        assert!(
            !renderable,
            "fixture requires non-renderable texture to exercise direct clear"
        );
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [2, 2, 1],
            array_layers: 2,
            format,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        };
        assert!(
            !matches!(d.texture_support(&desc), Support::Supported(_)),
            "fixture requires non-renderable format"
        );
    }
    let source=br#"@group(0) @binding(0) var a:texture_2d_array<f32>; @group(0) @binding(1) var s:sampler;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));return vec4f(p[i],0.5,1.);}
@fragment fn fs()->@location(0) vec4f {let a0=textureLoad(a,vec2i(0,0),1,1); return vec4f(a0.r,a0.g,textureLoad(a,vec2i(0,0),0,1).b,textureLoad(a,vec2i(0,0),1,0).b);}"#;
    let source = if format == TextureFormat::Rgba8Snorm {
        String::from_utf8(source.to_vec())
            .unwrap()
            .replace("a0.g,", "a0.g*0.5+0.5,")
    } else {
        String::from_utf8(source.to_vec()).unwrap()
    };
    let shader = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source.as_bytes()),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = d.shader(shader("vs", ShaderStage::Vertex)).await.unwrap();
    let fragment = d.shader(shader("fs", ShaderStage::Fragment)).await.unwrap();
    let entry = BindGroupLayoutEntry {
        binding: 0,
        visibility: vec![ShaderStage::Fragment],
        ty: BindingType::SampledTexture {
            sample: TextureSampleType::Float,
            dimension: TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let textures = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry,
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vec![ShaderStage::Fragment],
                    ty: BindingType::Sampler {
                        filtering: false,
                        comparison: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
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
    let input = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [2, 2, 1],
            array_layers: 2,
            format,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let out = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let iv = d
        .create_view(
            &input,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2Array,
                aspects: TextureAspects::COLOR,
                mips: 0..2,
                layers: 0..2,
            },
        )
        .unwrap();
    let ov = d
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
    let sampler = d
        .create_sampler(&SamplerDesc {
            label: None,
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..2.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &textures,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&iv),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        })
        .unwrap();
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
    let plan = common::single_commands_plan(&fixture, 155).unwrap();
    // SAFETY: resources and context remain alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    e.clear_texture(
        &input,
        0..2,
        0..2,
        TextureClearValue::Color(ClearColor::Float([0., 0., 1., 1.])),
    )
    .unwrap();
    e.clear_texture(
        &input,
        1..2,
        1..2,
        TextureClearValue::Color(ClearColor::Float(if format == TextureFormat::Rgba8Snorm {
            [1., -0.5, 0., 1.]
        } else {
            [1., 0.5, 0., 1.]
        })),
    )
    .unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&ov),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
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
    p.set_pipeline(&pipeline).unwrap();
    p.set_bind_group(0, &group, &[]).unwrap();
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
    // SAFETY: point covers draw and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let pixel = &bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4];
    eprintln!("{format:?} non-renderable clear + sentinels: {pixel:?}");
    assert_eq!([pixel[0], pixel[2], pixel[3]], [255, 255, 255]);
    if format == TextureFormat::Rgba8Snorm {
        assert!((63..=64).contains(&pixel[1]));
    } else {
        assert_eq!(pixel[1], 128);
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_clear_nonrenderable() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(clear_nonrenderable(&d, TextureFormat::Rgb9e5Ufloat));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_clear_nonrenderable() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(clear_nonrenderable(&d, TextureFormat::Rgb9e5Ufloat));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_clear_nonrenderable() {
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
    let d = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    for format in [TextureFormat::Rgb9e5Ufloat, TextureFormat::Rgba8Snorm] {
        clear_nonrenderable(&d, format).await;
    }
}
