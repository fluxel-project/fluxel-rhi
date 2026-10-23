//! Core cube view face selection and upper-left orientation.
use super::*;

async fn binding_cube_mirror(d: &Device<Gl>, integer: bool, signed: bool) {
    let source = br#"
@group(0) @binding(0) var a: texture_cube<f32>;
@group(0) @binding(1) var s: sampler;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}
@fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f {
 let dir=select(vec3f(1.,0.,0.),vec3f(1.,0.999,0.),p.x>=6.);
 let lod=select(0.,1.,p.y>=4.);
 return textureSampleLevel(a,s,dir,lod);
}
"#;
    let shader = |name: &str, stage| {
        let version = if cfg!(target_arch = "wasm32") {
            300
        } else {
            310
        };
        let integer_source = if stage == ShaderStage::Vertex {
            format!(
                "#version {version} es\nprecision highp float; void main(){{vec2 p[3]=vec2[3](vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));gl_Position=vec4(p[gl_VertexID],0.5,1.);}}"
            )
        } else {
            let sampler = if signed {
                "isamplerCube"
            } else {
                "usamplerCube"
            };
            let scale = if signed { "127." } else { "255." };
            format!(
                "#version {version} es\nprecision highp float; precision highp int; uniform highp {sampler} binding_0; layout(location=0) out vec4 result; void main(){{vec3 dirs[6]=vec3[6](vec3(1,0,0),vec3(-1,0,0),vec3(0,1,0),vec3(0,-1,0),vec3(0,0,1),vec3(0,0,-1)); int i=int(gl_FragCoord.x)%6; float lod=gl_FragCoord.y>=4.?1.:0.; result=vec4(textureLod(binding_0,dirs[i],lod))/{scale};}}"
            )
        };
        ShaderDesc {
            label: None,
            format: if integer {
                ShaderFormat::GlslEs(if cfg!(target_arch = "wasm32") {
                    300
                } else {
                    310
                })
            } else {
                ShaderFormat::Wgsl
            },
            bytes: if integer {
                Arc::from(integer_source.into_bytes())
            } else {
                Arc::from(source.as_slice())
            },
            entries: vec![ShaderEntryPoint {
                name: if integer { "main".into() } else { name.into() },
                stage,
                workgroup_size: None,
            }],
            required_features: vec![],
        }
    };
    let vertex = d.shader(shader("vs", ShaderStage::Vertex)).await.unwrap();
    let fragment = d.shader(shader("fs", ShaderStage::Fragment)).await.unwrap();
    let entry = BindGroupLayoutEntry {
        binding: 0,
        visibility: vec![ShaderStage::Fragment],
        ty: BindingType::SampledTexture {
            sample: if integer {
                if signed {
                    TextureSampleType::Sint
                } else {
                    TextureSampleType::Uint
                }
            } else {
                TextureSampleType::Float
            },
            dimension: TextureViewDimension::Cube,
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
                        filtering: !integer,
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
                entry: if integer { "main".into() } else { "vs".into() },
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment,
                entry: if integer { "main".into() } else { "fs".into() },
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
                mask: u64::MAX,
                alpha_to_coverage_enabled: false,
            },
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
            array_layers: 12,
            format: if integer {
                if signed {
                    TextureFormat::Rgba8Sint
                } else {
                    TextureFormat::Rgba8Uint
                }
            } else {
                TextureFormat::Rgba8Unorm
            },
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING
                | TextureUsage::COPY_DST
                | TextureUsage::RENDER_ATTACHMENT,
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
                format: if integer {
                    if signed {
                        TextureFormat::Rgba8Sint
                    } else {
                        TextureFormat::Rgba8Uint
                    }
                } else {
                    TextureFormat::Rgba8Unorm
                },
                dimension: TextureViewDimension::Cube,
                aspects: TextureAspects::COLOR,
                mips: 0..2,
                layers: 0..6,
            },
        )
        .unwrap();
    let iv2 = d
        .create_view(
            &input,
            &TextureViewDesc {
                format: if integer {
                    if signed {
                        TextureFormat::Rgba8Sint
                    } else {
                        TextureFormat::Rgba8Uint
                    }
                } else {
                    TextureFormat::Rgba8Unorm
                },
                dimension: TextureViewDimension::Cube,
                aspects: TextureAspects::COLOR,
                mips: 0..2,
                layers: 6..12,
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
            mag_filter: if integer {
                FilterMode::Nearest
            } else {
                FilterMode::Linear
            },
            min_filter: if integer {
                FilterMode::Nearest
            } else {
                FilterMode::Linear
            },
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
    let group2 = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &textures,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&iv2),
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
    let mut data = vec![0u8; 2 * 2 * 12 * 4];
    for face in 0..12 {
        for y in 0..2 {
            for x in 0..2 {
                let i = face * 16 + y * 8 + x * 4;
                let color = if integer {
                    let face_red = if signed { 15 } else { 30 };
                    let view_red = if signed { 10 } else { 20 };
                    [
                        16 + ((face % 6) as u8) * face_red + if face >= 6 { view_red } else { 0 },
                        0,
                        0,
                        if signed { 127 } else { 255 },
                    ]
                } else if face == 2 {
                    [0, 255, 0, 255]
                } else if face < 6 {
                    [255, 0, 0, 255]
                } else {
                    [0, 0, 255, 255]
                };
                data[i..i + 4].copy_from_slice(&color);
            }
        }
    }
    e.write_texture(
        TextureRegion {
            texture: &input,
            mip: 0,
            origin: [0, 0, 0],
            extent: [2, 2, 12],
            aspect: TextureAspects::COLOR,
        },
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 8,
            rows_per_image: 2,
        },
        &data,
    )
    .unwrap();
    let mut mip = vec![0u8; 12 * 4];
    for face in 0..12 {
        let color = if integer {
            let face_red = if signed { 8 } else { 10 };
            let view_red = if signed { 10 } else { 20 };
            [
                if signed { 30 } else { 128 }
                    + ((face % 6) as u8) * face_red
                    + if face >= 6 { view_red } else { 0 },
                0,
                0,
                if signed { 127 } else { 255 },
            ]
        } else if face < 6 {
            [0, 255, 0, 255]
        } else {
            [255, 255, 0, 255]
        };
        mip[face * 4..face * 4 + 4].copy_from_slice(&color);
    }
    e.write_texture(
        TextureRegion {
            texture: &input,
            mip: 1,
            origin: [0, 0, 0],
            extent: [1, 1, 12],
            aspect: TextureAspects::COLOR,
        },
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 4,
            rows_per_image: 1,
        },
        &mip,
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
    let updated = if signed {
        [0u8, 0, 63, 127].repeat(4)
    } else {
        [0u8, 0, 255, 255].repeat(4)
    };
    e.write_texture(
        TextureRegion {
            texture: &input,
            mip: 0,
            origin: [0, 0, 0],
            extent: [2, 2, 1],
            aspect: TextureAspects::COLOR,
        },
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 8,
            rows_per_image: 2,
        },
        &updated,
    )
    .unwrap();
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
    p.scissor(Rect {
        x: 0,
        y: 0,
        width: 2,
        height: 8,
    })
    .unwrap();
    p.draw(0..3, 0..1).unwrap();
    p.set_bind_group(0, &group2, &[]).unwrap();
    p.scissor(Rect {
        x: 2,
        y: 0,
        width: 3,
        height: 8,
    })
    .unwrap();
    p.draw(0..3, 0..1).unwrap();
    p.set_bind_group(0, &group, &[]).unwrap();
    p.scissor(Rect {
        x: 5,
        y: 0,
        width: 3,
        height: 8,
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
    // SAFETY: point covers draw and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    if integer {
        for (x, row) in (1..=6).map(|x| (x, 1)).chain((1..=6).map(|x| (x, 6))) {
            let face = (x % 6) as u8;
            let cube_offset = if (2..=4).contains(&x) { 20 } else { 0 };
            let raw_red = if row == 1 {
                16 + face * if signed { 15 } else { 30 } + cube_offset / if signed { 2 } else { 1 }
            } else {
                (if signed { 30 } else { 128 })
                    + face * if signed { 8 } else { 10 }
                    + cube_offset / if signed { 2 } else { 1 }
            };
            let red = if signed {
                ((u16::from(raw_red) * 255) / 127) as u8
            } else {
                raw_red
            };
            let expected = if row == 1 && x == 6 {
                [0, 0, if signed { 127 } else { 255 }, 255]
            } else {
                [red, 0, 0, 255]
            };
            let i = row * 32 + x * 4;
            eprintln!("integer cube mirror {x},{row}: {:?}", &bytes[i..i + 4]);
            if signed {
                for (actual, expected) in bytes[i..i + 4].iter().zip(expected) {
                    assert!(
                        actual.abs_diff(expected) <= 1,
                        "signed cube mirror {x},{row}: actual={actual}, expected={expected}"
                    );
                }
            } else {
                assert_eq!(&bytes[i..i + 4], &expected);
            }
        }
        return;
    }
    for (x, row, expected) in [
        (1, 1, [0, 0, 255, 255]),
        (3, 1, [0, 0, 255, 255]),
        (1, 6, [0, 255, 0, 255]),
        (3, 6, [255, 255, 0, 255]),
        (6, 6, [0, 255, 0, 255]),
    ] {
        let i = row * 32 + x * 4;
        eprintln!("cube mirror {x},{row}: {:?}", &bytes[i..i + 4]);
        assert_eq!(&bytes[i..i + 4], &expected);
    }
    if integer {
        return;
    }
    let seam = &bytes[32 + 6 * 4..32 + 6 * 4 + 4];
    eprintln!("cube mirror colored seam {seam:?}");
    assert!(
        seam[0] == 0
            && (119..=136).contains(&seam[1])
            && (119..=136).contains(&seam[2])
            && seam[3] == 255
    );
    // A source layer cannot simultaneously be the pass attachment while its
    // cube mirror is sampled. Use a fresh encoder: the rejected draw poisons it.
    let attachment_view = d
        .create_view(
            &input,
            &TextureViewDesc {
                format: if integer {
                    TextureFormat::Rgba8Uint
                } else {
                    TextureFormat::Rgba8Unorm
                },
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let alias_colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&attachment_view),
        resolve_target: None,
        load: LoadOp::Load,
        store: StoreOp::Store,
    })];
    let plan = common::single_commands_plan(&fixture, 156).unwrap();
    // SAFETY: the context and all borrowed views outlive this immediate scope.
    let mut negative =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut negative_run = negative
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin negative commands"));
    let mut pass = negative_run
        .commands()
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &alias_colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    let rejected = pass.draw(0..3, 0..1);
    assert!(
        matches!(
            rejected,
            Err(Error::Unsupported("GL cube mirror concurrent source write"))
        ),
        "same source attachment must be rejected: {rejected:?}"
    );
}
#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_integer_cube_mirror_sampling() {
    use std::{
        future::Future,
        task::{Context, Poll, Wake, Waker},
    };
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    fn block_on<F: Future>(future: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut context = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::park(),
            }
        }
    }
    let native = GlInstance::gles_pbuffer([8, 8]).expect("open GLES pbuffer");
    let device = block_on(open(native));
    // Exercise the array-to-cube mirror fallback even on drivers that expose
    // native texture views. The WebGL2 fixture always takes this path.
    let native = unsafe { device.native_ref() };
    native
        .0
        .with_native("force cube mirror fallback fixture", |provider| {
            provider.disable_texture_views_for_cube_mirror_test();
            Ok(())
        })
        .unwrap();
    block_on(binding_cube_mirror(&device, true, false));
    block_on(binding_cube_mirror(&device, true, true));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_binding_cube_mirror() {
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
    binding_cube_mirror(&d, false, false).await;
    binding_cube_mirror(&d, true, false).await;
    binding_cube_mirror(&d, true, true).await;
}
