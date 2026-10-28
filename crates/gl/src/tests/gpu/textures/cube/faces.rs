//! Six cube faces preserve face selection and vertical orientation.
use super::*;
async fn cube_faces(d: &Device<Gl>) {
    let source = br#"
@group(0) @binding(0) var image: texture_cube<f32>;
@group(0) @binding(1) var nearest: sampler;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));return vec4f(p[i],0.5,1.);
}
@fragment fn fs(@builtin(position) pos:vec4f)->@location(0) vec4f {
 let top=array<vec3f,6>(vec3f(1.,0.5,0.),vec3f(-1.,0.5,0.),vec3f(0.,1.,-0.5),vec3f(0.,-1.,0.5),vec3f(0.,0.5,1.),vec3f(0.,0.5,-1.));
 let bot=array<vec3f,6>(vec3f(1.,-0.5,0.),vec3f(-1.,-0.5,0.),vec3f(0.,1.,0.5),vec3f(0.,-1.,-0.5),vec3f(0.,-0.5,1.),vec3f(0.,-0.5,-1.));
 let i=min(u32(pos.x),5u); let a=textureSampleLevel(image,nearest,top[i],0.);let b=textureSampleLevel(image,nearest,bot[i],0.);return vec4f(a.r,a.g,b.g,1.);
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
    let textures = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry(
                    0,
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::Cube,
                        multisampled: false,
                    },
                ),
                entry(
                    1,
                    BindingType::Sampler {
                        filtering: false,
                        comparison: false,
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
    let descriptor = || RasterPipelineDesc {
        label: None,
        layout: layout.clone(),
        vertex: PipelineStage {
            module: vertex.clone(),
            entry: "vs".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: fragment.clone(),
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
    };
    let pipeline = d.raster_pipeline(descriptor()).await.unwrap();
    let second_pipeline = d.raster_pipeline(descriptor()).await.unwrap();
    let texture = |format, extent, usage| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent,
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage,
            view_formats: vec![],
        })
    };
    let parent = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [2, 2, 1],
            array_layers: 6,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let subview = match d.create_view(
        &parent,
        &TextureViewDesc {
            format: TextureFormat::Rgba8Unorm,
            dimension: TextureViewDimension::Cube,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..6,
        },
    ) {
        Ok(v) => v,
        Err(Error::Unsupported(reason)) => {
            eprintln!("SKIP cube view: {reason}");
            return;
        }
        Err(e) => panic!("cube view: {e:?}"),
    };
    let out = texture(
        TextureFormat::Rgba8Unorm,
        [8, 8, 1],
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    )
    .unwrap();
    let view = |t, format| {
        d.create_view(
            t,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap()
    };
    let out_v = view(&out, TextureFormat::Rgba8Unorm);
    let sampler = d
        .create_sampler(&SamplerDesc {
            label: None,
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })
        .unwrap();
    let group = |v| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &textures,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(v),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        })
        .unwrap()
    };
    let sub_g = group(&subview);
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
    let plan = common::single_commands_plan(&fixture, 230).unwrap();
    // SAFETY: resources and context survive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    for face in 0..6u32 {
        let r = (face as u8 + 1) * 32;
        let pixels = [r, 64, 0, 255, r, 64, 0, 255, r, 192, 0, 255, r, 192, 0, 255];
        e.write_texture(
            TextureRegion {
                texture: &parent,
                mip: 0,
                origin: [0, 0, face],
                extent: [2, 2, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 8,
                rows_per_image: 2,
            },
            &pixels,
        )
        .unwrap();
    }
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish setup"),
    };
    d.wait_completion(&point).await.unwrap();
    drop(parent);
    let plan = common::single_commands_plan(&fixture, 231).unwrap();
    // SAFETY: the view, group and context outlive the subsequent completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin draw"));
    let e = run.commands();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&out_v),
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
    p.set_pipeline(&pipeline).unwrap();
    for (x, g) in [(0, &sub_g), (4, &sub_g)] {
        p.set_bind_group(0, g, &[]).unwrap();
        p.set_pipeline(&second_pipeline).unwrap();
        p.scissor(Rect {
            x,
            y: 0,
            width: 4,
            height: 8,
        })
        .unwrap();
        p.draw(0..3, 0..1).unwrap();
    }
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
    // SAFETY: completion covers texture readback.
    let data = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for x in 0..6usize {
        let actual = &data[4 * 32 + x * 4..4 * 32 + x * 4 + 4];
        let expected = [((x + 1) * 32) as u8, 64, 192, 255];
        assert_eq!(actual, &expected, "cube face {x}");
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_cube_faces() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(cube_faces(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_cube_faces() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    use crate::backend::api::GlFamilyApi as _;
    // SAFETY: the fixture owns the current context for this inspection.
    let native = unsafe { d.native_ref() };
    let available = native
        .0
        .with_native("inspect GLES texture view extensions", |provider| {
            let discovery = provider.discovery();
            let names: Vec<_> = discovery
                .extensions()
                .raw_reported_names()
                .filter(|name| *name == "GL_OES_texture_view" || *name == "GL_EXT_texture_view")
                .map(str::to_owned)
                .collect();
            eprintln!("GLES texture view extensions: {names:?}");
            Ok(!names.is_empty())
        })
        .unwrap();
    if !available {
        eprintln!("SKIP native sampled views: no GLES texture-view extension");
        return;
    }
    block_on(cube_faces(&d));
}
