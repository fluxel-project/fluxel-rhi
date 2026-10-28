//! Private-SPI mixed dynamic and static storage-image array loads and stores.
use super::*;
async fn binding_dynamic_image_query(d: &Device<Gl>) {
    let Some(compute) = d.compute() else {
        eprintln!("SKIP: compute unavailable");
        return;
    };
    use crate::backend::api::{GlFamilyApi as _, GlFamilyProfile};
    // SAFETY: fixture owns this current context.
    let native = unsafe { d.native_ref() };
    let partial = native
        .0
        .with_native("inspect image views", |provider| {
            let d = provider.discovery();
            Ok(
                matches!(d.context().profile(), GlFamilyProfile::Desktop { .. })
                    || d.extensions()
                        .raw_reported_names()
                        .any(|v| v == "GL_OES_texture_view" || v == "GL_EXT_texture_view"),
            )
        })
        .unwrap();
    if !partial {
        eprintln!("SKIP partial array alias: no GLES texture view; testing all-layer mip view");
    }
    let compute_src=br#"@group(0) @binding(0) var img:binding_array<texture_storage_2d_array<r32float,read_write>,2>; @group(0) @binding(1) var out_img:texture_storage_2d_array<r32float,write>; @compute @workgroup_size(1) fn cs(@builtin(global_invocation_id) id:vec3u){ let index=id.x+1u; let dims=textureDimensions(img[index]); let layers=textureNumLayers(img[index]); let missing=textureDimensions(img[index+2u]); textureStore(img[index],vec2i(0,0),0,vec4f(0.75,0.,0.,1.)); storageBarrier(); let a=textureLoad(img[0],vec2i(0,0),0); let b=textureLoad(img[index],vec2i(0,0),0); textureStore(out_img,vec2i(0,0),0,vec4f(select(0.,a.r+b.r*0.5,all(dims==vec2u(4u,4u)) && layers>=2u && all(missing==vec2u(0u))),0.,0.,1.)); }"#;
    let raster_src=br#"@group(0) @binding(0) var img:texture_2d_array<f32>; struct O{@builtin(position) p:vec4f,@location(0) c:vec4f}; @vertex fn vs(@builtin(vertex_index) i:u32)->O{let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));var o:O;o.p=vec4f(p[i],0.5,1.);o.c=vec4f(textureLoad(img,vec2i(0,0),1,1).r,0.,0.,1.);return o;} @fragment fn fs(i:O)->@location(0) vec4f{return i.c;}"#;
    let raster_src = if partial {
        String::from_utf8(raster_src.to_vec()).unwrap()
    } else {
        String::from_utf8(raster_src.to_vec())
            .unwrap()
            .replace(
                "textureLoad(img,vec2i(0,0),1,1)",
                "textureLoad(img,vec2i(0,0),0,1)",
            )
            .replace(
                "textureLoad(img,vec2i(0,0),2,1)",
                "textureLoad(img,vec2i(0,0),1,1)",
            )
    };
    let shader = |src: &[u8], name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(src),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: (stage == ShaderStage::Compute).then_some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let cs = d
        .shader(shader(compute_src, "cs", ShaderStage::Compute))
        .await
        .unwrap();
    let vertex = d
        .shader(shader(raster_src.as_bytes(), "vs", ShaderStage::Vertex))
        .await
        .unwrap();
    let fragment = d
        .shader(shader(raster_src.as_bytes(), "fs", ShaderStage::Fragment))
        .await
        .unwrap();
    let entry = |stage, ty| BindGroupLayoutEntry {
        binding: 0,
        visibility: vec![stage],
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let write_desc = BindGroupLayoutDesc {
        label: None,
        entries: vec![entry(
            ShaderStage::Compute,
            BindingType::StorageTexture {
                format: TextureFormat::R32Float,
                dimension: TextureViewDimension::D2Array,
                access: StorageAccess::ReadWrite,
            },
        )],
    };
    let mut write_desc = write_desc;
    write_desc.entries[0].count = BindingCount::Fixed(2);
    let mut output_entry = write_desc.entries[0].clone();
    output_entry.binding = 1;
    output_entry.count = BindingCount::One;
    output_entry.ty = BindingType::StorageTexture {
        format: TextureFormat::R32Float,
        dimension: TextureViewDimension::D2Array,
        access: StorageAccess::Write,
    };
    write_desc.entries.push(output_entry);
    // SAFETY: the SPI creates the private fixture layout on this same live device.
    let write_layout = unsafe {
        BindGroupLayout::from_native(
            d.id(),
            Gl::create_bind_group_layout(d.native_ref(), &write_desc).unwrap(),
            Arc::from(write_desc.entries.clone()),
        )
    };
    let read_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![entry(
                ShaderStage::Vertex,
                BindingType::SampledTexture {
                    sample: TextureSampleType::UnfilterableFloat,
                    dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
            )],
        })
        .unwrap();
    let cl = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&write_layout],
            push_constants: &[],
        })
        .unwrap();
    let cp = compute
        .create_pipeline(ComputePipelineDesc {
            label: None,
            layout: cl,
            compute: PipelineStage {
                module: cs,
                entry: "cs".into(),
                constants: vec![],
            },
            required_subgroup_size: None,
            cache: None,
        })
        .await
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&read_layout],
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

    let image = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 3,
            format: TextureFormat::R32Float,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::STORAGE_BINDING
                | TextureUsage::TEXTURE_BINDING
                | TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let other = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 3,
            format: TextureFormat::R32Float,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::STORAGE_BINDING | TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let input = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 3,
            format: TextureFormat::R32Float,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::STORAGE_BINDING | TextureUsage::RENDER_ATTACHMENT,
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
    let view = |t, dim, mips, layers, format| {
        d.create_view(
            t,
            &TextureViewDesc {
                format,
                dimension: dim,
                aspects: TextureAspects::COLOR,
                mips,
                layers,
            },
        )
        .unwrap()
    };
    let write_view = view(
        &image,
        TextureViewDimension::D2Array,
        1..2,
        if partial { 1..3 } else { 0..3 },
        TextureFormat::R32Float,
    );
    let other_view = view(
        &other,
        TextureViewDimension::D2Array,
        1..2,
        if partial { 1..3 } else { 0..3 },
        TextureFormat::R32Float,
    );
    let input_view = view(
        &input,
        TextureViewDimension::D2Array,
        1..2,
        if partial { 1..3 } else { 0..3 },
        TextureFormat::R32Float,
    );
    let read_view = view(
        &image,
        TextureViewDimension::D2Array,
        0..2,
        0..3,
        TextureFormat::R32Float,
    );
    let ov = view(
        &out,
        TextureViewDimension::D2,
        0..1,
        0..1,
        TextureFormat::Rgba8Unorm,
    );
    let group = |layout: &BindGroupLayout<Gl>, v| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(v),
            }],
        })
        .unwrap()
    };
    let write = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &write_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Array(vec![
                        Some(BindingResource::TextureView(&input_view)),
                        Some(BindingResource::TextureView(&other_view)),
                    ]),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&write_view),
                },
            ],
        })
        .unwrap();
    let read = group(&read_layout, &read_view);
    let make_buffer = || {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let drawbuf = make_buffer();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 157).unwrap();
    // SAFETY: context and resources stay alive until completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    let transition = |from_layout, to_layout, from_stages, from_access, to_stages, to_access| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &image,
                aspects: TextureAspects::COLOR,
                mips: 1..2,
                layers: if partial { 1..3 } else { 0..3 },
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: from_stages,
                    accesses: from_access,
                },
                to: TextureUses {
                    stages: to_stages,
                    accesses: to_access,
                },
            },
            layout: Some((from_layout, to_layout)),
            ownership: None,
        })
    };
    let layers = if partial { 1..3 } else { 0..3 };
    for (texture, value) in [(&input, 0.25), (&other, 0.5)] {
        e.clear_texture(
            texture,
            1..2,
            layers.clone(),
            TextureClearValue::Color(ClearColor::Float([value, 0., 0., 1.])),
        )
        .unwrap();
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture,
                aspects: TextureAspects::COLOR,
                mips: 1..2,
                layers: layers.clone(),
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COLOR_OUTPUT,
                    accesses: AccessMask::COLOR_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_READ | AccessMask::SHADER_WRITE,
                },
            },
            layout: Some((ImageLayout::ColorAttachment, ImageLayout::General)),
            ownership: None,
        })])
        .unwrap();
    }
    e.transition_resources(&[transition(
        ImageLayout::Undefined,
        ImageLayout::General,
        StageMask::HOST,
        AccessMask::EMPTY,
        StageMask::COMPUTE,
        AccessMask::SHADER_WRITE,
    )])
    .unwrap();
    let mut p = compute.begin(e).unwrap();
    p.set_pipeline(&cp).unwrap();
    p.set_bind_group(0, &write, &[]).unwrap();
    p.dispatch([1, 1, 1]).unwrap();
    p.finish().unwrap();
    e.transition_resources(&[transition(
        ImageLayout::General,
        ImageLayout::ShaderReadOnly,
        StageMask::COMPUTE,
        AccessMask::SHADER_WRITE,
        StageMask::VERTEX,
        AccessMask::SHADER_READ,
    )])
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
    p.set_bind_group(0, &read, &[]).unwrap();
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
        &drawbuf,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 32,
            rows_per_image: 8,
        },
    )
    .unwrap();
    let b = e.readback_buffer(&drawbuf, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers writes and both readbacks.
    let draw = unsafe { d.read_after(b, &point).await }.unwrap().bytes;
    let pixel = &draw[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4];
    eprintln!("storage array: {pixel:?}");
    assert_eq!(pixel, &[159, 0, 0, 255]);
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_binding_dynamic_image_query() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(binding_dynamic_image_query(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_binding_dynamic_image_query() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(binding_dynamic_image_query(&d));
}
