//! GLES standalone depth CPU and pixel-buffer uploads observed with depth testing.
use super::*;
async fn standalone_depth(d: &Device<Gl>, format: TextureFormat) {
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [8, 8, 1],
        array_layers: 2,
        format,
        mip_levels: 2,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    };
    if !matches!(d.texture_support(&desc), Support::Supported(_)) {
        eprintln!("SKIP {format:?}");
        return;
    }
    let tex = d.create_texture(&desc).unwrap();
    let dest = d.create_texture(&desc).unwrap();
    let dest_view = d
        .create_view(
            &dest,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH,
                mips: 1..2,
                layers: 1..2,
            },
        )
        .unwrap();
    let view = d
        .create_view(
            &tex,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH,
                mips: 1..2,
                layers: 1..2,
            },
        )
        .unwrap();
    let color = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let cv = d
        .create_view(
            &color,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let buf = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let shader=d.shader(ShaderDesc { label:None,format:ShaderFormat::Wgsl,bytes:Arc::from(br"@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.); } @fragment fn fs()->@location(0) vec4f { return vec4f(1.,0.,0.,1.); }".as_slice()),entries:vec![ShaderEntryPoint{name:"vs".into(),stage:ShaderStage::Vertex,workgroup_size:None},ShaderEntryPoint{name:"fs".into(),stage:ShaderStage::Fragment,workgroup_size:None}],required_features:vec![] }).await.unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let face = StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    let pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: shader.clone(),
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: shader,
                entry: "fs".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![],
            color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
            primitive: triangle_list_state(),
            depth_stencil: Some(DepthStencilState {
                format,
                depth_write_enabled: false,
                depth_compare: CompareFunction::Greater,
                stencil_front: face,
                stencil_back: face,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
                depth_bias_constant: 0,
                depth_bias_slope_scale: 0.,
                depth_bias_clamp: 0.,
            }),
            multisample: single_sample_state(u64::MAX),
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 493).unwrap();
    // SAFETY: resources outlive completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin GLES depth"));
    let e = run.commands();
    let pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&view),
                depth: Some((LoadOp::Clear(0.75), StoreOp::Store)),
                stencil: None,
                depth_read_only: false,
                stencil_read_only: false,
            }),
            render_area: Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.finish().unwrap();
    let region = || TextureRegion {
        texture: &tex,
        mip: 1,
        origin: [1, 1, 1],
        extent: [2, 2, 1],
        aspect: TextureAspects::DEPTH,
    };
    let values = if format == TextureFormat::Depth16Unorm {
        [
            0x4000u16.to_le_bytes(),
            0x4000u16.to_le_bytes(),
            0xc000u16.to_le_bytes(),
            0xc000u16.to_le_bytes(),
        ]
        .concat()
    } else {
        [
            0.25f32.to_le_bytes(),
            0.25f32.to_le_bytes(),
            0.75f32.to_le_bytes(),
            0.75f32.to_le_bytes(),
        ]
        .concat()
    };
    let row = if format == TextureFormat::Depth16Unorm {
        4
    } else {
        8
    };
    let mut data = [0u8; 32];
    data[4..4 + row].copy_from_slice(&values[..row]);
    data[20..20 + row].copy_from_slice(&values[row..]);
    e.write_buffer(&buf, 192, &data).unwrap();
    let layout = |offset| TexelCopyBufferLayout {
        offset,
        bytes_per_row: 16,
        rows_per_image: 2,
    };
    e.write_texture(region(), layout(4), &data).unwrap();
    let draw = |e: &mut CommandEncoder<Gl>, target_view: &TextureView<Gl>, offset| {
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&cv),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 1., 1.])),
            store: StoreOp::Store,
        })];
        let mut pass = e
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &colors,
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(target_view),
                    depth: Some((LoadOp::Load, StoreOp::Store)),
                    stencil: None,
                    depth_read_only: false,
                    stencil_read_only: false,
                }),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.draw(0..3, 0..1).unwrap();
        pass.finish().unwrap();
        e.texture_to_buffer(
            TextureRegion {
                texture: &color,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &buf,
            TexelCopyBufferLayout {
                offset,
                bytes_per_row: 16,
                rows_per_image: 4,
            },
        )
        .unwrap();
    };
    draw(e, &view, 0);
    e.buffer_to_texture(&buf, layout(196), region()).unwrap();
    draw(e, &view, 64);
    let pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&dest_view),
                depth: Some((LoadOp::Clear(0.75), StoreOp::Store)),
                stencil: None,
                depth_read_only: false,
                stencil_read_only: false,
            }),
            render_area: Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.finish().unwrap();
    e.copy_texture(
        region(),
        TextureRegion {
            texture: &dest,
            mip: 1,
            origin: [1, 1, 1],
            extent: [2, 2, 1],
            aspect: TextureAspects::DEPTH,
        },
    )
    .unwrap();
    draw(e, &dest_view, 128);
    let ticket = e.readback_buffer(&buf, 0..192).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(p) => p.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion point covers both draws.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    eprintln!(
        "{format:?} upload CPU pixels: {:?}; PBO pixels: {:?}",
        &bytes[..64],
        &bytes[64..128]
    );
    for offset in [0, 64, 128] {
        let p = offset + 16 + 4;
        assert_eq!(&bytes[p..p + 4], &[255, 0, 0, 255]);
        assert_eq!(&bytes[p + 16..p + 20], &[0, 0, 255, 255]);
        assert_eq!(&bytes[offset..offset + 4], &[0, 0, 255, 255]);
    }
}
#[cfg(feature = "native-gles-egl")]
fn run_standalone_depth_upload(format: TextureFormat) {
    let d = block_on(open(
        GlInstance::gles_pbuffer([8, 8]).expect("EGL GLES pbuffer"),
    ));
    block_on(standalone_depth(&d, format));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_standalone_depth_upload() {
    run_standalone_depth_upload(TextureFormat::Depth32Float);
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_standalone_depth16_upload() {
    run_standalone_depth_upload(TextureFormat::Depth16Unorm);
}
