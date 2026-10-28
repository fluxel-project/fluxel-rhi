//! Packed depth/stencil plane copies preserve the other plane.
use super::*;

async fn packed_plane_copy(d: &Device<Gl>, format: TextureFormat) {
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [8, 8, 1],
        array_layers: 2,
        format,
        mip_levels: 2,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    if !matches!(d.texture_support(&desc), Support::Supported(_)) {
        eprintln!("SKIP: {format:?} format unsupported");
        return;
    }
    let a = d.create_texture(&desc).unwrap();
    let b = d.create_texture(&desc).unwrap();
    let view = |texture| {
        d.create_view(
            texture,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH | TextureAspects::STENCIL,
                mips: 1..2,
                layers: 1..2,
            },
        )
        .unwrap()
    };
    let av = view(&a);
    let bv = view(&b);
    let buf = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 512,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    // The Depth24Plus plane has no portable byte representation. A depth
    // comparison observes it without inventing one: 0.5 must pass Greater
    // against the copied 0.25, and fail if the old 0.75 was retained.
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
    let shader = d.shader(ShaderDesc { label: None, format: ShaderFormat::Wgsl, bytes: Arc::from(br"@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.); } @fragment fn fs()->@location(0) vec4f { return vec4f(1.,0.,0.,1.); }".as_slice()), entries: vec![ShaderEntryPoint { name: "vs".into(), stage: ShaderStage::Vertex, workgroup_size: None }, ShaderEntryPoint { name: "fs".into(), stage: ShaderStage::Fragment, workgroup_size: None }], required_features: vec![] }).await.unwrap();
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
    let plan = common::single_commands_plan(&fixture, 486).unwrap();
    // SAFETY: all resources and the current context live through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin plane copies"));
    let e = run.commands();
    for (v, depth, stencil) in [(&av, 0.25, 17), (&bv, 0.75, 91)] {
        let pass = e
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &[],
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(v),
                    depth: Some((LoadOp::Clear(depth), StoreOp::Store)),
                    stencil: Some((LoadOp::Clear(stencil), StoreOp::Store)),
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
    }
    e.write_buffer(&buf, 0, &[0xa5; 512]).unwrap();
    let region = |texture, aspect| TextureRegion {
        texture,
        mip: 1,
        origin: [1, 1, 1],
        extent: [2, 2, 1],
        aspect,
    };
    let stencil_layout = |offset| TexelCopyBufferLayout {
        offset,
        bytes_per_row: 4,
        rows_per_image: 2,
    };
    let depth_layout = |offset| TexelCopyBufferLayout {
        offset,
        bytes_per_row: 8,
        rows_per_image: 2,
    };
    e.copy_texture(
        region(&a, TextureAspects::DEPTH),
        region(&b, TextureAspects::DEPTH),
    )
    .unwrap();
    e.texture_to_buffer(region(&b, TextureAspects::STENCIL), &buf, stencil_layout(4))
        .unwrap();
    if format == TextureFormat::Depth32FloatStencil8 {
        e.texture_to_buffer(region(&b, TextureAspects::DEPTH), &buf, depth_layout(16))
            .unwrap();
    }
    e.copy_texture(
        region(&a, TextureAspects::STENCIL),
        region(&b, TextureAspects::STENCIL),
    )
    .unwrap();
    e.texture_to_buffer(
        region(&b, TextureAspects::STENCIL),
        &buf,
        stencil_layout(40),
    )
    .unwrap();
    if format == TextureFormat::Depth32FloatStencil8 {
        e.texture_to_buffer(region(&b, TextureAspects::DEPTH), &buf, depth_layout(48))
            .unwrap();
    }
    let cpu_stencil = [0xee, 0xee, 0xee, 0xee, 3, 19, 0xee, 0xee, 77, 251];
    e.write_texture(
        region(&b, TextureAspects::STENCIL),
        stencil_layout(4),
        &cpu_stencil,
    )
    .unwrap();
    e.texture_to_buffer(
        region(&b, TextureAspects::STENCIL),
        &buf,
        stencil_layout(80),
    )
    .unwrap();
    if format == TextureFormat::Depth32FloatStencil8 {
        e.texture_to_buffer(region(&b, TextureAspects::DEPTH), &buf, depth_layout(96))
            .unwrap();
        let depths = [0.5f32.to_le_bytes(); 4].concat();
        e.write_texture(region(&b, TextureAspects::DEPTH), depth_layout(0), &depths)
            .unwrap();
        e.texture_to_buffer(region(&b, TextureAspects::DEPTH), &buf, depth_layout(112))
            .unwrap();
        e.texture_to_buffer(
            region(&b, TextureAspects::STENCIL),
            &buf,
            stencil_layout(128),
        )
        .unwrap();
        e.write_buffer(&buf, 192, &depths).unwrap();
        e.buffer_to_texture(&buf, depth_layout(192), region(&b, TextureAspects::DEPTH))
            .unwrap();
        e.texture_to_buffer(region(&b, TextureAspects::DEPTH), &buf, depth_layout(144))
            .unwrap();
    }
    if d.compute().is_some() {
        e.write_buffer(&buf, 224, &[0, 0, 0, 0, 9, 23, 0, 0, 81, 203])
            .unwrap();
        e.buffer_to_texture(
            &buf,
            stencil_layout(228),
            region(&b, TextureAspects::STENCIL),
        )
        .unwrap();
        e.texture_to_buffer(
            region(&b, TextureAspects::STENCIL),
            &buf,
            stencil_layout(160),
        )
        .unwrap();
        if format == TextureFormat::Depth32FloatStencil8 {
            e.texture_to_buffer(region(&b, TextureAspects::DEPTH), &buf, depth_layout(176))
                .unwrap();
        }
    }
    if format == TextureFormat::Depth24PlusStencil8 {
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
                    view: AttachmentView::Texture(&bv),
                    depth: Some((LoadOp::Load, StoreOp::Store)),
                    stencil: Some((LoadOp::Load, StoreOp::Store)),
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
                offset: 256,
                bytes_per_row: 16,
                rows_per_image: 4,
            },
        )
        .unwrap();
    }
    let ticket = e.readback_buffer(&buf, 0..512).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish plane copies"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: point covers copies and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for (offset, expected) in [(4, 91), (40, 17)] {
        assert_eq!(&bytes[offset..offset + 2], &[expected; 2]);
        assert_eq!(&bytes[offset + 4..offset + 6], &[expected; 2]);
        assert_eq!(&bytes[offset + 2..offset + 4], &[0xa5; 2]);
    }
    for (offset, expected) in [(80, [3, 19, 77, 251]), (128, [3, 19, 77, 251])] {
        if offset == 128 && format != TextureFormat::Depth32FloatStencil8 {
            continue;
        }
        assert_eq!(&bytes[offset..offset + 2], &expected[..2]);
        assert_eq!(&bytes[offset + 4..offset + 6], &expected[2..]);
        assert_eq!(&bytes[offset + 2..offset + 4], &[0xa5; 2]);
    }
    if d.compute().is_some() {
        assert_eq!(&bytes[160..162], &[9, 23]);
        assert_eq!(&bytes[164..166], &[81, 203]);
    }
    if format == TextureFormat::Depth24PlusStencil8 {
        let pixel = 256 + 16 + 4;
        assert_eq!(
            &bytes[pixel..pixel + 4],
            &[255, 0, 0, 255],
            "D24 depth after stencil uploads"
        );
        assert_eq!(
            &bytes[256..260],
            &[0, 0, 255, 255],
            "uncopied D24 depth sentinel"
        );
    }
    if format == TextureFormat::Depth32FloatStencil8 {
        for offset in [16, 48, 96, 112, 144, 176] {
            for i in 0..4 {
                let start = offset + i * 4;
                if offset == 176 && d.compute().is_none() {
                    continue;
                }
                assert_eq!(
                    f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap()),
                    if offset < 112 { 0.25 } else { 0.5 }
                );
            }
        }
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_packed_plane_copy() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL packed plane copy", 1024, 768).unwrap()).unwrap();
    // SAFETY: window remains alive on context owner thread.
    let instance = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let d = block_on(open(instance));
    for format in [
        TextureFormat::Depth24PlusStencil8,
        TextureFormat::Depth32FloatStencil8,
    ] {
        block_on(packed_plane_copy(&d, format));
    }
}
