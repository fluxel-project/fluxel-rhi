//! MRT multisample resolves preserve attachment slots and partial destinations.
use super::*;

async fn resolve_mrt(d: &Device<Gl>, standalone: bool) {
    let texture = |samples| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples,
            usage: if samples == 1 {
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC
            } else {
                TextureUsage::RENDER_ATTACHMENT
            },
            view_formats: vec![],
        })
        .unwrap()
    };
    let a = texture(4);
    let b = texture(4);
    let ra = texture(1);
    let rb = texture(1);
    let view = |t: &Texture<Gl>| {
        d.create_view(
            t,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap()
    };
    let av = view(&a);
    let bv = view(&b);
    let rav = view(&ra);
    let rbv = view(&rb);
    let buffer = || {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let ba = buffer();
    let bb = buffer();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let full = Rect {
        x: 0,
        y: 0,
        width: 8,
        height: 8,
    };
    let partial = Rect {
        x: 2,
        y: 2,
        width: 4,
        height: 4,
    };
    let pass = |colors, render_area| RasterPassDesc {
        label: None,
        colors,
        depth_stencil: None,
        render_area,
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    };
    let sentinel = |view| {
        Some(ColorAttachment {
            view: AttachmentView::Texture(view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 1., 1.])),
            store: StoreOp::Store,
        })
    };
    let resolving = |view, dst, color| {
        Some(ColorAttachment {
            view: AttachmentView::Texture(view),
            resolve_target: Some(AttachmentView::Texture(dst)),
            load: if standalone {
                LoadOp::Load
            } else {
                LoadOp::Clear(ClearColor::Float(color))
            },
            store: StoreOp::Store,
        })
    };
    let initial = [sentinel(&rav), None, sentinel(&rbv)];
    let colors = [
        resolving(&av, &rav, [1., 0., 0., 1.]),
        None,
        resolving(&bv, &rbv, [0., 1., 0., 1.]),
    ];
    let plan = common::single_commands_plan(&fixture, 141).unwrap();
    // SAFETY: all textures and buffers outlive the immediate sessions and their checkpoints.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    let e = run.commands();
    e.begin_raster(&pass(&initial, full))
        .unwrap()
        .finish()
        .unwrap();
    if standalone {
        e.clear_texture(
            &a,
            0..1,
            0..1,
            TextureClearValue::Color(ClearColor::Float([1., 0., 0., 1.])),
        )
        .unwrap();
        e.clear_texture(
            &b,
            0..1,
            0..1,
            TextureClearValue::Color(ClearColor::Float([0., 1., 0., 1.])),
        )
        .unwrap();
    }
    e.begin_raster(&pass(&colors, partial))
        .unwrap()
        .finish()
        .unwrap();
    for (texture, buffer) in [(&ra, &ba), (&rb, &bb)] {
        e.texture_to_buffer(
            TextureRegion {
                texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            buffer,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
    }
    let ta = e.readback_buffer(&ba, 0..256).unwrap();
    let tb = e.readback_buffer(&bb, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    for (ticket, center) in [(ta, [255, 0, 0, 255]), (tb, [0, 255, 0, 255])] {
        // SAFETY: point covers the preceding copy and readback.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        assert_eq!(&bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4], &center);
        assert_eq!(&bytes[0..4], &[0, 0, 255, 255]);
    }
    // Dropping an unfinished pass aborts it; the already resolved destinations
    // must retain their previous pixels. A new session observes that result.
    let plan = common::single_commands_plan(&fixture, 142).unwrap();
    // SAFETY: all resources remain alive through the subsequent readback.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin abort"));
    {
        let _unfinished = run.commands().begin_raster(&pass(&colors, full)).unwrap();
    }
    let _ = run.close();
    let _ = session.abort(Error::InvalidState);
    let plan = common::single_commands_plan(&fixture, 143).unwrap();
    // SAFETY: all resources are live and the same immediate context preserves command ordering.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin readback"));
    let e = run.commands();
    for (texture, buffer) in [(&ra, &ba), (&rb, &bb)] {
        e.texture_to_buffer(
            TextureRegion {
                texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            buffer,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
    }
    let ta = e.readback_buffer(&ba, 0..256).unwrap();
    let tb = e.readback_buffer(&bb, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish readback"),
    };
    d.wait_completion(&point).await.unwrap();
    for (ticket, center) in [(ta, [255, 0, 0, 255]), (tb, [0, 255, 0, 255])] {
        // SAFETY: this completion covers the copies and readbacks.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        assert_eq!(&bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4], &center);
        assert_eq!(&bytes[0..4], &[0, 0, 255, 255]);
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_mrt_resolve() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(resolve_mrt(&d, false));
    block_on(resolve_mrt(&d, true));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_mrt_resolve() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(resolve_mrt(&d, false));
    block_on(resolve_mrt(&d, true));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_mrt_resolve() {
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
    let native = GlInstance::from_webgl2_canvas(canvas.clone()).unwrap();
    let d = open(native).await;
    resolve_mrt(&d, false).await;
    resolve_mrt(&d, true).await;
}
