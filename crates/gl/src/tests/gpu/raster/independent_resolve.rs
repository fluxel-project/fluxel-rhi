//! Standalone `CommandEncoder::resolve` hardware fixture.
//!
//! This intentionally resolves after the raster pass has ended, so it cannot
//! accidentally exercise the pass-attachment resolve path.

use super::*;

fn clear_color(
    encoder: &mut CommandEncoder<Gl>,
    view: &TextureView<Gl>,
    area: Rect,
    color: [f32; 4],
) {
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float(color)),
        store: StoreOp::Store,
    })];
    encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: area,
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap()
        .finish()
        .unwrap();
}

async fn standalone_resolve(device: &Device<Gl>) {
    let texture = |samples, usage| {
        device
            .create_texture(&TextureDesc {
                label: None,
                dimension: TextureDimension::D2,
                extent: [8, 8, 1],
                array_layers: 1,
                format: TextureFormat::Rgba8Unorm,
                mip_levels: 1,
                samples,
                usage,
                view_formats: vec![],
            })
            .unwrap()
    };
    let source = texture(4, TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC);
    let destination = texture(
        1,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
    );
    let source_view = device
        .create_view(
            &source,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let destination_view = device
        .create_view(
            &destination,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device,
        lane: device.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 245).unwrap();
    // SAFETY: every resource outlives this direct-issue session and its fence.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin standalone resolve commands"));
    let encoder = run.commands();
    let full = Rect {
        x: 0,
        y: 0,
        width: 8,
        height: 8,
    };
    // Three asymmetric source rows make a vertical inversion observable.  The
    // destination starts magenta so both rectangle placement and padding are
    // checked after resolving a nonzero source Y into a different target Y.
    clear_color(encoder, &source_view, full, [0., 0., 0., 1.]);
    clear_color(
        encoder,
        &source_view,
        Rect {
            x: 0,
            y: 1,
            width: 8,
            height: 1,
        },
        [1., 0., 0., 1.],
    );
    clear_color(
        encoder,
        &source_view,
        Rect {
            x: 0,
            y: 2,
            width: 8,
            height: 1,
        },
        [0., 1., 0., 1.],
    );
    clear_color(
        encoder,
        &source_view,
        Rect {
            x: 0,
            y: 3,
            width: 8,
            height: 1,
        },
        [0., 0., 1., 1.],
    );
    clear_color(encoder, &destination_view, full, [1., 0., 1., 1.]);
    let region = |texture| TextureRegion {
        texture,
        mip: 0,
        origin: [0, 0, 0],
        extent: [8, 8, 1],
        aspect: TextureAspects::COLOR,
    };
    encoder
        .resolve(
            TextureRegion {
                origin: [2, 1, 0],
                extent: [4, 3, 1],
                ..region(&source)
            },
            TextureRegion {
                texture: &destination,
                mip: 0,
                origin: [1, 4, 0],
                extent: [4, 3, 1],
                aspect: TextureAspects::COLOR,
            },
        )
        .unwrap();
    encoder
        .texture_to_buffer(
            region(&destination),
            &output,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
    let ticket = encoder.readback_buffer(&output, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("standalone resolve did not issue"),
    };
    device.wait_completion(&point).await.unwrap();
    // SAFETY: `point` includes the resolve, copy, and readback request.
    let bytes = unsafe { device.read_after(ticket, &point).await }
        .unwrap()
        .bytes;
    let pixel = |x: usize, y: usize| &bytes[y * 32 + x * 4..y * 32 + x * 4 + 4];
    for x in 1..5 {
        assert_eq!(pixel(x, 4), &[255, 0, 0, 255]);
        assert_eq!(pixel(x, 5), &[0, 255, 0, 255]);
        assert_eq!(pixel(x, 6), &[0, 0, 255, 255]);
    }
    assert_eq!(pixel(1, 3), &[255, 0, 255, 255]);
    assert_eq!(pixel(1, 7), &[255, 0, 255, 255]);
    assert_eq!(pixel(0, 5), &[255, 0, 255, 255]);
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_independent_resolve() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL standalone resolve", 1024, 768).unwrap()).unwrap();
    // SAFETY: window and its GL context outlive the fixture.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    block_on(standalone_resolve(&block_on(open(native))));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_independent_resolve() {
    let device = block_on(open(GlInstance::gles_pbuffer([8, 8]).unwrap()));
    block_on(standalone_resolve(&device));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_independent_resolve() {
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
    let device = open(native).await;
    let raw = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::WebGl2RenderingContext>()
        .unwrap();
    if raw
        .get_extension("WEBGL_debug_renderer_info")
        .unwrap()
        .is_some()
    {
        wasm_bindgen_test::console_log!(
            "WEBGL2_RESOLVE_RENDERER {:?}",
            raw.get_parameter(0x9246).unwrap()
        );
    }
    let mut desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 4,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
        view_formats: vec![],
    };
    assert!(matches!(
        device.texture_support(&desc),
        Support::Supported(_)
    ));
    // WebGL2 can resolve this source but has no same-sample image-copy route:
    // ES framebuffer blits reject a multisampled draw framebuffer.
    desc.usage |= TextureUsage::COPY_DST;
    assert!(matches!(
        device.texture_support(&desc),
        Support::Unsupported(_)
    ));
    standalone_resolve(&device).await;
}
