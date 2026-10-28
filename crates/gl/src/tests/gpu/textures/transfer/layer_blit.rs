//! Layered and 3D blits through the public Core path.
use super::*;

async fn layer_blit(d: &Device<Gl>, required: bool) {
    for dimension in [TextureDimension::D2, TextureDimension::D3] {
        let desc = TextureDesc {
            label: None,
            dimension,
            extent: if dimension == TextureDimension::D3 {
                [2, 2, 4]
            } else {
                [2, 2, 1]
            },
            array_layers: if dimension == TextureDimension::D3 {
                1
            } else {
                4
            },
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        };
        if !matches!(d.texture_support(&desc), Support::Supported(_)) {
            assert!(!required, "WebGL2 requires RGBA8 layered textures");
            continue;
        }
        let src = d.create_texture(&desc).unwrap();
        let dst = d.create_texture(&desc).unwrap();
        let buffer = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 32,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let fixture = common::ImmediateFixture {
            device: d,
            lane: d.lanes()[0].id,
        };
        let plan = common::single_commands_plan(&fixture, 451).unwrap();
        // SAFETY: resources and device outlive the submitted work.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin layer blit"));
        let e = run.commands();
        let mut pixels = Vec::new();
        for color in [
            [1, 2, 3, 255],
            [21, 22, 23, 255],
            [41, 42, 43, 255],
            [61, 62, 63, 255],
        ] {
            for _ in 0..4 {
                pixels.extend_from_slice(&color);
            }
        }
        e.write_texture(
            TextureRegion {
                texture: &src,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 2, 4],
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
        e.blit(
            TextureRegion {
                texture: &src,
                mip: 0,
                origin: [0, 0, 1],
                extent: [2, 2, 2],
                aspect: TextureAspects::COLOR,
            },
            TextureRegion {
                texture: &dst,
                mip: 0,
                origin: [0, 0, 2],
                extent: [2, 2, 2],
                aspect: TextureAspects::COLOR,
            },
            FilterMode::Nearest,
        )
        .unwrap();
        e.texture_to_buffer(
            TextureRegion {
                texture: &dst,
                mip: 0,
                origin: [0, 0, 2],
                extent: [2, 2, 2],
                aspect: TextureAspects::COLOR,
            },
            &buffer,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 8,
                rows_per_image: 2,
            },
        )
        .unwrap();
        let ticket = e.readback_buffer(&buffer, 0..32).unwrap();
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish layer blit"),
        };
        d.wait_completion(&point).await.unwrap();
        // SAFETY: the completion covers the readback.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        for i in 0..4 {
            assert_eq!(&bytes[i * 4..i * 4 + 4], &[21, 22, 23, 255]);
        }
        for i in 4..8 {
            assert_eq!(&bytes[i * 4..i * 4 + 4], &[41, 42, 43, 255]);
        }
        eprintln!("layer blit {dimension:?} passed");
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_layer_blit() {
    let window = fluxel_host::Window::new(
        fluxel_host::WindowConfig::new("GL layer blit fixture", 64, 64).unwrap(),
    )
    .unwrap();
    // SAFETY: the window outlives the context and device.
    let native = unsafe { GlInstance::from_wgl_window(&window, [64, 64]) }.unwrap();
    block_on(layer_blit(&block_on(open(native)), false));
}
#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_layer_blit() {
    block_on(layer_blit(
        &block_on(open(GlInstance::gles_pbuffer([8, 8]).unwrap())),
        false,
    ));
}
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_layer_blit() {
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
    layer_blit(
        &open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await,
        true,
    )
    .await;
}
