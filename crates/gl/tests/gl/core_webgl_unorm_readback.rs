//! WebGL2 exact UNORM pixel-pack readback after discovery's byte probe.
use super::*;

async fn webgl_unorm_readback(d: &Device<Gl>, format: TextureFormat, pixel: &[u8]) {
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
    if let Support::Unsupported(reason) = d.texture_support(&desc) {
        wasm_bindgen_test::console_log!("SKIP {format:?}: {reason}");
        return;
    }
    let texture = d.create_texture(&desc).unwrap();
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 128,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let stride = pixel.len() * 2 + 3;
    let mut expected = vec![0xad; 128];
    for row in 0..2 {
        for column in 0..2 {
            let offset = 7 + row * stride + column * pixel.len();
            if row == 0 && column == 1 {
                expected[offset..offset + pixel.len()].fill(0);
            } else {
                expected[offset..offset + pixel.len()].copy_from_slice(pixel);
            }
        }
    }
    let layout = TexelCopyBufferLayout {
        offset: 7,
        bytes_per_row: stride as u32,
        rows_per_image: 2,
    };
    let region = || TextureRegion {
        texture: &texture,
        mip: 1,
        origin: [1, 1, 1],
        extent: [2, 2, 1],
        aspect: TextureAspects::COLOR,
    };
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 963).unwrap();
    // SAFETY: all resources remain live through the returned completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin UNORM readback"));
    let commands = run.commands();
    commands.write_buffer(&buffer, 0, &[0xad; 128]).unwrap();
    commands.write_texture(region(), layout, &expected).unwrap();
    commands
        .texture_to_buffer(region(), &buffer, layout)
        .unwrap_or_else(|error| panic!("{format:?}: texture-to-buffer: {error:?}"));
    let ticket = commands.readback_buffer(&buffer, 0..128).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: the point covers the transfer and browser staging readback.
    assert_eq!(
        unsafe { d.read_after(ticket, &point).await }.unwrap().bytes,
        expected,
        "{format:?}"
    );
    wasm_bindgen_test::console_log!("EXECUTED {format:?}: exact UNORM PBO readback");
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_unorm_readback() {
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
    for (format, pixel) in [
        (TextureFormat::R8Unorm, &[0x40][..]),
        (TextureFormat::Rg8Unorm, &[0x40, 0xc0][..]),
        (TextureFormat::Rgba8Unorm, &[0x40, 0xc0, 0x11, 0xff][..]),
        (TextureFormat::R16Unorm, &[0x00, 0x40][..]),
        (TextureFormat::Rg16Unorm, &[0x00, 0x40, 0x00, 0xc0][..]),
        (
            TextureFormat::Rgba16Unorm,
            &[0x00, 0x40, 0x00, 0xc0, 0xff, 0xff, 0, 0][..],
        ),
    ] {
        webgl_unorm_readback(&d, format, pixel).await;
    }
}
