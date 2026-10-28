//! Exact integer texture copy through WebGL pixel-pack buffers.
use super::*;

#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl_integer_copy_bits_and_padding() {
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
    let mut tested = 0;
    for format in [TextureFormat::Rgba32Uint, TextureFormat::Rgba32Sint] {
        let texture = match d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [2, 2, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        }) {
            Ok(v) => v,
            Err(Error::Unsupported(_)) => continue,
            Err(e) => panic!("{format:?}: {e:?}"),
        };
        tested += 1;
        let mut bytes = vec![0xadu8; 100];
        for row in 0..2 {
            for byte in 0..32 {
                bytes[7 + row * 35 + byte] = ((row * 37 + byte * 11) as u8).wrapping_add(1);
            }
        }
        let buffer = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 100,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let fixture = common::ImmediateFixture {
            device: &d,
            lane: d.lanes()[0].id,
        };
        let plan = common::single_commands_plan(&fixture, 913).unwrap();
        // SAFETY: resource lifetimes extend through the completion point.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
        let e = run.commands();
        e.write_buffer(&buffer, 0, &[0xad; 100]).unwrap();
        let region = || TextureRegion {
            texture: &texture,
            mip: 0,
            origin: [0, 0, 0],
            extent: [2, 2, 1],
            aspect: TextureAspects::COLOR,
        };
        let layout = || TexelCopyBufferLayout {
            offset: 7,
            bytes_per_row: 35,
            rows_per_image: 2,
        };
        e.write_texture(region(), layout(), &bytes).unwrap();
        e.texture_to_buffer(region(), &buffer, layout()).unwrap();
        let ticket = e.readback_buffer(&buffer, 0..100).unwrap();
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish"),
        };
        d.wait_completion(&point).await.unwrap();
        // SAFETY: completion covers the readback.
        let actual = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        assert_eq!(actual, bytes, "{format:?} raw bits and untouched padding");
    }
    web_sys::console::log_1(&format!("tested {tested} RGBA32 integer formats").into());
    assert!(tested > 0, "no supported integer format exercised");
}
