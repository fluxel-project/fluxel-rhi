//! Every D16 code point survives the normalized sampling / TF reconstruction.
use super::*;

#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_depth16_all_code_points() {
    use wasm_bindgen::JsCast;
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    let device = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    let texture = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [256, 256, 1],
            array_layers: 1,
            format: TextureFormat::Depth16Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let expected: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    let buffer = device
        .create_buffer(&BufferDesc {
            label: None,
            size: expected.len() as u64,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: &device,
        lane: device.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 977).unwrap();
    // SAFETY: all source/destination objects outlive the completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin D16 domain"));
    let region = || TextureRegion {
        texture: &texture,
        mip: 0,
        origin: [0; 3],
        extent: [256, 256, 1],
        aspect: TextureAspects::DEPTH,
    };
    let layout = TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: 512,
        rows_per_image: 256,
    };
    run.commands()
        .write_texture(region(), layout, &expected)
        .unwrap();
    run.commands()
        .texture_to_buffer(region(), &buffer, layout)
        .unwrap();
    let ticket = run
        .commands()
        .readback_buffer(&buffer, 0..expected.len() as u64)
        .unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish D16 domain"),
    };
    device.wait_completion(&point).await.unwrap();
    // SAFETY: the point covers the TF output and readback staging copy.
    let actual = unsafe { device.read_after(ticket, &point).await }
        .unwrap()
        .bytes;
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual
        .chunks_exact(2)
        .zip(expected.chunks_exact(2))
        .enumerate()
    {
        assert_eq!(actual, expected, "D16 code point {index}");
    }
}
