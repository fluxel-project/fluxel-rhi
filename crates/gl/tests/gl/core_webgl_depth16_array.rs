//! WebGL2 D16 array upload and transform-feedback readback coverage.
//!
//! A mip-level subrectangle across two nonzero layers verifies that the
//! readback shader selects each `sampler2DArray` layer and honors destination
//! row/image strides without touching their padding.
use super::*;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_depth16_array_partial_layers_readback() {
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
    let device = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    let texture = || {
        device
            .create_texture(&TextureDesc {
                label: None,
                dimension: TextureDimension::D2,
                extent: [6, 4, 1],
                array_layers: 3,
                format: TextureFormat::Depth16Unorm,
                mip_levels: 2,
                samples: 1,
                usage: TextureUsage::COPY_SRC
                    | TextureUsage::COPY_DST
                    | TextureUsage::TEXTURE_BINDING,
                view_formats: vec![],
            })
            .unwrap()
    };
    let cpu = texture();
    let pbo = texture();
    let upload = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 48,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 112,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    // Mip 1 is 3x2.  Layer 1 and 2 intentionally have distinct values, with
    // both padded rows and a padded third row in every image.
    let mut source = vec![0xd3; 48];
    source[4..10].copy_from_slice(&[0, 0, 1, 0x40, 0xfe, 0xff]);
    source[12..18].copy_from_slice(&[2, 0, 0, 0x80, 0xff, 0xff]);
    source[28..34].copy_from_slice(&[0x11, 0x11, 0x22, 0x22, 0x33, 0x33]);
    source[36..42].copy_from_slice(&[0x44, 0x44, 0x55, 0x55, 0x66, 0x66]);
    let fixture = common::ImmediateFixture {
        device: &device,
        lane: device.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 979).unwrap();
    // SAFETY: the texture and output remain live through the completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin D16 array"));
    let region = |texture| TextureRegion {
        texture,
        mip: 1,
        origin: [0, 0, 1],
        extent: [3, 2, 2],
        aspect: TextureAspects::DEPTH,
    };
    let layout = TexelCopyBufferLayout {
        offset: 4,
        bytes_per_row: 8,
        rows_per_image: 3,
    };
    run.commands().write_buffer(&upload, 0, &source).unwrap();
    run.commands()
        .write_texture(region(&cpu), layout, &source)
        .unwrap();
    run.commands()
        .buffer_to_texture(&upload, layout, region(&pbo))
        .unwrap();
    run.commands()
        .write_buffer(&output, 0, &[0xa7; 112])
        .unwrap();
    let output_layout = |offset| TexelCopyBufferLayout {
        offset,
        bytes_per_row: 8,
        rows_per_image: 3,
    };
    run.commands()
        .texture_to_buffer(region(&cpu), &output, output_layout(5))
        .unwrap();
    run.commands()
        .texture_to_buffer(region(&pbo), &output, output_layout(53))
        .unwrap();
    let ticket = run.commands().readback_buffer(&output, 0..112).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish WebGL D16 array fixture"),
    };
    device.wait_completion(&point).await.unwrap();
    // SAFETY: `point` covers CPU and PBO uploads, transform-feedback
    // conversion, row/image-strided destination copies, and staging readback.
    let actual = unsafe { device.read_after(ticket, &point).await }
        .unwrap()
        .bytes;
    let expected = [
        [0, 0, 1, 0x40, 0xfe, 0xff],
        [2, 0, 0, 0x80, 0xff, 0xff],
        [0x11, 0x11, 0x22, 0x22, 0x33, 0x33],
        [0x44, 0x44, 0x55, 0x55, 0x66, 0x66],
    ];
    for (copy, bases) in [[5usize, 13, 29, 37], [53, 61, 77, 85]]
        .into_iter()
        .enumerate()
    {
        for (row, base) in bases.into_iter().enumerate() {
            assert_eq!(
                &actual[base..base + 6],
                expected[row],
                "copy {copy}, row {row}"
            );
            assert_eq!(
                &actual[base + 6..base + 8],
                &[0xa7; 2],
                "copy {copy}, row padding {row}"
            );
        }
    }
    for index in [
        0usize, 4, 11, 12, 20, 28, 35, 36, 44, 52, 59, 60, 68, 76, 83, 84, 92, 111,
    ] {
        assert_eq!(actual[index], 0xa7, "untouched byte {index}");
    }
}
