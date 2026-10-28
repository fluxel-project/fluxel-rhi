//! WebGL2 D16 upload, depth copy, and transform-feedback readback coverage.
//!
//! The odd width makes the readback shader pack a final single-depth word;
//! nonzero offsets and padded rows prove that the private pixel/TF buffers do
//! not overwrite caller-owned padding.
use super::*;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_depth16_upload_copy_and_readback() {
    let device = webgl2_device([8, 8]).await;
    let texture = || {
        device
            .create_texture(&TextureDesc {
                label: None,
                dimension: TextureDimension::D2,
                extent: [6, 4, 1],
                array_layers: 1,
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
    let copied_cpu = texture();
    let copied_pbo = texture();
    let upload = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 24,
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
    // Mip 1 is 3x2. Its six D16 words deliberately cover both endpoints and
    // nontrivial values; bytes outside the two 6-byte payload rows are sentinels.
    let mut source = vec![0xd3; 24];
    source[4..10].copy_from_slice(&[0, 0, 1, 0x40, 0xfe, 0xff]);
    source[12..18].copy_from_slice(&[2, 0, 0, 0x80, 0xff, 0xff]);
    let fixture = common::ImmediateFixture {
        device: &device,
        lane: device.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 976).unwrap();
    // SAFETY: all fixtures remain live through the returned completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin WebGL D16 fixture"));
    let commands = run.commands();
    let layout = TexelCopyBufferLayout {
        offset: 4,
        bytes_per_row: 8,
        rows_per_image: 2,
    };
    let region = |texture| TextureRegion {
        texture,
        mip: 1,
        origin: [0, 0, 0],
        extent: [3, 2, 1],
        aspect: TextureAspects::DEPTH,
    };
    commands.write_buffer(&upload, 0, &source).unwrap();
    commands
        .write_texture(region(&cpu), layout, &source)
        .unwrap();
    commands
        .buffer_to_texture(&upload, layout, region(&pbo))
        .unwrap();
    commands
        .copy_texture(region(&cpu), region(&copied_cpu))
        .unwrap();
    commands
        .copy_texture(region(&pbo), region(&copied_pbo))
        .unwrap();
    commands.write_buffer(&output, 0, &[0xa7; 112]).unwrap();
    let output_layout = |offset| TexelCopyBufferLayout {
        offset,
        bytes_per_row: 8,
        rows_per_image: 2,
    };
    commands
        .texture_to_buffer(region(&copied_cpu), &output, output_layout(5))
        .unwrap();
    commands
        .texture_to_buffer(region(&copied_pbo), &output, output_layout(29))
        .unwrap();
    commands
        .texture_to_buffer(region(&cpu), &output, output_layout(53))
        .unwrap();
    commands
        .texture_to_buffer(region(&pbo), &output, output_layout(77))
        .unwrap();
    let ticket = commands.readback_buffer(&output, 0..112).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish WebGL D16 fixture"),
    };
    device.wait_completion(&point).await.unwrap();
    // SAFETY: `point` covers both uploads, both depth copies, and the readback.
    let actual = unsafe { device.read_after(ticket, &point).await }
        .unwrap()
        .bytes;
    let expected_rows = [[0, 0, 1, 0x40, 0xfe, 0xff], [2, 0, 0, 0x80, 0xff, 0xff]];
    for base in [53usize, 77, 5, 29] {
        assert_eq!(&actual[base..base + 6], expected_rows[0]);
        assert_eq!(
            &actual[base + 8..base + 14],
            expected_rows[1],
            "second row base {base}"
        );
        assert_eq!(actual[base + 6], 0xa7);
        assert_eq!(actual[base + 7], 0xa7);
    }
    for index in [0, 4, 11, 12, 28, 35, 36, 52, 59, 60, 76, 83, 84, 111] {
        assert_eq!(actual[index], 0xa7, "padding byte {index} changed");
    }
}
