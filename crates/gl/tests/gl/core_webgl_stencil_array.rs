//! WebGL2 packed stencil-plane transfers over a partial D2-array layer range.
//!
//! The immediate lowering must attach each selected packed layer separately.
//! This fixture uses an odd row pitch, mip 1, two middle layers and padded
//! buffer rows to cover CPU and PBO writes plus GPU-only readback placement.

use super::*;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
async fn packed_array_round_trip(device: &Device<Gl>, format: TextureFormat, serial: u64) {
    const WIDTH: u32 = 5;
    const HEIGHT: u32 = 4;
    const LAYERS: u32 = 2;
    const ROW_BYTES: u32 = 9;
    const ROWS_PER_IMAGE: u32 = 6;
    const CPU_SOURCE: u64 = 1;
    const PBO_SOURCE: u64 = 200;
    const CPU_OUTPUT: u64 = 7;
    const PBO_OUTPUT: u64 = 300;
    const COPY_OUTPUT: u64 = 410;
    const LAYER_STRIDE: usize = (ROW_BYTES * ROWS_PER_IMAGE) as usize;
    let create = || {
        device
            .create_texture(&TextureDesc {
                label: None,
                dimension: TextureDimension::D2,
                extent: [18, 16, 1],
                array_layers: 4,
                format,
                mip_levels: 2,
                samples: 1,
                usage: TextureUsage::COPY_SRC
                    | TextureUsage::COPY_DST
                    | TextureUsage::RENDER_ATTACHMENT,
                view_formats: vec![],
            })
            .unwrap()
    };
    let cpu = create();
    let pbo = create();
    let copied_cpu = create();
    let copied_pbo = create();
    let copied_array = create();
    let source = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 512,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 512,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut bytes = vec![0x91; 512];
    for layer in 0..LAYERS as usize {
        for y in 0..HEIGHT as usize {
            for x in 0..WIDTH as usize {
                let value = (layer * 64 + y * WIDTH as usize + x) as u8;
                bytes[CPU_SOURCE as usize + layer * LAYER_STRIDE + y * ROW_BYTES as usize + x] =
                    value;
                bytes[PBO_SOURCE as usize + layer * LAYER_STRIDE + y * ROW_BYTES as usize + x] =
                    value.reverse_bits();
            }
        }
    }
    let fixture = common::ImmediateFixture {
        device,
        lane: device.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, serial).unwrap();
    // SAFETY: all resources outlive the issued immediate work and completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin WebGL packed stencil array transfer"));
    let commands = run.commands();
    let layout = |offset| TexelCopyBufferLayout {
        offset,
        bytes_per_row: ROW_BYTES,
        rows_per_image: ROWS_PER_IMAGE,
    };
    let region = |texture| TextureRegion {
        texture,
        mip: 1,
        origin: [2, 1, 1],
        extent: [WIDTH, HEIGHT, LAYERS],
        aspect: TextureAspects::STENCIL,
    };
    commands
        .write_texture(region(&cpu), layout(CPU_SOURCE), &bytes)
        .unwrap();
    commands.write_buffer(&source, 0, &bytes).unwrap();
    commands
        .buffer_to_texture(&source, layout(PBO_SOURCE), region(&pbo))
        .unwrap();
    commands
        .copy_texture(region(&cpu), region(&copied_cpu))
        .unwrap();
    commands
        .copy_texture(region(&pbo), region(&copied_pbo))
        .unwrap();
    // Verify the public texture-copy path uses the same partial mip/layer
    // region without touching the independent PBO source pattern.
    commands
        .copy_texture(region(&cpu), region(&copied_array))
        .unwrap();
    commands.write_buffer(&output, 0, &[0xa9; 512]).unwrap();
    commands
        .texture_to_buffer(region(&copied_cpu), &output, layout(CPU_OUTPUT))
        .unwrap();
    commands
        .texture_to_buffer(region(&copied_pbo), &output, layout(PBO_OUTPUT))
        .unwrap();
    commands
        .texture_to_buffer(region(&copied_array), &output, layout(COPY_OUTPUT))
        .unwrap();
    let ticket = commands.readback_buffer(&output, 0..512).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish WebGL packed stencil array transfer"),
    };
    device.wait_completion(&point).await.unwrap();
    // SAFETY: the completion point includes both multi-layer writes and reads.
    let actual = unsafe { device.read_after(ticket, &point).await }
        .unwrap()
        .bytes;
    for layer in 0..LAYERS as usize {
        for y in 0..HEIGHT as usize {
            let cpu = CPU_OUTPUT as usize + layer * LAYER_STRIDE + y * ROW_BYTES as usize;
            let pbo = PBO_OUTPUT as usize + layer * LAYER_STRIDE + y * ROW_BYTES as usize;
            let copied = COPY_OUTPUT as usize + layer * LAYER_STRIDE + y * ROW_BYTES as usize;
            for x in 0..WIDTH as usize {
                let value = (layer * 64 + y * WIDTH as usize + x) as u8;
                assert_eq!(actual[cpu + x], value, "{format:?} CPU ({layer}, {x}, {y})");
                assert_eq!(
                    actual[pbo + x],
                    value.reverse_bits(),
                    "{format:?} PBO ({layer}, {x}, {y})"
                );
                assert_eq!(
                    actual[copied + x],
                    value,
                    "{format:?} copy ({layer}, {x}, {y})"
                );
            }
            assert_eq!(
                &actual[cpu + WIDTH as usize..cpu + ROW_BYTES as usize],
                &[0xa9; 4]
            );
            assert_eq!(
                &actual[pbo + WIDTH as usize..pbo + ROW_BYTES as usize],
                &[0xa9; 4]
            );
            assert_eq!(
                &actual[copied + WIDTH as usize..copied + ROW_BYTES as usize],
                &[0xa9; 4]
            );
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
pub(super) async fn webgl2_packed_stencil_array_transfers() {
    use wasm_bindgen::JsCast;

    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(16);
    canvas.set_height(16);
    let device = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    for (index, format) in [
        TextureFormat::Depth24PlusStencil8,
        TextureFormat::Depth32FloatStencil8,
    ]
    .into_iter()
    .enumerate()
    {
        wasm_bindgen_test::console_log!("Packed stencil array transfer: {format:?}");
        packed_array_round_trip(&device, format, 991 + index as u64).await;
    }
}
