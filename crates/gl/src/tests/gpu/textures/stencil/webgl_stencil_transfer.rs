//! WebGL2 stencil-plane transfer coverage for standalone and packed formats.
//!
//! The Core transfer aspect remains `STENCIL` throughout: client and PBO writes,
//! same-format plane copies, and raw one-byte readback must not reinterpret a
//! packed texture as a color image. A WebGL packed-depth preservation fixture
//! remains to be added; this workload deliberately isolates all 256 stencil
//! values and padding.
use super::*;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
async fn stencil_plane_round_trip(device: &Device<Gl>, format: TextureFormat, serial: u64) {
    // Odd byte rows force UNPACK/PACK_ALIGNMENT=1 in the browser primitives.
    // `17 * 16` still exercises every 8-bit stencil code at least once.
    const WIDTH: u32 = 17;
    const HEIGHT: u32 = 16;
    const SOURCE_OFFSET: u64 = 3;
    const ROW_BYTES: u32 = 21;
    const CPU_OFFSET: u64 = 5;
    const PBO_OFFSET: u64 = 360;
    let create = || {
        device
            .create_texture(&TextureDesc {
                label: None,
                dimension: TextureDimension::D2,
                extent: [19, 18, 1],
                array_layers: 1,
                format,
                mip_levels: 1,
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
    let source_buffer = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 384,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 720,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut cpu_bytes = vec![0x91; 384];
    let mut pbo_bytes = vec![0x91; 384];
    for y in 0..HEIGHT as usize {
        for x in 0..WIDTH as usize {
            let byte = (y * WIDTH as usize + x) as u8;
            let index = SOURCE_OFFSET as usize + y * ROW_BYTES as usize + x;
            cpu_bytes[index] = byte;
            pbo_bytes[index] = byte.reverse_bits();
        }
    }
    let fixture = common::ImmediateFixture {
        device,
        lane: device.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, serial).unwrap();
    // SAFETY: all resources survive the issue and completion point below.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin WebGL stencil transfer"));
    let commands = run.commands();
    let layout = TexelCopyBufferLayout {
        offset: SOURCE_OFFSET,
        bytes_per_row: ROW_BYTES,
        rows_per_image: HEIGHT,
    };
    let region = |texture| TextureRegion {
        texture,
        mip: 0,
        origin: [1, 1, 0],
        extent: [WIDTH, HEIGHT, 1],
        aspect: TextureAspects::STENCIL,
    };
    commands
        .write_texture(region(&cpu), layout, &cpu_bytes)
        .unwrap();
    commands
        .write_buffer(&source_buffer, 0, &pbo_bytes)
        .unwrap();
    commands
        .buffer_to_texture(&source_buffer, layout, region(&pbo))
        .unwrap();
    commands
        .copy_texture(region(&cpu), region(&copied_cpu))
        .unwrap();
    commands
        .copy_texture(region(&pbo), region(&copied_pbo))
        .unwrap();
    commands.write_buffer(&output, 0, &[0xa9; 720]).unwrap();
    let output_layout = |offset| TexelCopyBufferLayout {
        offset,
        bytes_per_row: ROW_BYTES,
        rows_per_image: HEIGHT,
    };
    commands
        .texture_to_buffer(region(&copied_cpu), &output, output_layout(CPU_OFFSET))
        .unwrap();
    commands
        .texture_to_buffer(region(&copied_pbo), &output, output_layout(PBO_OFFSET))
        .unwrap();
    let ticket = commands.readback_buffer(&output, 0..720).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish WebGL stencil transfer"),
    };
    device.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers both write routes, stencil copies and readback.
    let actual = unsafe { device.read_after(ticket, &point).await }
        .unwrap()
        .bytes;
    for y in 0..HEIGHT as usize {
        let cpu_start = CPU_OFFSET as usize + y * ROW_BYTES as usize;
        let pbo_start = PBO_OFFSET as usize + y * ROW_BYTES as usize;
        for x in 0..WIDTH as usize {
            let byte = (y * WIDTH as usize + x) as u8;
            assert_eq!(actual[cpu_start + x], byte, "{format:?} CPU ({x}, {y})");
            assert_eq!(
                actual[pbo_start + x],
                byte.reverse_bits(),
                "{format:?} PBO ({x}, {y})"
            );
        }
        assert_eq!(
            &actual[cpu_start + WIDTH as usize..cpu_start + ROW_BYTES as usize],
            &[0xa9; 4]
        );
        assert_eq!(
            &actual[pbo_start + WIDTH as usize..pbo_start + ROW_BYTES as usize],
            &[0xa9; 4]
        );
    }
    for index in [0, 4, 337, 338, 339, 358, 359, 695, 719] {
        assert_eq!(actual[index], 0xa9, "{format:?} padding byte {index}");
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_stencil_plane_transfers() {
    let device = webgl2_device([16, 16]).await;
    for (index, format) in [
        TextureFormat::Stencil8,
        TextureFormat::Depth24PlusStencil8,
        TextureFormat::Depth32FloatStencil8,
    ]
    .into_iter()
    .enumerate()
    {
        wasm_bindgen_test::console_log!("Stencil transfer format: {format:?}");
        stencil_plane_round_trip(&device, format, 981 + index as u64).await;
    }
}
