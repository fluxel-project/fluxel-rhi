//! Exhaustive WebGL2 normalized-format round trips through Core transfers.
//!
//! These fixtures deliberately use padded, nonzero buffer layouts. The 16-bit
//! cases cover every normalized code on two array layers; RGB10A2 covers every
//! 10-bit code in each RGB channel and every 2-bit alpha code in a 3D texture.

use super::*;

const SENTINEL: u8 = 0xa9;

fn padded_layout(
    width: u32,
    height: u32,
    layers: u32,
    bytes_per_pixel: u32,
) -> (TexelCopyBufferLayout, usize) {
    let bytes_per_row = width * bytes_per_pixel + 5;
    let rows_per_image = height + 3;
    let layout = TexelCopyBufferLayout {
        offset: 11,
        bytes_per_row,
        rows_per_image,
    };
    let required = usize::try_from(
        u64::from(layout.offset)
            + u64::from(bytes_per_row) * u64::from(rows_per_image) * u64::from(layers - 1)
            + u64::from(bytes_per_row) * u64::from(height - 1)
            + u64::from(width) * u64::from(bytes_per_pixel),
    )
    .unwrap();
    (layout, required + 17)
}

fn put_unorm16_plane(
    bytes: &mut [u8],
    layout: TexelCopyBufferLayout,
    width: u32,
    height: u32,
    layer: u32,
    channels: u32,
) {
    let layer_base = usize::try_from(
        u64::from(layout.offset)
            + u64::from(layer) * u64::from(layout.bytes_per_row) * u64::from(layout.rows_per_image),
    )
    .unwrap();
    for y in 0..height {
        for x in 0..width {
            let element = (layer * width * height + y * width + x) as u16;
            let pixel = layer_base
                + y as usize * layout.bytes_per_row as usize
                + x as usize * channels as usize * 2;
            for channel in 0..channels {
                let value = element.wrapping_add((channel * 0x3157) as u16);
                bytes[pixel + channel as usize * 2..pixel + channel as usize * 2 + 2]
                    .copy_from_slice(&value.to_le_bytes());
            }
        }
    }
}

async fn exhaustive_unorm16(device: &Device<Gl>, format: TextureFormat, channels: u32) {
    const WIDTH: u32 = 257;
    const HEIGHT: u32 = 128;
    const LAYERS: u32 = 2;
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [WIDTH * 2, HEIGHT * 2, 1],
        array_layers: 3,
        format,
        mip_levels: 2,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
        view_formats: vec![],
    };
    if let Support::Unsupported(reason) = device.texture_support(&desc) {
        wasm_bindgen_test::console_log!("SKIP {format:?}: {reason}");
        return;
    }
    let texture = device.create_texture(&desc).unwrap();
    let (layout, size) = padded_layout(WIDTH, HEIGHT, LAYERS, channels * 2);
    let mut expected = vec![SENTINEL; size];
    for layer in 0..LAYERS {
        put_unorm16_plane(&mut expected, layout, WIDTH, HEIGHT, layer, channels);
    }
    let buffer = device
        .create_buffer(&BufferDesc {
            label: None,
            size: size as u64,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let region = || TextureRegion {
        texture: &texture,
        mip: 1,
        origin: [0, 0, 1],
        extent: [WIDTH, HEIGHT, LAYERS],
        aspect: TextureAspects::COLOR,
    };
    let fixture = common::ImmediateFixture {
        device,
        lane: device.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 981).unwrap();
    // SAFETY: all resources remain live through the returned completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin normalized readback"));
    let commands = run.commands();
    commands.write_buffer(&buffer, 0, &expected).unwrap();
    commands.write_texture(region(), layout, &expected).unwrap();
    commands
        .texture_to_buffer(region(), &buffer, layout)
        .unwrap_or_else(|error| panic!("{format:?}: normalized 16-bit transfer: {error:?}"));
    let ticket = commands.readback_buffer(&buffer, 0..size as u64).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish {format:?}"),
    };
    device.wait_completion(&point).await.unwrap();
    // SAFETY: `point` covers upload, normalized transform-feedback readback, and staging readback.
    assert_eq!(
        unsafe { device.read_after(ticket, &point).await }
            .unwrap()
            .bytes,
        expected,
        "{format:?}"
    );
    wasm_bindgen_test::console_log!("EXECUTED {format:?}: exhaustive 16-bit normalized transfer");
}

async fn exhaustive_rgb10a2(device: &Device<Gl>) {
    const WIDTH: u32 = 1024;
    const HEIGHT: u32 = 4;
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D3,
        extent: [WIDTH, HEIGHT, 1],
        array_layers: 1,
        format: TextureFormat::Rgb10a2Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
        view_formats: vec![],
    };
    if let Support::Unsupported(reason) = device.texture_support(&desc) {
        wasm_bindgen_test::console_log!("SKIP Rgb10a2Unorm: {reason}");
        return;
    }
    let texture = device.create_texture(&desc).unwrap();
    let (layout, size) = padded_layout(WIDTH, HEIGHT, 1, 4);
    let mut expected = vec![SENTINEL; size];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let red = x;
            let green = (x.wrapping_mul(17)) & 0x3ff;
            let blue = (x.wrapping_add(511)) & 0x3ff;
            let packed = red | (green << 10) | (blue << 20) | (y << 30);
            let offset = layout.offset as usize
                + y as usize * layout.bytes_per_row as usize
                + x as usize * 4;
            expected[offset..offset + 4].copy_from_slice(&packed.to_le_bytes());
        }
    }
    let buffer = device
        .create_buffer(&BufferDesc {
            label: None,
            size: size as u64,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let region = || TextureRegion {
        texture: &texture,
        mip: 0,
        origin: [0, 0, 0],
        extent: [WIDTH, HEIGHT, 1],
        aspect: TextureAspects::COLOR,
    };
    let fixture = common::ImmediateFixture {
        device,
        lane: device.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 982).unwrap();
    // SAFETY: all resources remain live through the returned completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin normalized readback"));
    let commands = run.commands();
    commands.write_buffer(&buffer, 0, &expected).unwrap();
    commands.write_texture(region(), layout, &expected).unwrap();
    commands
        .texture_to_buffer(region(), &buffer, layout)
        .unwrap();
    let ticket = commands.readback_buffer(&buffer, 0..size as u64).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish RGB10A2"),
    };
    device.wait_completion(&point).await.unwrap();
    // SAFETY: `point` covers upload, normalized transform-feedback readback, and staging readback.
    assert_eq!(
        unsafe { device.read_after(ticket, &point).await }
            .unwrap()
            .bytes,
        expected,
        "Rgb10a2Unorm"
    );
    wasm_bindgen_test::console_log!("EXECUTED Rgb10a2Unorm: exhaustive normalized 3D transfer");
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_exhaustive_normalized_feedback() {
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
    exhaustive_unorm16(&device, TextureFormat::R16Unorm, 1).await;
    exhaustive_unorm16(&device, TextureFormat::Rg16Unorm, 2).await;
    exhaustive_unorm16(&device, TextureFormat::Rgba16Unorm, 4).await;
    exhaustive_rgb10a2(&device).await;
}
