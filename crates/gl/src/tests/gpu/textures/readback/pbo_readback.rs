//! PBO readback preserves Core row order and all untouched buffer padding.
use super::*;

async fn readback_volume(d: &Device<Gl>, dimension: TextureDimension, format: TextureFormat) {
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension,
            extent: if dimension == TextureDimension::D3 {
                [8, 8, 8]
            } else {
                [8, 8, 1]
            },
            array_layers: if dimension == TextureDimension::D3 {
                1
            } else {
                4
            },
            format,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let create_buffer = || {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 160,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let upload = create_buffer();
    let result = create_buffer();
    let mut initial = vec![0xadu8; 160];
    for layer in 0..2usize {
        for row in 0..2usize {
            for x in 0..2usize {
                let offset = 16 + layer * 48 + row * 16 + x * 4;
                initial[offset..offset + 4].copy_from_slice(&[
                    10 + layer as u8 * 60,
                    20 + row as u8 * 50,
                    30 + x as u8 * 40,
                    255,
                ]);
            }
        }
    }
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 731).unwrap();
    // SAFETY: device and resources remain live through the completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    e.write_buffer(&upload, 0, &initial).unwrap();
    e.write_buffer(&result, 0, &[0xadu8; 160]).unwrap();
    let layout = || TexelCopyBufferLayout {
        offset: 16,
        bytes_per_row: 16,
        rows_per_image: 3,
    };
    let region = || TextureRegion {
        texture: &texture,
        mip: 1,
        origin: [1, 1, 1],
        extent: [2, 2, 2],
        aspect: TextureAspects::COLOR,
    };
    e.buffer_to_texture(&upload, layout(), region()).unwrap();
    e.texture_to_buffer(region(), &result, layout()).unwrap();
    let ticket = e.readback_buffer(&result, 0..160).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: this completion point covers every copy and the readback.
    let actual = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    assert_eq!(
        actual, initial,
        "{dimension:?}: incorrect data, row order, or padding"
    );
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_pbo_readback_volume() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL PBO readback", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL context and device.
    let d = block_on(open(
        unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap(),
    ));
    block_on(readback_volume(
        &d,
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
    ));
    block_on(readback_volume(
        &d,
        TextureDimension::D3,
        TextureFormat::Rgba8Unorm,
    ));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_pbo_readback_volume() {
    let d = block_on(open(GlInstance::gles_pbuffer([8, 8]).unwrap()));
    block_on(readback_volume(
        &d,
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
    ));
    block_on(readback_volume(
        &d,
        TextureDimension::D3,
        TextureFormat::Rgba8Unorm,
    ));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_pbo_readback_volume() {
    let d = webgl2_device([8, 8]).await;
    for format in [TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb] {
        readback_volume(&d, TextureDimension::D2, format).await;
        readback_volume(&d, TextureDimension::D3, format).await;
    }
}
