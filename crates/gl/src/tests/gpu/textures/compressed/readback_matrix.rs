//! Exact compressed-block copies across the Core texture-to-buffer contract.
use super::*;

async fn compressed_readback_matrix(d: &Device<Gl>, expect_no_copy: bool, gles: bool) {
    let small = [0xf0, 0, 0, 1, 0x44, 0x11, 0x55, 0x11];
    let mut large = vec![0x80, 0, 0, 0, 0, 0, 0, 0];
    large.extend_from_slice(&small);
    let bc6 = (3u128 | (450u128 << 5) | (400u128 << 15) | (450u128 << 35) | (400u128 << 45))
        .to_le_bytes();
    let bc7 = ((1u128 << 6) | (127u128 << 7) | (127u128 << 14)).to_le_bytes();
    let astc_ldr = [
        0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff,
    ];
    let astc_hdr = [
        0xfc, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0x3c, 0, 0, 0, 0, 0, 0x3c,
    ];
    let cases: Vec<(TextureFormat, Vec<u8>)> = vec![
        (TextureFormat::Bc1RgbaUnorm, small.to_vec()),
        (TextureFormat::Bc1RgbaUnormSrgb, small.to_vec()),
        (TextureFormat::Bc2RgbaUnorm, large.clone()),
        (TextureFormat::Bc2RgbaUnormSrgb, large.clone()),
        (TextureFormat::Bc3RgbaUnorm, large.clone()),
        (TextureFormat::Bc3RgbaUnormSrgb, large.clone()),
        (TextureFormat::Bc4RUnorm, small.to_vec()),
        (TextureFormat::Bc5RgUnorm, large.clone()),
        (TextureFormat::Bc6hRgbUfloat, bc6.to_vec()),
        (TextureFormat::Bc7RgbaUnorm, bc7.to_vec()),
        (TextureFormat::Etc2Rgb8Unorm, small.to_vec()),
        (TextureFormat::Etc2Rgba8Unorm, large.clone()),
        (TextureFormat::EacR11Unorm, small.to_vec()),
        (TextureFormat::EacRg11Unorm, large),
        (
            TextureFormat::Astc {
                block: [4, 4],
                srgb: false,
                hdr: false,
            },
            astc_ldr.to_vec(),
        ),
        (
            TextureFormat::Astc {
                block: [4, 4],
                srgb: false,
                hdr: true,
            },
            astc_hdr.to_vec(),
        ),
    ];
    let mut executed = 0;
    for (format, block) in cases {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: if expect_no_copy { 1 } else { 2 },
            format,
            mip_levels: if expect_no_copy { 1 } else { 2 },
            samples: 1,
            usage: TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
            view_formats: vec![],
        };
        let reason = match d.texture_support(&desc) {
            Support::Supported(_) => None,
            Support::Unsupported(reason) => Some(reason),
        };
        if expect_no_copy {
            let upload = TextureDesc {
                usage: TextureUsage::COPY_DST,
                ..desc.clone()
            };
            let upload_available = matches!(d.texture_support(&upload), Support::Supported(_));
            let reason = reason.unwrap_or_else(|| {
                panic!("WebGL compressed COPY_SRC must be rejected: {format:?}")
            });
            let message = if upload_available {
                format!("REJECT COPY_SRC {format:?}: {reason}")
            } else {
                format!(
                    "SKIP {format:?}: compressed upload unavailable; COPY_SRC rejected ({reason})"
                )
            };
            eprintln!("{message}");
            #[cfg(target_arch = "wasm32")]
            web_sys::console::log_1(&message.into());
            continue;
        }
        if let Some(reason) = reason {
            eprintln!("SKIP {format:?}: {reason}");
            continue;
        }
        let texture = d.create_texture(&desc).unwrap();
        #[cfg(feature = "native-gles-egl")]
        if gles {
            // SAFETY: the texture and device belong to this current context.
            let native = unsafe { d.native_ref() };
            let texture_id = unsafe { texture.native_ref() }.id;
            if !native
                .0
                .with_native("compressed alias fixture", |p| {
                    p.compressed_alias_for_test(texture_id)
                })
                .unwrap()
            {
                eprintln!("SKIP {format:?}: compressed block raw readback probe failed");
                continue;
            }
        }
        let _ = gles;
        let buffer = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 96,
                usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let fixture = common::ImmediateFixture {
            device: d,
            lane: d.lanes()[0].id,
        };
        let plan = common::single_commands_plan(&fixture, 231).unwrap();
        // SAFETY: all resources and the context outlive submission and completion.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
        let encoder = run.commands();
        for mip in 0..2u32 {
            for layer in 0..2u32 {
                let n = if mip == 0 { 8 } else { 4 };
                let count = if mip == 0 { 4 } else { 1 };
                let encoded = block.repeat(count);
                encoder
                    .write_texture(
                        TextureRegion {
                            texture: &texture,
                            mip,
                            origin: [0, 0, layer],
                            extent: [n, n, 1],
                            aspect: TextureAspects::COLOR,
                        },
                        TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: (if mip == 0 { 2 } else { 1 }) * block.len() as u32,
                            rows_per_image: if mip == 0 { 2 } else { 1 },
                        },
                        &encoded,
                    )
                    .unwrap();
            }
        }
        encoder.write_buffer(&buffer, 0, &[0xcc; 96]).unwrap();
        for (mip, origin, extent, offset, pitch) in [
            (0, [4, 0, 1], [4, 8, 1], 8, 32),
            (1, [0, 0, 0], [4, 4, 1], 72, 16),
        ] {
            encoder
                .texture_to_buffer(
                    TextureRegion {
                        texture: &texture,
                        mip,
                        origin,
                        extent,
                        aspect: TextureAspects::COLOR,
                    },
                    &buffer,
                    TexelCopyBufferLayout {
                        offset,
                        bytes_per_row: pitch,
                        rows_per_image: if mip == 0 { 2 } else { 1 },
                    },
                )
                .unwrap();
        }
        let ticket = encoder.readback_buffer(&buffer, 0..96).unwrap();
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish"),
        };
        d.wait_completion(&point).await.unwrap();
        // SAFETY: completion covers both compressed writes and all reads.
        let actual = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        let mut expected = [0xcc; 96];
        for offset in [8, 40, 72] {
            expected[offset..offset + block.len()].copy_from_slice(&block);
        }
        assert_eq!(
            actual.as_slice(),
            expected,
            "raw compressed blocks for {format:?}"
        );
        executed += 1;
        eprintln!(
            "EXECUTED {format:?}: {}-byte blocks, mip/layer/subregion/padding",
            block.len()
        );
    }
    if executed == 0 {
        eprintln!("SKIP compressed readback: no format passed the runtime capability checks");
    } else {
        eprintln!("EXECUTED compressed readback formats: {executed}");
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_compressed_readback_matrix() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(compressed_readback_matrix(&d, false, false));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_compressed_readback_matrix() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(compressed_readback_matrix(&d, false, true));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_compressed_readback_matrix() {
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
    compressed_readback_matrix(&d, true, false).await;
}
