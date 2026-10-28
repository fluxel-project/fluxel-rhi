//! Bit-preserving signed-normalized texture readback at duplicate endpoints.
use super::*;

async fn snorm_endpoint_readback(d: &Device<Gl>, gles: bool) {
    use TextureFormat::*;
    for (format, alias, bpp, word) in [
        (R8Snorm, R8Uint, 1usize, 1usize),
        (Rg8Snorm, R16Uint, 2, 1),
        (Rgba8Snorm, R32Uint, 4, 1),
        (R16Snorm, R16Uint, 2, 2),
        (Rg16Snorm, R32Uint, 4, 2),
        (Rgba16Snorm, Rg32Uint, 8, 2),
    ] {
        let desc = |format| TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 2,
            format,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        };
        if let Support::Unsupported(reason) = d.texture_support(&desc(format)) {
            assert!(gles, "{format:?} requires exact alias route: {reason}");
            eprintln!("SKIP {format:?}: {reason}");
            continue;
        }
        let integer = d.create_texture(&desc(alias)).unwrap();
        let snorm = d.create_texture(&desc(format)).unwrap();
        if gles {
            // SAFETY: the texture belongs to this live device/context.
            let native = unsafe { d.native_ref() };
            let texture = unsafe { snorm.native_ref() }.id;
            if !native
                .0
                .with_native("SNORM alias availability", |p| {
                    p.gles_exact_alias_for_test(texture)
                })
                .unwrap()
            {
                eprintln!("SKIP {format:?}: GLES exact image alias route unavailable");
                continue;
            }
        }
        let buffer = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 96,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let stride = bpp * 2 + 3;
        let mut bytes = vec![0xadu8; 96];
        for row in 0..2 {
            for x in 0..2 {
                for component in 0..bpp / word {
                    let at = 7 + row * stride + x * bpp + component * word;
                    let low = ((row + x + component) & 1) as u8;
                    if word == 1 {
                        bytes[at] = 0x80 + low;
                    } else {
                        bytes[at] = low;
                        bytes[at + 1] = 0x80;
                    }
                }
            }
        }
        let layout = || TexelCopyBufferLayout {
            offset: 7,
            bytes_per_row: stride as u32,
            rows_per_image: 2,
        };
        let region = |texture| TextureRegion {
            texture,
            mip: 1,
            origin: [1, 1, 1],
            extent: [2, 2, 1],
            aspect: TextureAspects::COLOR,
        };
        let fixture = common::ImmediateFixture {
            device: d,
            lane: d.lanes()[0].id,
        };
        let plan = common::single_commands_plan(&fixture, 991).unwrap();
        // SAFETY: all resources survive the completion point.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
        let e = run.commands();
        e.write_buffer(&buffer, 0, &[0xad; 96]).unwrap();
        e.write_texture(region(&integer), layout(), &bytes).unwrap();
        // SAFETY: both textures belong to this current context and are live;
        // the provider's test helper checks compatible image-size classes.
        let native = unsafe { d.native_ref() };
        let source = unsafe { integer.native_ref() }.id;
        let destination = unsafe { snorm.native_ref() }.id;
        native
            .0
            .with_native("seed SNORM endpoint bits", |p| {
                p.seed_exact_alias_for_test(source, destination)
            })
            .unwrap();
        e.texture_to_buffer(region(&snorm), &buffer, layout())
            .unwrap();
        let ticket = e.readback_buffer(&buffer, 0..96).unwrap();
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
        assert_eq!(actual, bytes, "{format:?} endpoint bits or padding");
        eprintln!("EXECUTED {format:?}: both negative endpoints, mip 1, layer 1");
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_snorm_endpoint_readback() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL SNORM endpoint readback", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the context.
    let d = block_on(open(
        unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap(),
    ));
    block_on(snorm_endpoint_readback(&d, false));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_snorm_endpoint_readback() {
    let d = block_on(open(
        GlInstance::gles_pbuffer([8, 8]).expect("EGL GLES pbuffer"),
    ));
    block_on(snorm_endpoint_readback(&d, true));
}
