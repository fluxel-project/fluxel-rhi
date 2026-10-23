//! GLES bit-preserving float and packed image-class alias readback.
use super::*;

async fn float_alias_readback(d: &Device<Gl>) {
    use TextureFormat::*;
    for (format, alias, bpp, word) in [
        (R16Float, R16Uint, 2usize, 2usize),
        (Rg16Float, R32Uint, 4, 2),
        (Rgba16Float, Rg32Uint, 8, 2),
        (R32Float, R32Uint, 4, 4),
        (Rg32Float, Rg32Uint, 8, 4),
        (Rgba32Float, Rgba32Uint, 16, 4),
        (Rg11b10Ufloat, R32Uint, 4, 0),
        (Rgb10a2Unorm, R32Uint, 4, 0),
        (Rgb9e5Ufloat, R32Uint, 4, 0),
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
            eprintln!("SKIP {format:?}: {reason}");
            continue;
        }
        let integer = d.create_texture(&desc(alias)).unwrap();
        let snorm = d.create_texture(&desc(format)).unwrap();
        {
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
                size: 192,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let stride = bpp * 2 + 3;
        let mut bytes = vec![0xadu8; 192];
        for row in 0..2 {
            for x in 0..2 {
                let at = 7 + row * stride + x * bpp;
                if word == 0 {
                    let packed = [0u32, 0xffff_ffff, 0x8a01_0080, 0x7f80_0001][row * 2 + x];
                    bytes[at..at + 4].copy_from_slice(&packed.to_le_bytes());
                } else {
                    for component in 0..bpp / word {
                        let index = (row * 2 + x + component) % 4;
                        if word == 2 {
                            let bits = [0x8000u16, 0x7e01, 0x3c00, 0x0001][index];
                            bytes[at + component * word..at + (component + 1) * word]
                                .copy_from_slice(&bits.to_le_bytes());
                        } else {
                            let bits =
                                [0x8000_0000u32, 0x7fc0_0001, 0x3f80_0000, 0x0000_0001][index];
                            bytes[at + component * word..at + (component + 1) * word]
                                .copy_from_slice(&bits.to_le_bytes());
                        }
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
        e.write_buffer(&buffer, 0, &[0xad; 192]).unwrap();
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
        let ticket = e.readback_buffer(&buffer, 0..192).unwrap();
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
        eprintln!("EXECUTED {format:?}: raw floating/packed patterns, mip 1, layer 1");
    }
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_float_alias_readback() {
    use std::{
        future::Future,
        task::{Context, Poll, Wake, Waker},
    };
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    fn block_on<F: Future>(f: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }
    let d = block_on(open(
        GlInstance::gles_pbuffer([8, 8]).expect("EGL GLES pbuffer"),
    ));
    block_on(float_alias_readback(&d));
}
