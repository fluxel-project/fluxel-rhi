//! Direct desktop color subimage PBO copies, including non-attachment textures.
use super::*;

async fn direct_color_readback(d: &Device<Gl>, legacy: bool) {
    use TextureFormat::*;
    // SAFETY: this fixture owns the current context and keeps the device live.
    let native = unsafe { d.native_ref() };
    let (available, start_count) = native
        .0
        .with_native("direct readback fixture", |p| {
            Ok(if legacy {
                p.legacy_color_readback_test_state()
            } else {
                p.direct_color_readback_test_state()
            })
        })
        .unwrap();
    assert!(
        available,
        "fixture needs the actual desktop entry point and capability"
    );
    // All values are finite, exactly representable, and avoid the duplicate
    // negative endpoint encodings of signed normalized formats.
    let cases: &[(TextureFormat, &[u8])] = &[
        (Rgba8Unorm, &[1, 2, 3, 4]),
        (Rgba8UnormSrgb, &[1, 2, 3, 4]),
        (R8Uint, &[1]),
        (Rg8Uint, &[1, 2]),
        (Rgba8Uint, &[1, 2, 3, 4]),
        (R8Sint, &[1]),
        (Rg8Sint, &[1, 2]),
        (Rgba8Sint, &[1, 2, 3, 4]),
        (R16Uint, &[1, 2]),
        (Rg16Uint, &[1, 2, 3, 4]),
        (Rgba16Uint, &[1, 2, 3, 4, 5, 6, 7, 8]),
        (R16Sint, &[1, 2]),
        (Rg16Sint, &[1, 2, 3, 4]),
        (Rgba16Sint, &[1, 2, 3, 4, 5, 6, 7, 8]),
        (R32Uint, &[1, 2, 3, 4]),
        (Rg32Uint, &[1, 2, 3, 4, 5, 6, 7, 8]),
        (
            Rgba32Uint,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ),
        (R32Sint, &[1, 2, 3, 4]),
        (Rg32Sint, &[1, 2, 3, 4, 5, 6, 7, 8]),
        (
            Rgba32Sint,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ),
        (Rgb10a2Uint, &[0x00, 0x02, 0x08, 0xd0]),
        (R8Unorm, &[0x40]),
        (Rg8Unorm, &[0x40, 0xc0]),
        (R8Snorm, &[0xc0]),
        (Rg8Snorm, &[0xc0, 0x40]),
        (Rgba8Snorm, &[0xc0, 0x40, 0x7f, 0]),
        (R16Unorm, &[0x00, 0x40]),
        (Rg16Unorm, &[0x00, 0x40, 0x00, 0xc0]),
        (Rgba16Unorm, &[0x00, 0x40, 0x00, 0xc0, 0xff, 0xff, 0, 0]),
        (R16Snorm, &[0x00, 0xc0]),
        (Rg16Snorm, &[0x00, 0xc0, 0x00, 0x40]),
        (Rgba16Snorm, &[0x00, 0xc0, 0x00, 0x40, 0xff, 0x7f, 0, 0]),
        (R16Float, &[0x00, 0xb8]),
        (Rg16Float, &[0x00, 0xb8, 0x00, 0x3c]),
        (Rgba16Float, &[0x00, 0xb8, 0x00, 0x3c, 0x00, 0x40, 0, 0]),
        (R32Float, &[0, 0, 0, 0xbf]),
        (Rg32Float, &[0, 0, 0, 0xbf, 0, 0, 0x80, 0x3f]),
        (
            Rgba32Float,
            &[0, 0, 0, 0xbf, 0, 0, 0x80, 0x3f, 0, 0, 0, 0x40, 0, 0, 0, 0],
        ),
        (Rgb10a2Unorm, &[0x00, 0x02, 0x08, 0xd0]),
        (Rg11b10Ufloat, &[0xc0, 0x03, 0x1e, 0x78]),
    ];
    let mut tested = 0;
    for &(format, pixel) in cases {
        let desc = TextureDesc {
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
        if let Support::Unsupported(reason) = d.texture_support(&desc) {
            eprintln!("SKIP {format:?}: {reason}");
            continue;
        }
        let texture = d.create_texture(&desc).unwrap();
        let buffer = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 128,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let bpp = pixel.len();
        let stride = bpp * 2 + 3;
        let mut source = vec![0xadu8; 128];
        for row in 0..2 {
            for x in 0..2 {
                let at = 7 + row * stride + x * bpp;
                if row == 0 && x == 1 {
                    source[at..at + bpp].fill(0);
                } else {
                    source[at..at + bpp].copy_from_slice(pixel);
                }
            }
        }
        let layout = TexelCopyBufferLayout {
            offset: 7,
            bytes_per_row: stride as u32,
            rows_per_image: 2,
        };
        let region = || TextureRegion {
            texture: &texture,
            mip: 1,
            origin: [1, 1, 1],
            extent: [2, 2, 1],
            aspect: TextureAspects::COLOR,
        };
        let fixture = common::ImmediateFixture {
            device: d,
            lane: d.lanes()[0].id,
        };
        let plan = common::single_commands_plan(&fixture, 914).unwrap();
        // SAFETY: the context and resources live through completion.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
        let e = run.commands();
        e.write_buffer(&buffer, 0, &[0xad; 128]).unwrap();
        e.write_texture(region(), layout, &source).unwrap();
        e.texture_to_buffer(region(), &buffer, layout)
            .unwrap_or_else(|err| panic!("{format:?}: {err:?}"));
        let ticket = e.readback_buffer(&buffer, 0..128).unwrap();
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish"),
        };
        d.wait_completion(&point).await.unwrap();
        // SAFETY: completion covers the copy and readback.
        let actual = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        assert_eq!(actual, source, "{format:?}: value or padding mismatch");
        tested += 1;
        eprintln!("EXECUTED {format:?}: mip 1, layer 1, x/y 1, offset 7, stride {stride}");
    }
    assert!(tested > 0, "no color readback format supported");
    let (_, end_count) = native
        .0
        .with_native("direct readback count", |p| {
            Ok(if legacy {
                p.legacy_color_readback_test_state()
            } else {
                p.direct_color_readback_test_state()
            })
        })
        .unwrap();
    assert_eq!(
        end_count - start_count,
        tested,
        "all admitted cases must take the direct route"
    );
    for (dimension, extent, origin, copy_extent) in [
        (TextureDimension::D1, [8, 1, 1], [1, 0, 0], [2, 1, 1]),
        (TextureDimension::D3, [8, 8, 4], [1, 1, 1], [2, 2, 1]),
    ] {
        let texture = d
            .create_texture(&TextureDesc {
                label: None,
                dimension,
                extent,
                array_layers: 1,
                format: Rgba8Unorm,
                mip_levels: 2,
                samples: 1,
                usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
                view_formats: vec![],
            })
            .unwrap();
        let buffer = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 64,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let mut bytes = vec![0xadu8; 64];
        for row in 0..copy_extent[1] as usize {
            for i in 0..8 {
                bytes[7 + row * 11 + i] = (i + row * 13 + 1) as u8;
            }
        }
        let layout = || TexelCopyBufferLayout {
            offset: 7,
            bytes_per_row: 11,
            rows_per_image: copy_extent[1],
        };
        let region = || TextureRegion {
            texture: &texture,
            mip: 1,
            origin,
            extent: copy_extent,
            aspect: TextureAspects::COLOR,
        };
        let fixture = common::ImmediateFixture {
            device: d,
            lane: d.lanes()[0].id,
        };
        let plan = common::single_commands_plan(&fixture, 915).unwrap();
        // SAFETY: all resources remain live through completion.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin dimensional copy"));
        let e = run.commands();
        e.write_buffer(&buffer, 0, &[0xad; 64]).unwrap();
        e.write_texture(region(), layout(), &bytes).unwrap();
        e.texture_to_buffer(region(), &buffer, layout()).unwrap();
        let ticket = e.readback_buffer(&buffer, 0..64).unwrap();
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish dimensional copy"),
        };
        d.wait_completion(&point).await.unwrap();
        // SAFETY: completion covers the readback.
        let actual = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        assert_eq!(actual, bytes, "{dimension:?} subregion/padding");
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_direct_color_readback() {
    use fluxel_host::{Window, WindowConfig};
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
    let window =
        Window::new(WindowConfig::new("GL color readback fixture", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the context and its resources.
    let d = block_on(open(
        unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap(),
    ));
    block_on(direct_color_readback(&d, false));
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_legacy_color_readback() {
    use fluxel_host::{Window, WindowConfig};
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
    let window =
        Window::new(WindowConfig::new("GL legacy color readback", 1024, 768).unwrap()).unwrap();
    // SAFETY: window outlives the context.
    let d = block_on(open(
        unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap(),
    ));
    struct LegacyGuard<'a>(&'a Device<Gl>);
    impl Drop for LegacyGuard<'_> {
        fn drop(&mut self) {
            // SAFETY: the guard drops before the device on the context thread.
            let native = unsafe { self.0.native_ref() };
            let _ = native.0.with_native("restore direct route", |p| {
                p.force_legacy_color_readback_test(false);
                Ok(())
            });
        }
    }
    // SAFETY: device is on its current context thread.
    let native = unsafe { d.native_ref() };
    native
        .0
        .with_native("force legacy route", |p| {
            p.force_legacy_color_readback_test(true);
            Ok(())
        })
        .unwrap();
    let _guard = LegacyGuard(&d);
    block_on(direct_color_readback(&d, true));
}
