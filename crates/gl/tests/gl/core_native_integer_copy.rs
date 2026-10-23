//! Bit-exact integer texture copies and PBO padding on native GL.
use super::*;

async fn integer_copy(d: &Device<Gl>) {
    let mut tested = 0;
    use TextureFormat::*;
    for (format, bpp) in [
        (R8Uint, 1),
        (Rg8Uint, 2),
        (Rgba8Uint, 4),
        (R8Sint, 1),
        (Rg8Sint, 2),
        (Rgba8Sint, 4),
        (R16Uint, 2),
        (Rg16Uint, 4),
        (Rgba16Uint, 8),
        (R16Sint, 2),
        (Rg16Sint, 4),
        (Rgba16Sint, 8),
        (R32Uint, 4),
        (Rg32Uint, 8),
        (Rgba32Uint, 16),
        (R32Sint, 4),
        (Rg32Sint, 8),
        (Rgba32Sint, 16),
        (Rgb10a2Uint, 4),
    ] {
        let stride = bpp * 2 + 3;
        let texture = match d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [2, 2, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        }) {
            Ok(v) => v,
            Err(Error::Unsupported(reason)) => {
                eprintln!("SKIP {format:?}: {reason}");
                continue;
            }
            Err(e) => panic!("{format:?}: {e:?}"),
        };
        tested += 1;
        let copied = d
            .create_texture(&TextureDesc {
                label: None,
                dimension: TextureDimension::D2,
                extent: [2, 2, 1],
                array_layers: 1,
                format,
                mip_levels: 1,
                samples: 1,
                usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
                view_formats: vec![],
            })
            .unwrap();
        let mut bytes = vec![0xadu8; 100];
        for row in 0..2 {
            for byte in 0..bpp * 2 {
                bytes[7 + row * stride + byte] = ((row * 37 + byte * 11) as u8).wrapping_add(1);
            }
        }
        let buffer = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 100,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let fixture = common::ImmediateFixture {
            device: &d,
            lane: d.lanes()[0].id,
        };
        let plan = common::single_commands_plan(&fixture, 913).unwrap();
        // SAFETY: resource lifetimes extend through the completion point.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
        let e = run.commands();
        e.write_buffer(&buffer, 0, &[0xad; 100]).unwrap();
        let region = || TextureRegion {
            texture: &texture,
            mip: 0,
            origin: [0, 0, 0],
            extent: [2, 2, 1],
            aspect: TextureAspects::COLOR,
        };
        let layout = || TexelCopyBufferLayout {
            offset: 7,
            bytes_per_row: stride as u32,
            rows_per_image: 2,
        };
        e.write_texture(region(), layout(), &bytes).unwrap();
        let copied_region = || TextureRegion {
            texture: &copied,
            mip: 0,
            origin: [0, 0, 0],
            extent: [2, 2, 1],
            aspect: TextureAspects::COLOR,
        };
        e.copy_texture(region(), copied_region()).unwrap();
        e.texture_to_buffer(copied_region(), &buffer, layout())
            .unwrap();
        let ticket = e.readback_buffer(&buffer, 0..100).unwrap();
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
        assert_eq!(actual, bytes, "{format:?} raw bits and untouched padding");
        eprintln!("PASS {format:?} offset=7 stride={stride}");
    }
    assert!(tested > 0, "no integer format supported");
}

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    use std::task::{Context, Poll, Wake, Waker};
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
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

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_integer_copy() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL PBO readback", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL context and device.
    let d = block_on(open(
        unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap(),
    ));
    block_on(integer_copy(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_integer_copy() {
    let d = block_on(open(GlInstance::gles_pbuffer([8, 8]).unwrap()));
    block_on(integer_copy(&d));
}
