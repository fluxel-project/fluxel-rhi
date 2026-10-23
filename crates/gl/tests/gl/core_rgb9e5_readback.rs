//! Desktop RGB9E5 native image copy and whole-mip readback.
use super::*;

pub(super) async fn rgb9e5_readback(d: &Device<Gl>) {
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [8, 8, 1],
        array_layers: 2,
        format: TextureFormat::Rgb9e5Ufloat,
        mip_levels: 2,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    if let Support::Unsupported(reason) = d.texture_support(&desc) {
        eprintln!("SKIP RGB9E5: {reason}");
        return;
    }
    let src = d.create_texture(&desc).unwrap();
    let dst = d.create_texture(&desc).unwrap();
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 96,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    // Include distinct shared-exponent encodings of the same color, so a
    // native value conversion cannot silently replace an exact byte transfer.
    let pixels = [0x84020100u32, 0x8a010080, 0xf8000000, 0];
    let mut bytes = [0xadu8; 96];
    for row in 0..2 {
        for x in 0..2 {
            let at = 7 + row * 13 + x * 4;
            bytes[at..at + 4].copy_from_slice(&pixels[row * 2 + x].to_le_bytes());
        }
    }
    let layout = TexelCopyBufferLayout {
        offset: 7,
        bytes_per_row: 13,
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
    let plan = common::single_commands_plan(&fixture, 515).unwrap();
    // SAFETY: context and resources live until completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    let e = run.commands();
    e.write_buffer(&buffer, 0, &[0xad; 96]).unwrap();
    e.write_texture(region(&src), layout, &bytes).unwrap();
    e.copy_texture(region(&src), region(&dst)).unwrap();
    e.texture_to_buffer(region(&dst), &buffer, layout).unwrap();
    let ticket = e.readback_buffer(&buffer, 0..96).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers native copy and readback.
    let actual = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    assert_eq!(actual, bytes, "RGB9E5 packed pixels and padding");
    eprintln!("EXECUTED RGB9E5 mip 1 layer 1 nonzero x/y image copy and padded readback");
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_rgb9e5_readback() {
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
    let window = Window::new(WindowConfig::new("GL RGB9E5 readback", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the context and resources.
    let d = block_on(open(
        unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap(),
    ));
    block_on(rgb9e5_readback(&d));
}
