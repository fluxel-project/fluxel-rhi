//! GLES shader-based exact depth/stencil buffer readback.
use super::*;

async fn gles_plane_readback(
    d: &Device<Gl>,
    format: TextureFormat,
    aspect: TextureAspects,
    values: &[u8],
    bpp: usize,
) {
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
    if !matches!(d.texture_support(&desc), Support::Supported(_)) {
        eprintln!("SKIP: {format:?} unavailable");
        return;
    }
    let texture = d.create_texture(&desc).unwrap();
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut input = vec![0x11; 32];
    let row = bpp * 2;
    input[4..4 + row].copy_from_slice(&values[..row]);
    input[20..20 + row].copy_from_slice(&values[row..]);
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 491).unwrap();
    // SAFETY: context and resources remain alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin GLES plane readback"));
    let e = run.commands();
    e.write_buffer(&buffer, 0, &[0xa5; 64]).unwrap();
    let region = || TextureRegion {
        texture: &texture,
        mip: 1,
        origin: [1, 1, 1],
        extent: [2, 2, 1],
        aspect,
    };
    e.write_texture(
        region(),
        TexelCopyBufferLayout {
            offset: 4,
            bytes_per_row: 16,
            rows_per_image: 2,
        },
        &input,
    )
    .unwrap();
    e.texture_to_buffer(
        region(),
        &buffer,
        TexelCopyBufferLayout {
            offset: 4,
            bytes_per_row: 16,
            rows_per_image: 2,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&buffer, 0..64).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish GLES plane readback"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: point covers texture readback.
    let actual = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    assert_eq!(&actual[4..4 + row], &values[..row]);
    assert_eq!(&actual[20..20 + row], &values[row..]);
    assert_eq!(&actual[..4], &[0xa5; 4]);
    assert!(actual[4 + row..20].iter().all(|x| *x == 0xa5));
    assert!(actual[20 + row..40].iter().all(|x| *x == 0xa5));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_plane_readback() {
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
        let w = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&w);
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
    if d.compute().is_none() {
        eprintln!("SKIP: ES 3.1 compute unavailable");
        return;
    }
    let d16 = [0u16, 1, 32768, 65535]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let d32 = [0.0f32, 0.125, 0.5, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    block_on(gles_plane_readback(
        &d,
        TextureFormat::Depth16Unorm,
        TextureAspects::DEPTH,
        &d16,
        2,
    ));
    block_on(gles_plane_readback(
        &d,
        TextureFormat::Depth32Float,
        TextureAspects::DEPTH,
        &d32,
        4,
    ));
    block_on(gles_plane_readback(
        &d,
        TextureFormat::Depth32FloatStencil8,
        TextureAspects::DEPTH,
        &d32,
        4,
    ));
    for format in [
        TextureFormat::Stencil8,
        TextureFormat::Depth24PlusStencil8,
        TextureFormat::Depth32FloatStencil8,
    ] {
        block_on(gles_plane_readback(
            &d,
            format,
            TextureAspects::STENCIL,
            &[0, 1, 128, 255],
            1,
        ));
    }
}
