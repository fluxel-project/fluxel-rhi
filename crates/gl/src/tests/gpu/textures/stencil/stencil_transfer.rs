//! Desktop standalone stencil transfers preserve one-byte texels and padding.
use super::*;

async fn stencil_transfer(d: &Device<Gl>) {
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [8, 8, 1],
        array_layers: 2,
        format: TextureFormat::Stencil8,
        mip_levels: 2,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    if !matches!(d.texture_support(&desc), Support::Supported(_)) {
        eprintln!("SKIP: standalone Stencil8 texture lacks runtime format facts");
        return;
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
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 481).unwrap();
    // SAFETY: device and both resources outlive the completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin stencil transfer"));
    let e = run.commands();
    e.write_buffer(&buffer, 0, &[0xa5; 128]).unwrap();
    let region = || TextureRegion {
        texture: &texture,
        mip: 1,
        origin: [1, 1, 1],
        extent: [2, 2, 1],
        aspect: TextureAspects::STENCIL,
    };
    let layout = |offset| TexelCopyBufferLayout {
        offset,
        bytes_per_row: 4,
        rows_per_image: 2,
    };
    let cpu = [0xee, 0xee, 0xee, 0xee, 3, 19, 0xee, 0xee, 77, 251];
    e.write_texture(region(), layout(4), &cpu).unwrap();
    e.texture_to_buffer(region(), &buffer, layout(36)).unwrap();
    let mut source = [0xceu8; 16];
    source[4..6].copy_from_slice(&[9, 17]);
    source[8..10].copy_from_slice(&[83, 201]);
    e.write_buffer(&buffer, 64, &source).unwrap();
    e.buffer_to_texture(&buffer, layout(68), region()).unwrap();
    e.texture_to_buffer(region(), &buffer, layout(100)).unwrap();
    let ticket = e.readback_buffer(&buffer, 0..128).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish stencil transfer"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: point covers all uploads and the readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for (offset, expected) in [(36, [3, 19, 77, 251]), (100, [9, 17, 83, 201])] {
        assert_eq!(&bytes[offset..offset + 2], &expected[..2]);
        assert_eq!(&bytes[offset + 4..offset + 6], &expected[2..]);
        assert_eq!(&bytes[offset - 2..offset], &[0xa5; 2]);
        assert_eq!(&bytes[offset + 2..offset + 4], &[0xa5; 2]);
        assert_eq!(&bytes[offset + 6..offset + 8], &[0xa5; 2]);
    }
    let plan = common::single_commands_plan(&fixture, 482).unwrap();
    // SAFETY: context and resources remain live for the rejected operation.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin invalid stencil transfer"));
    let wrong = TextureRegion {
        aspect: TextureAspects::DEPTH,
        ..region()
    };
    assert!(matches!(
        run.commands().write_texture(wrong, layout(4), &cpu),
        Err(Error::Unsupported(_)) | Err(Error::InvalidRange)
    ));
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a desktop WGL hardware context"]
fn wgl_core_stencil_transfer() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL stencil transfer", 1024, 768).unwrap()).unwrap();
    // SAFETY: window and context are kept alive on their owner thread.
    let instance = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let d = block_on(open(instance));
    block_on(stencil_transfer(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_stencil_transfer() {
    let d = block_on(open(
        GlInstance::gles_pbuffer([8, 8]).expect("EGL GLES pbuffer"),
    ));
    block_on(stencil_transfer(&d));
}
