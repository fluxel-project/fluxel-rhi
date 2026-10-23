//! Desktop timestamp query writes and ordered GPU-visible resolution.
use super::*;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_timestamp_query() {
    run_timestamp_query(false);
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_timestamp_query_without_query_buffer() {
    run_timestamp_query(true);
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
fn run_timestamp_query(fallback: bool) {
    crate::backend::gl::backend::spi::TEST_DISABLE_QUERY_BUFFER.with(|v| v.set(fallback));
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
    let window = Window::new(WindowConfig::new("GL timestamp", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the device and its current context.
    let d = block_on(open(
        unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap(),
    ));
    let queries = d
        .queries(QueryKind::Timestamp)
        .expect("desktop GL timestamp capability");
    assert_eq!(
        queries.resolve_mode(),
        if fallback {
            QueryResolveMode::AfterCompletion
        } else {
            QueryResolveMode::Ordered
        }
    );
    let clock = queries.timestamp_clock(d.lanes()[0].id).unwrap();
    assert_eq!(clock.period_ns, 1.0);
    let set = queries
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 2,
        })
        .unwrap();
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let view = d
        .create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 32,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: &d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 745).unwrap();
    // SAFETY: the device and resources survive the completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let encoder = run.commands();
    encoder.write_buffer(&buffer, 0, &[0xadu8; 32]).unwrap();
    queries.reset(encoder, &set, 0..2).unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
        store: StoreOp::Store,
    })];
    let pass = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            multiview_mask: None,
            timestamp_writes: Some(PassTimestampWrites {
                set: &set,
                begin: Some(0),
                end: Some(1),
            }),
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.finish().unwrap();
    if fallback {
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish producer"),
        };
        block_on(d.wait_completion(&point)).unwrap();
        let plan = common::single_commands_plan(&fixture, 747).unwrap();
        // SAFETY: the producer is complete and resources remain live.
        let mut resolve_session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut resolve_run = resolve_session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin resolve"));
        let e = resolve_run.commands();
        queries.resolve(e, &set, 0..2, &buffer, 8).unwrap();
        let ticket = e.readback_buffer(&buffer, 0..32).unwrap();
        assert!(matches!(
            resolve_run.close(),
            ImmediateRunOutcome::Issued(_)
        ));
        let point = match resolve_session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish resolve"),
        };
        block_on(d.wait_completion(&point)).unwrap();
        // SAFETY: completion covers the readback.
        let bytes = block_on(unsafe { d.read_after(ticket, &point) })
            .unwrap()
            .bytes;
        assert_eq!(&bytes[..8], &[0xad; 8]);
        assert_eq!(&bytes[24..], &[0xad; 8]);
        assert_ne!(u64::from_ne_bytes(bytes[8..16].try_into().unwrap()), 0);
        // A completed slot followed by an unwritten slot must reject the whole
        // resolve without changing any bytes in the destination.
        let partial = queries
            .create(&QueryDesc {
                kind: QueryKind::Timestamp,
                count: 2,
            })
            .unwrap();
        let plan = common::single_commands_plan(&fixture, 748).unwrap();
        // SAFETY: resources stay live until producer completion.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin partial producer"));
        let e = run.commands();
        e.write_buffer(&buffer, 0, &[0xadu8; 32]).unwrap();
        queries.reset(e, &partial, 0..2).unwrap();
        queries.timestamp(e, &partial, 0).unwrap();
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish partial producer"),
        };
        block_on(d.wait_completion(&point)).unwrap();
        let plan = common::single_commands_plan(&fixture, 749).unwrap();
        // SAFETY: resources remain live while this run is rejected.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin invalid resolve"));
        assert!(matches!(
            queries.resolve(run.commands(), &partial, 0..2, &buffer, 8),
            Err(Error::InvalidState)
        ));
        assert!(matches!(run.close(), ImmediateRunOutcome::Failed { .. }));
        assert!(matches!(
            session.finish(),
            ImmediateFinishOutcome::AlreadyReported
        ));
        let plan = common::single_commands_plan(&fixture, 750).unwrap();
        // SAFETY: the destination remains live through readback completion.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin sentinel readback"));
        let ticket = run.commands().readback_buffer(&buffer, 0..32).unwrap();
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish sentinel readback"),
        };
        block_on(d.wait_completion(&point)).unwrap();
        // SAFETY: the completion point covers the readback.
        let bytes = block_on(unsafe { d.read_after(ticket, &point) })
            .unwrap()
            .bytes;
        assert_eq!(bytes.as_slice(), &[0xadu8; 32]);
        crate::backend::gl::backend::spi::TEST_DISABLE_QUERY_BUFFER.with(|v| v.set(false));
        return;
    }
    queries.resolve(encoder, &set, 0..2, &buffer, 8).unwrap();
    let ticket = encoder.readback_buffer(&buffer, 0..32).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    block_on(d.wait_completion(&point)).unwrap();
    // SAFETY: completion covers query resolution and buffer readback.
    let bytes = block_on(unsafe { d.read_after(ticket, &point) })
        .unwrap()
        .bytes;
    assert_eq!(&bytes[..8], &[0xad; 8]);
    assert_eq!(&bytes[24..], &[0xad; 8]);
    let first = u64::from_ne_bytes(bytes[8..16].try_into().unwrap());
    let second = u64::from_ne_bytes(bytes[16..24].try_into().unwrap());
    assert_ne!(first, 0);
    if clock.valid_bits == 64 {
        assert!(second >= first);
    }

    // The same slots cannot be recorded twice without a logical reset. The
    // failed begin poisons this run; it must not issue a timestamp or clear.
    let plan = common::single_commands_plan(&fixture, 746).unwrap();
    // SAFETY: the same live objects remain owned through the failed run.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin invalid producer"));
    let e = run.commands();
    queries.reset(e, &set, 0..1).unwrap();
    let invalid = e.begin_raster(&RasterPassDesc {
        label: None,
        colors: &colors,
        depth_stencil: None,
        render_area: Rect {
            x: 0,
            y: 0,
            width: 8,
            height: 8,
        },
        multiview_mask: None,
        timestamp_writes: Some(PassTimestampWrites {
            set: &set,
            begin: Some(0),
            end: Some(1),
        }),
        occlusion_query_set: None,
        secondary_inheritance: None,
    });
    assert!(matches!(invalid, Err(Error::InvalidState)));
    drop(invalid);
    assert!(matches!(run.close(), ImmediateRunOutcome::Failed { .. }));
    assert!(matches!(
        session.finish(),
        ImmediateFinishOutcome::AlreadyReported
    ));
    let plan = common::single_commands_plan(&fixture, 747).unwrap();
    // SAFETY: the query set remains live through this run's close.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin preflight check"));
    // Slot zero remains unwritten when slot one fails preflight.
    queries.timestamp(run.commands(), &set, 0).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    assert!(matches!(
        session.finish(),
        ImmediateFinishOutcome::Issued(_)
    ));
}
