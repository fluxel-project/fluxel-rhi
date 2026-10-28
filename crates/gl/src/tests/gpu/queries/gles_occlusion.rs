//! GLES Core occlusion resolution after explicit producer completion.
use super::*;

async fn gles_occlusion_after_completion(d: Device<Gl>) {
    let queries = d.queries(QueryKind::Occlusion).unwrap();
    assert_eq!(queries.resolve_mode(), QueryResolveMode::AfterCompletion);
    let set = queries
        .create(&QueryDesc {
            kind: QueryKind::Occlusion,
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
            size: 24,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: &d,
        lane: d.lanes()[0].id,
    };
    // Abort an active query in this very context. The following independent
    // producer must be able to start another query without reopening it.
    let abandoned = queries
        .create(&QueryDesc {
            kind: QueryKind::Occlusion,
            count: 1,
        })
        .unwrap();
    let plan = common::single_commands_plan(&fixture, 750).unwrap();
    // SAFETY: the device, attachment and query remain live until this run closes.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin abandoned producer"));
    let e = run.commands();
    queries.reset(e, &abandoned, 0..1).unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
        store: StoreOp::Store,
    })];
    let mut pass = e
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
            timestamp_writes: None,
            occlusion_query_set: Some(&abandoned),
            secondary_inheritance: None,
        })
        .unwrap();
    queries.begin_occlusion(&mut pass, &abandoned, 0).unwrap();
    drop(pass);
    assert!(matches!(run.close(), ImmediateRunOutcome::Failed { .. }));
    assert!(matches!(
        session.finish(),
        ImmediateFinishOutcome::AlreadyReported
    ));
    drop(abandoned);
    let plan = common::single_commands_plan(&fixture, 751).unwrap();
    // SAFETY: all resources remain alive until completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin producer"));
    let e = run.commands();
    e.write_buffer(&buffer, 0, &[0xad; 24]).unwrap();
    queries.reset(e, &set, 0..2).unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
        store: StoreOp::Store,
    })];
    let mut pass = e
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
            timestamp_writes: None,
            occlusion_query_set: Some(&set),
            secondary_inheritance: None,
        })
        .unwrap();
    queries.begin_occlusion(&mut pass, &set, 0).unwrap();
    queries.end_occlusion(&mut pass).unwrap();
    pass.finish().unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let producer = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish producer"),
    };
    d.wait_completion(&producer).await.unwrap();
    let plan = common::single_commands_plan(&fixture, 753).unwrap();
    // SAFETY: the producer has completed and resources remain live.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin invalid resolve"));
    assert!(matches!(
        queries.resolve(run.commands(), &set, 0..2, &buffer, 0),
        Err(Error::InvalidState)
    ));
    assert!(matches!(run.close(), ImmediateRunOutcome::Failed { .. }));
    assert!(matches!(
        session.finish(),
        ImmediateFinishOutcome::AlreadyReported
    ));
    let plan = common::single_commands_plan(&fixture, 754).unwrap();
    // SAFETY: the destination remains live until readback completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin sentinel readback"));
    let ticket = run.commands().readback_buffer(&buffer, 0..24).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish sentinel readback"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers the readback.
    let sentinel = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    assert_eq!(sentinel.as_slice(), &[0xad; 24]);
    let plan = common::single_commands_plan(&fixture, 752).unwrap();
    // SAFETY: the same live resources remain retained through consumer completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin consumer"));
    let e = run.commands();
    queries.resolve(e, &set, 0..1, &buffer, 8).unwrap();
    let ticket = e.readback_buffer(&buffer, 0..24).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish consumer"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: point covers the query-result copy and buffer readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    assert_eq!(&bytes[..8], &[0xad; 8]);
    assert_eq!(u64::from_ne_bytes(bytes[8..16].try_into().unwrap()), 0);
    assert_eq!(&bytes[16..], &[0xad; 8]);
}

#[test]
#[ignore = "requires an EGL GLES context"]
fn gles_core_occlusion_after_completion() {
    let d = block_on(open(
        GlInstance::gles_pbuffer([8, 8]).expect("EGL GLES context"),
    ));
    block_on(gles_occlusion_after_completion(d));
}
