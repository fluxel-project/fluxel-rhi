//! Optional GLES timestamp query writes and completion-gated resolution.
use super::*;

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES context"]
fn gles_core_timestamp_query() {
    let d = block_on(open(
        GlInstance::gles_pbuffer([8, 8]).expect("EGL GLES context"),
    ));
    let Some(queries) = d.queries(QueryKind::Timestamp) else {
        eprintln!("SKIP: GLES timestamp extension unavailable or zero counter bits");
        return;
    };
    assert_eq!(queries.resolve_mode(), QueryResolveMode::AfterCompletion);
    let clock = queries.timestamp_clock(d.lanes()[0].id).unwrap();
    assert_eq!(clock.period_ns, 1.0);
    eprintln!("GLES timestamp available: {clock:?}");
    let set = queries
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 2,
        })
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
    queries.timestamp(encoder, &set, 0).unwrap();
    queries.timestamp(encoder, &set, 1).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let producer = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish producer"),
    };
    block_on(d.wait_completion(&producer)).unwrap();
    let plan = common::single_commands_plan(&fixture, 746).unwrap();
    // SAFETY: all resources remain live through the resolve completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin resolve commands"));
    let encoder = run.commands();
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
    eprintln!("timestamp results {first}, {second}");
    if first == u64::MAX || second == u64::MAX {
        eprintln!("timestamp disjoint sentinel observed");
        return;
    }
    assert_ne!(first, 0);
    if clock.valid_bits == 64 {
        assert!(second >= first);
    }
}
