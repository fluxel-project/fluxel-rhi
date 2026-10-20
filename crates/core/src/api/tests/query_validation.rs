//! Query domain and bounds validation before native calls.
use super::{type_contract_tests::NativeShared, *};
use std::cell::Cell;

thread_local! { static CALLS: Cell<u32> = const { Cell::new(0) }; }
fn called() {
    CALLS.with(|c| c.set(c.get() + 1));
}
fn calls() -> u32 {
    CALLS.with(Cell::get)
}

impl QueryBackend for NativeShared {
    fn query_supported(_: &Arc<()>, _: QueryKind) -> bool {
        true
    }
    fn create_query_set(_: &Arc<()>, _: &QueryDesc) -> Result<Arc<()>> {
        called();
        Ok(Arc::new(()))
    }
    fn write_timestamp(_: &mut (), _: &Arc<()>, _: u32) -> Result<()> {
        called();
        Ok(())
    }
    fn begin_occlusion(_: &mut (), _: &Arc<()>, _: u32) -> Result<()> {
        called();
        Ok(())
    }
    fn reset_queries(_: &mut (), _: &Arc<()>, _: Range<u32>) -> Result<()> {
        called();
        Ok(())
    }
    fn end_occlusion(_: &mut ()) -> Result<()> {
        called();
        Ok(())
    }
    fn begin_statistics(_: &mut (), _: &Arc<()>, _: u32) -> Result<()> {
        called();
        Ok(())
    }
    fn end_statistics(_: &mut ()) -> Result<()> {
        called();
        Ok(())
    }
    fn resolve_queries(_: &mut (), _: &Arc<()>, _: Range<u32>, _: &Arc<()>, _: u64) -> Result<()> {
        called();
        Ok(())
    }
}
fn device() -> Device<NativeShared> {
    let id = DeviceId::allocate().unwrap();
    // SAFETY: the mock native device belongs to this newly allocated ID.
    unsafe { Device::from_native(id, Arc::new(()), DeviceCapabilities::default()) }
}
fn encoder(device: &Device<NativeShared>) -> CommandEncoder<NativeShared> {
    CommandEncoder::new(
        device,
        LaneId {
            device: device.id(),
            ordinal: 0,
        },
    )
    .unwrap()
}
#[test]
fn query_domain_and_bounds_are_checked_before_backend() {
    CALLS.with(|c| c.set(0));
    let device = device();
    let other = self::device();
    let timestamps = device.queries(QueryKind::Timestamp).unwrap();
    let occlusions = device.queries(QueryKind::Occlusion).unwrap();
    let stats = device.queries(QueryKind::PipelineStatistics).unwrap();
    assert!(matches!(
        timestamps.create(&QueryDesc {
            kind: QueryKind::Occlusion,
            count: 1
        }),
        Err(Error::InvalidState)
    ));
    assert!(matches!(
        timestamps.create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 0
        }),
        Err(Error::InvalidRange)
    ));
    assert_eq!(calls(), 0);
    let ts = timestamps
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 2,
        })
        .unwrap();
    let occ = occlusions
        .create(&QueryDesc {
            kind: QueryKind::Occlusion,
            count: 1,
        })
        .unwrap();
    let foreign = other
        .queries(QueryKind::Timestamp)
        .unwrap()
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 1,
        })
        .unwrap();
    assert_eq!((ts.kind(), ts.count()), (QueryKind::Timestamp, 2));
    let mut enc = encoder(&device);
    let foreign_enc = &mut encoder(&other);
    // SAFETY: the mock buffer is owned by the corresponding mock device.
    let dst = unsafe { Buffer::from_native(device.id(), Arc::new(())) };
    let before = calls();
    assert!(matches!(
        timestamps.timestamp(&mut enc, &occ, 0),
        Err(Error::InvalidState)
    ));
    assert!(matches!(
        occlusions.timestamp(&mut enc, &occ, 0),
        Err(Error::InvalidState)
    ));
    assert!(matches!(
        timestamps.timestamp(&mut enc, &ts, 2),
        Err(Error::InvalidRange)
    ));
    assert!(matches!(
        timestamps.timestamp(&mut enc, &foreign, 0),
        Err(Error::WrongDevice)
    ));
    assert!(matches!(
        timestamps.timestamp(foreign_enc, &ts, 0),
        Err(Error::WrongDevice)
    ));
    assert!(matches!(stats.end_statistics(&mut enc), Ok(())));
    let before = before + 1;
    assert!(matches!(
        timestamps.end_statistics(&mut enc),
        Err(Error::InvalidState)
    ));
    assert!(matches!(
        timestamps.reset(&mut enc, &ts, 0..0),
        Err(Error::InvalidRange)
    ));
    assert!(matches!(
        timestamps.reset(&mut enc, &ts, 1..3),
        Err(Error::InvalidRange)
    ));
    assert!(matches!(
        timestamps.resolve(&mut enc, &ts, Range { start: 2, end: 1 }, &dst, 0),
        Err(Error::InvalidRange)
    ));
    assert!(matches!(
        occlusions.resolve(&mut enc, &ts, 0..1, &dst, 0),
        Err(Error::InvalidState)
    ));
    assert_eq!(calls(), before);
    timestamps.timestamp(&mut enc, &ts, 1).unwrap();
    timestamps.reset(&mut enc, &ts, 0..2).unwrap();
    timestamps.resolve(&mut enc, &ts, 0..2, &dst, 0).unwrap();
    assert_eq!(calls(), before + 3);
}

#[test]
fn raster_pass_rejects_invalid_query_metadata_before_backend() {
    let device = device();
    let other = self::device();
    let ts = device
        .queries(QueryKind::Timestamp)
        .unwrap()
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 2,
        })
        .unwrap();
    let occ = device
        .queries(QueryKind::Occlusion)
        .unwrap()
        .create(&QueryDesc {
            kind: QueryKind::Occlusion,
            count: 1,
        })
        .unwrap();
    let foreign = other
        .queries(QueryKind::Timestamp)
        .unwrap()
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 1,
        })
        .unwrap();
    let mut enc = encoder(&device);
    let mut desc = RasterPassDesc {
        label: None,
        colors: &[],
        depth_stencil: None,
        render_area: Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: Some(&ts),
        secondary_inheritance: None,
    };
    assert!(matches!(enc.begin_raster(&desc), Err(Error::InvalidState)));
    desc.occlusion_query_set = Some(&occ);
    desc.timestamp_writes = Some(PassTimestampWrites {
        set: &occ,
        begin: Some(0),
        end: None,
    });
    assert!(matches!(enc.begin_raster(&desc), Err(Error::InvalidState)));
    desc.timestamp_writes = Some(PassTimestampWrites {
        set: &ts,
        begin: Some(2),
        end: None,
    });
    assert!(matches!(enc.begin_raster(&desc), Err(Error::InvalidRange)));
    desc.timestamp_writes = Some(PassTimestampWrites {
        set: &ts,
        begin: None,
        end: Some(2),
    });
    assert!(matches!(enc.begin_raster(&desc), Err(Error::InvalidRange)));
    desc.timestamp_writes = Some(PassTimestampWrites {
        set: &foreign,
        begin: Some(0),
        end: None,
    });
    assert!(matches!(enc.begin_raster(&desc), Err(Error::WrongDevice)));
    desc.timestamp_writes = Some(PassTimestampWrites {
        set: &ts,
        begin: Some(0),
        end: Some(1),
    });
    enc.begin_raster(&desc).unwrap().finish().unwrap();
}
