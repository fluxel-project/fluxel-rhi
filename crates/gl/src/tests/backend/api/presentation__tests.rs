use super::{
    GlSurfaceLeaseBook, GlSurfaceSize, GlTextureDesc, GlTextureDimension, validate_publish_source,
};
use crate::backend::api::{
    ContextEpoch, ContextStamp, DeviceIdentity, GlError, GlExtent3d, GlFormat, GlTextureUsage,
    SurfaceImageId,
};

fn stamp() -> ContextStamp {
    ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL)
}

/// One acquisition at `width` x `height`, which is the extent a publish
/// source has to match.
fn lease(width: u32, height: u32) -> super::GlSurfaceLease {
    let mut book = GlSurfaceLeaseBook::new();
    book.acquire(
        SurfaceImageId::new(stamp(), 1, 1),
        GlSurfaceSize { width, height },
    )
    .expect("lease")
}

fn source(dimension: GlTextureDimension, extent: GlExtent3d, sample_count: u32) -> GlTextureDesc {
    GlTextureDesc {
        dimension,
        extent,
        mip_level_count: 1,
        sample_count,
        format: GlFormat::Rgba8Unorm,
        usage: GlTextureUsage::RENDER_ATTACHMENT,
    }
}

const OP: &str = "publish-surface-image";

#[test]
fn a_refused_lease_names_the_operation_that_was_attempted() {
    let mut book = GlSurfaceLeaseBook::new();
    let lease = book
        .acquire(
            SurfaceImageId::new(stamp(), 1, 1),
            GlSurfaceSize {
                width: 1,
                height: 1,
            },
        )
        .unwrap();
    book.consume("present-surface", lease)
        .expect("the first consume owns the lease");
    // A present and a publish are refused from the same lines, so the
    // second attempt is what the diagnostic has to name: reporting the
    // book's own fixed verb would send a caller looking at the wrong call.
    assert_eq!(
        book.consume("publish-surface-image", lease),
        Err(GlError::Validation {
            operation: "publish-surface-image",
            message: "surface acquire lease is stale or already consumed".into(),
        })
    );
}

#[test]
fn resize_invalidates_an_old_acquire_lease() {
    let s = stamp();
    let mut b = GlSurfaceLeaseBook::new();
    let lease = b
        .acquire(
            SurfaceImageId::new(s, 1, 1),
            GlSurfaceSize {
                width: 1,
                height: 1,
            },
        )
        .unwrap();
    b.invalidate_generation().unwrap();
    assert!(b.validate("present-surface", lease).is_err());
}

#[test]
fn a_presentation_source_must_be_a_single_sample_2d_texture_of_the_acquired_extent() {
    let plain = GlExtent3d {
        width: 1,
        height: 1,
        depth_or_layers: 1,
    };
    assert!(
        validate_publish_source(OP, source(GlTextureDimension::D2, plain, 1), lease(1, 1)).is_ok()
    );

    // Three refusals, each of which a driver would otherwise resolve by its
    // own choice of what to keep: a source that is the wrong size, one with
    // samples to reconcile, and one with layers to choose between.
    let wrong_size = GlExtent3d {
        width: 2,
        height: 2,
        depth_or_layers: 1,
    };
    let layered = GlExtent3d {
        width: 1,
        height: 1,
        depth_or_layers: 2,
    };
    for (descriptor, against) in [
        (source(GlTextureDimension::D2, wrong_size, 1), lease(1, 1)),
        (source(GlTextureDimension::D2, plain, 4), lease(1, 1)),
        (source(GlTextureDimension::D3, layered, 1), lease(1, 1)),
    ] {
        let refused = validate_publish_source(OP, descriptor, against)
            .expect_err("refused before any driver call");
        assert!(
            matches!(refused, GlError::Validation { operation, .. } if operation == OP),
            "expected a validation refusal naming the verb, got {refused:?}"
        );
    }
}
