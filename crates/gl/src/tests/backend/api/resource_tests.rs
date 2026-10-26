//! Tests for the private resource.rs contract.
use super::*;

fn buffer(usage: GlBufferUsage) -> GlBufferDesc {
    GlBufferDesc { size: 64, usage }
}

#[test]
fn consumption_roles_combine_because_the_gl_family_stores_untyped_bytes() {
    // Vertex, index, uniform, copy-source, and copy-destination share one
    // buffer legally; the layer validates the declared role at each bind
    // site instead of forbidding the combination here.
    let usage = GlBufferUsage::VERTEX
        | GlBufferUsage::INDEX
        | GlBufferUsage::UNIFORM
        | GlBufferUsage::COPY_SOURCE
        | GlBufferUsage::COPY_DESTINATION;
    assert_eq!(buffer(usage).validate(), Ok(()));
    assert_eq!(usage.conflicting_roles(), None);
}

#[test]
fn staging_roles_stay_legal_next_to_a_mapping() {
    // Upload staging and readback staging are the two patterns that need a
    // mapping beside a copy role.
    assert_eq!(
        buffer(GlBufferUsage::MAP_WRITE | GlBufferUsage::COPY_SOURCE).validate(),
        Ok(())
    );
    assert_eq!(
        buffer(GlBufferUsage::MAP_READ | GlBufferUsage::COPY_DESTINATION).validate(),
        Ok(())
    );
}

#[test]
fn mapping_roles_are_rejected_beside_each_other_and_beside_gpu_use() {
    let read_write = GlBufferUsage::MAP_READ | GlBufferUsage::MAP_WRITE;
    assert_eq!(
        buffer(read_write).validate(),
        Err(GlResourceValidationError::ConflictingBufferRoles {
            first: GlBufferRole::MapRead,
            second: GlBufferRole::MapWrite,
        })
    );
    for role in [
        GlBufferRole::Vertex,
        GlBufferRole::Index,
        GlBufferRole::Uniform,
        GlBufferRole::Storage,
        GlBufferRole::Indirect,
    ] {
        for mapping in [GlBufferRole::MapRead, GlBufferRole::MapWrite] {
            let usage = mapping.usage() | role.usage();
            assert_eq!(
                buffer(usage).validate(),
                Err(GlResourceValidationError::ConflictingBufferRoles {
                    first: mapping,
                    second: role,
                }),
                "{mapping:?} beside {role:?} must be rejected"
            );
        }
    }
}

#[test]
fn the_reported_conflict_pair_is_deterministic_and_named() {
    // `ALL` order decides which pair is reported when several conflicts
    // exist, so the same descriptor always produces the same error.
    let error = buffer(GlBufferUsage::VERTEX | GlBufferUsage::MAP_WRITE | GlBufferUsage::MAP_READ)
        .validate()
        .expect_err("three-way conflict is rejected");
    assert_eq!(
        error,
        GlResourceValidationError::ConflictingBufferRoles {
            first: GlBufferRole::MapRead,
            second: GlBufferRole::MapWrite,
        }
    );
    assert_eq!(
        error.message(),
        "buffer usage combines conflicting roles `map-read` and `map-write`"
    );
}

#[test]
fn every_role_reports_a_distinct_usage_bit() {
    let mut seen = GlBufferUsage::EMPTY;
    for role in GlBufferRole::ALL {
        assert!(
            !seen.contains(role.usage()),
            "{role:?} reuses another role's usage bit"
        );
        seen = seen | role.usage();
    }
}

#[test]
fn rejects_overflowing_buffer_ranges_before_use() {
    let d = GlBufferDesc {
        size: 8,
        usage: GlBufferUsage::COPY_SOURCE,
    };
    assert_eq!(
        GlBufferDesc {
            size: 0,
            usage: GlBufferUsage::EMPTY
        }
        .validate(),
        Err(GlResourceValidationError::ZeroBufferSize)
    );
    let id = BufferId::new(
        super::super::ContextStamp::new(
            super::super::DeviceIdentity::new(1).unwrap(),
            super::super::ContextEpoch::INITIAL,
        ),
        0,
        0,
    );
    assert_eq!(
        GlBufferRange {
            buffer: id,
            offset: u64::MAX,
            size: 1
        }
        .validate_for(d),
        Err(GlResourceValidationError::BufferRangeOutOfBounds)
    );
}
#[test]
fn renderbuffer_descriptors_reject_zero_extent_and_samples() {
    let base = GlRenderBufferDesc {
        format: GlFormat::Rgba8Unorm,
        width: 4,
        height: 4,
        samples: 4,
    };
    assert_eq!(base.validate(), Ok(()));
    assert_eq!(
        GlRenderBufferDesc { width: 0, ..base }.validate(),
        Err(GlResourceValidationError::ZeroRenderBufferExtent)
    );
    assert_eq!(
        GlRenderBufferDesc { samples: 0, ..base }.validate(),
        Err(GlResourceValidationError::ZeroSampleCount)
    );
}
#[test]
fn multisample_textures_cannot_have_mips() {
    let desc = GlTextureDesc {
        dimension: GlTextureDimension::D2,
        extent: GlExtent3d {
            width: 4,
            height: 4,
            depth_or_layers: 1,
        },
        mip_level_count: 2,
        sample_count: 4,
        format: GlFormat::Rgba8Unorm,
        usage: GlTextureUsage::SAMPLED,
    };
    assert_eq!(
        desc.validate(),
        Err(GlResourceValidationError::MultisampleMipmapped)
    );
}
#[test]
fn mip_count_and_multisample_dimension_are_bounded() {
    let mut desc = GlTextureDesc {
        dimension: GlTextureDimension::D2,
        extent: GlExtent3d {
            width: 4,
            height: 4,
            depth_or_layers: 1,
        },
        mip_level_count: 4,
        sample_count: 1,
        format: GlFormat::Rgba8Unorm,
        usage: GlTextureUsage::SAMPLED,
    };
    assert_eq!(
        desc.validate(),
        Err(GlResourceValidationError::MipLevelCountOutOfBounds)
    );
    desc.mip_level_count = 1;
    desc.dimension = GlTextureDimension::D3;
    desc.sample_count = 4;
    assert_eq!(
        desc.validate(),
        Err(GlResourceValidationError::MultisampleRequiresD2)
    );
    desc.dimension = GlTextureDimension::D2Array;
    desc.extent.depth_or_layers = 2;
    assert_eq!(desc.validate(), Ok(()));
}
