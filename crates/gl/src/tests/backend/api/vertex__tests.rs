use super::*;
#[test]
fn attribute_is_hashable_structural_state() {
    let a = GlVertexAttribute {
        location: 2,
        buffer_slot: 1,
        format: GlVertexFormat::Float32x3,
        offset: 12,
    };
    assert_eq!(a, a);
}
#[test]
fn float64_attribute_extent_uses_its_native_byte_width() {
    let layout = GlVertexLayout {
        buffers: vec![GlVertexBufferLayout {
            slot: 0,
            stride: 16,
            step_mode: GlVertexStepMode::Vertex,
        }],
        attributes: vec![GlVertexAttribute {
            location: 0,
            buffer_slot: 0,
            format: GlVertexFormat::Float64x2,
            offset: 0,
        }],
    };
    assert_eq!(layout.validate(), Ok(()));
}
#[test]
fn zero_stride_uses_fixed_address_and_checks_attribute_extent() {
    let stamp = super::super::ContextStamp::new(
        super::super::DeviceIdentity::new(1).unwrap(),
        super::super::ContextEpoch::INITIAL,
    );
    let buffer = BufferId::new(stamp, 0, 0);
    let layout = GlVertexLayout {
        buffers: vec![GlVertexBufferLayout {
            slot: 0,
            stride: 0,
            step_mode: GlVertexStepMode::Vertex,
        }],
        attributes: vec![GlVertexAttribute {
            location: 0,
            buffer_slot: 0,
            format: GlVertexFormat::Float32,
            offset: 4,
        }],
    };
    let binding = GlVertexBufferBinding {
        slot: 0,
        buffer,
        offset: 4,
    };
    let metadata = |byte_length| GlVertexBufferMetadata {
        buffer,
        byte_length,
        usage: GlBufferUsage::VERTEX,
    };
    assert_eq!(
        layout.validate_bindings(&[binding], None, &[metadata(12)], stamp),
        Ok(())
    );
    assert_eq!(
        layout.validate_bindings(&[binding], None, &[metadata(11)], stamp),
        Err(GlVertexValidationError::BindingOffsetExceedsBuffer)
    );
}
#[test]
fn rejects_duplicate_buffer_slots() {
    let layout = GlVertexLayout {
        buffers: vec![
            GlVertexBufferLayout {
                slot: 0,
                stride: 12,
                step_mode: GlVertexStepMode::Vertex,
            },
            GlVertexBufferLayout {
                slot: 0,
                stride: 12,
                step_mode: GlVertexStepMode::Vertex,
            },
        ],
        attributes: vec![],
    };
    assert_eq!(
        layout.validate(),
        Err(GlVertexValidationError::DuplicateBufferSlot)
    );
}
#[test]
fn missing_binding_metadata_is_distinct_from_an_exceeded_offset() {
    let stamp = super::super::ContextStamp::new(
        super::super::DeviceIdentity::new(1).unwrap(),
        super::super::ContextEpoch::INITIAL,
    );
    let buffer = BufferId::new(stamp, 0, 0);
    let layout = GlVertexLayout {
        buffers: vec![GlVertexBufferLayout {
            slot: 0,
            stride: 12,
            step_mode: GlVertexStepMode::Vertex,
        }],
        attributes: vec![],
    };
    let binding = GlVertexBufferBinding {
        slot: 0,
        buffer,
        offset: 0,
    };
    assert_eq!(
        layout.validate_bindings(&[binding], None, &[], stamp),
        Err(GlVertexValidationError::BindingMetadataMissing)
    );
    let beyond = GlVertexBufferBinding {
        slot: 0,
        buffer,
        offset: 8,
    };
    assert_eq!(
        layout.validate_bindings(
            &[beyond],
            None,
            &[GlVertexBufferMetadata {
                buffer,
                byte_length: 4,
                usage: GlBufferUsage::VERTEX,
            }],
            stamp,
        ),
        Err(GlVertexValidationError::BindingOffsetExceedsBuffer)
    );
}

#[test]
fn allocation_lookup_validation_matches_slice_validation() {
    let stamp = super::super::ContextStamp::new(
        super::super::DeviceIdentity::new(1).unwrap(),
        super::super::ContextEpoch::INITIAL,
    );
    let vertex = BufferId::new(stamp, 0, 0);
    let index_buffer = BufferId::new(stamp, 1, 0);
    let layout = GlVertexLayout {
        buffers: vec![GlVertexBufferLayout {
            slot: 0,
            stride: 16,
            step_mode: GlVertexStepMode::Vertex,
        }],
        attributes: vec![GlVertexAttribute {
            location: 0,
            buffer_slot: 0,
            format: GlVertexFormat::Float32x4,
            offset: 0,
        }],
    };
    let binding = GlVertexBufferBinding {
        slot: 0,
        buffer: vertex,
        offset: 16,
    };
    let index = GlIndexBinding {
        buffer: index_buffer,
        format: GlIndexFormat::Uint16,
        offset: 2,
    };
    let facts = [
        GlVertexBufferMetadata {
            buffer: vertex,
            byte_length: 32,
            usage: GlBufferUsage::VERTEX,
        },
        GlVertexBufferMetadata {
            buffer: index_buffer,
            byte_length: 4,
            usage: GlBufferUsage::INDEX,
        },
    ];
    macro_rules! assert_equivalent {
        ($bindings:expr, $index:expr, $facts:expr) => {{
            let expected = layout.validate_bindings($bindings, $index, $facts, stamp);
            let actual = layout.validate_bound_bindings_with($bindings, $index, stamp, |buffer| {
                $facts.iter().find(|fact| fact.buffer == buffer).copied()
            });
            assert_eq!(actual, expected);
            expected
        }};
    }

    assert_eq!(layout.validate(), Ok(()));
    assert_eq!(assert_equivalent!(&[binding], Some(index), &facts), Ok(()));

    let stale = BufferId::new(stamp, 77, 1);
    assert_eq!(
        assert_equivalent!(
            &[GlVertexBufferBinding {
                buffer: stale,
                ..binding
            }],
            None,
            &facts
        ),
        Err(GlVertexValidationError::BindingMetadataMissing)
    );
    assert_eq!(
        assert_equivalent!(
            &[GlVertexBufferBinding {
                offset: 33,
                ..binding
            }],
            None,
            &facts
        ),
        Err(GlVertexValidationError::BindingOffsetExceedsBuffer)
    );

    let wrong_vertex_role = [GlVertexBufferMetadata {
        usage: GlBufferUsage::COPY_SOURCE,
        ..facts[0]
    }];
    assert_eq!(
        assert_equivalent!(&[binding], None, &wrong_vertex_role),
        Err(GlVertexValidationError::BindingBufferRoleMissing)
    );

    let other_stamp = super::super::ContextStamp::new(
        super::super::DeviceIdentity::new(2).unwrap(),
        super::super::ContextEpoch::INITIAL,
    );
    let foreign = GlVertexBufferBinding {
        buffer: BufferId::new(other_stamp, 0, 0),
        ..binding
    };
    assert_eq!(
        assert_equivalent!(&[foreign], None, &facts),
        Err(GlVertexValidationError::ForeignBuffer)
    );

    assert_eq!(
        assert_equivalent!(&[], None, &facts),
        Err(GlVertexValidationError::BindingSlotMissing)
    );
    assert_eq!(
        assert_equivalent!(&[binding, binding], None, &facts),
        Err(GlVertexValidationError::DuplicateBindingSlot)
    );

    let misaligned = GlIndexBinding { offset: 1, ..index };
    assert_eq!(
        assert_equivalent!(&[binding], Some(misaligned), &facts),
        Err(GlVertexValidationError::IndexOffsetMisaligned)
    );
    let out_of_range = GlIndexBinding { offset: 6, ..index };
    assert_eq!(
        assert_equivalent!(&[binding], Some(out_of_range), &facts),
        Err(GlVertexValidationError::IndexOffsetExceedsBuffer)
    );
    let wrong_index_role = [
        facts[0],
        GlVertexBufferMetadata {
            usage: GlBufferUsage::COPY_DESTINATION,
            ..facts[1]
        },
    ];
    assert_eq!(
        assert_equivalent!(&[binding], Some(index), &wrong_index_role),
        Err(GlVertexValidationError::IndexBufferRoleMissing)
    );
    assert_eq!(
        assert_equivalent!(&[binding], Some(index), &facts[..1]),
        Err(GlVertexValidationError::IndexMetadataMissing)
    );
}

#[test]
fn a_buffer_must_declare_the_role_it_is_bound_for() {
    let layout = GlVertexLayout {
        buffers: vec![super::super::GlVertexBufferLayout {
            slot: 0,
            stride: 4,
            step_mode: GlVertexStepMode::Vertex,
        }],
        attributes: vec![],
    };
    let stamp = super::super::ContextStamp::new(
        super::super::DeviceIdentity::new(1).unwrap(),
        super::super::ContextEpoch::INITIAL,
    );
    let buffer = BufferId::new(stamp, 0, 0);
    let binding = GlVertexBufferBinding {
        slot: 0,
        buffer,
        offset: 0,
    };
    // Created for the copy roles only: legal as storage, wrong as an
    // attribute source.
    let metadata = [GlVertexBufferMetadata {
        buffer,
        byte_length: 4,
        usage: GlBufferUsage::COPY_SOURCE,
    }];
    assert_eq!(
        layout.validate_bindings(&[binding], None, &metadata, stamp),
        Err(GlVertexValidationError::BindingBufferRoleMissing)
    );

    // The index buffer is bound through its own point and is checked
    // against its own role, including when it is not an attribute buffer.
    let index_buffer = BufferId::new(stamp, 1, 0);
    let index = super::super::GlIndexBinding {
        buffer: index_buffer,
        format: GlIndexFormat::Uint16,
        offset: 0,
    };
    let index_only = [
        GlVertexBufferMetadata {
            buffer,
            byte_length: 4,
            usage: GlBufferUsage::VERTEX,
        },
        GlVertexBufferMetadata {
            buffer: index_buffer,
            byte_length: 4,
            usage: GlBufferUsage::COPY_DESTINATION,
        },
    ];
    assert_eq!(
        layout.validate_bindings(&[binding], Some(index), &index_only, stamp),
        Err(GlVertexValidationError::IndexBufferRoleMissing)
    );

    let mut index_facts = index_only;
    index_facts[1].usage = GlBufferUsage::INDEX | GlBufferUsage::COPY_SOURCE;
    assert_eq!(
        layout.validate_bindings(&[binding], Some(index), &index_facts, stamp),
        Ok(())
    );

    // No facts at all for the index buffer is its own failure, not a
    // bounds failure.
    assert_eq!(
        layout.validate_bindings(&[binding], Some(index), &index_only[..1], stamp),
        Err(GlVertexValidationError::IndexMetadataMissing)
    );
}
