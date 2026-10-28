use super::{
    GlDrawCommand, GlIndexedDraw, GlMultiDraw, GlMultiDrawValidationError, GlNonIndexedDraw,
};

fn non_indexed(first_vertex: u32, vertex_count: u32, instance_count: u32) -> GlDrawCommand {
    GlDrawCommand::NonIndexed(GlNonIndexedDraw {
        first_vertex,
        vertex_count,
        instance_count,
    })
}

fn indexed(first_index: u32, index_count: u32, instance_count: u32) -> GlDrawCommand {
    GlDrawCommand::Indexed(GlIndexedDraw {
        first_index,
        index_count,
        instance_count,
    })
}

#[test]
fn a_batch_with_no_draw_is_not_a_batch() {
    assert_eq!(
        GlMultiDraw::new(vec![]).err(),
        Some(GlMultiDrawValidationError::EmptyBatch)
    );
}

/// One command carries one shape of per-draw parameter slice, so a mixed
/// batch is refused instead of being silently split into two submissions.
#[test]
fn a_batch_mixing_shapes_is_rejected() {
    assert_eq!(
        GlMultiDraw::new(vec![non_indexed(0, 3, 1), indexed(0, 3, 1)]).err(),
        Some(GlMultiDrawValidationError::MixedDrawShapes)
    );
    assert_eq!(
        GlMultiDraw::new(vec![indexed(0, 3, 1), non_indexed(0, 3, 1)]).err(),
        Some(GlMultiDrawValidationError::MixedDrawShapes)
    );
}

/// A draw that reads nothing or runs nothing is refused for every draw of
/// the batch, not only the first one, because a later zero would otherwise
/// be found halfway through a decomposed submission.
#[test]
fn every_draw_of_the_batch_must_be_a_real_draw() {
    assert_eq!(
        GlMultiDraw::new(vec![non_indexed(0, 3, 1), non_indexed(3, 0, 1)]).err(),
        Some(GlMultiDrawValidationError::ZeroDrawCount)
    );
    assert_eq!(
        GlMultiDraw::new(vec![non_indexed(0, 3, 1), non_indexed(3, 3, 0)]).err(),
        Some(GlMultiDrawValidationError::ZeroInstanceCount)
    );
    assert_eq!(
        GlMultiDraw::new(vec![indexed(0, 6, 1), indexed(6, 0, 1)]).err(),
        Some(GlMultiDrawValidationError::ZeroDrawCount)
    );
    assert_eq!(
        GlMultiDraw::new(vec![indexed(0, 6, 1), indexed(6, 6, 0)]).err(),
        Some(GlMultiDrawValidationError::ZeroInstanceCount)
    );
}

#[test]
fn per_draw_values_must_survive_one_parameter_slot() {
    assert_eq!(
        GlMultiDraw::new(vec![non_indexed(u32::MAX, 3, 1)]).err(),
        Some(GlMultiDrawValidationError::ValueExceedsCommandRange)
    );
    assert_eq!(
        GlMultiDraw::new(vec![non_indexed(0, u32::MAX, 1)]).err(),
        Some(GlMultiDrawValidationError::ValueExceedsCommandRange)
    );
    assert_eq!(
        GlMultiDraw::new(vec![indexed(0, 6, u32::MAX)]).err(),
        Some(GlMultiDrawValidationError::ValueExceedsCommandRange)
    );
    // The exact boundary of the signed slot is still one draw.
    assert!(GlMultiDraw::new(vec![non_indexed(i32::MAX as u32, i32::MAX as u32, 1)]).is_ok());
}

/// `first_index` is not a per-draw parameter: it is only ever turned into a
/// byte offset against the bound index format, so its range belongs to the
/// provider that knows the format.
#[test]
fn an_index_offset_beyond_the_parameter_slot_stays_the_providers_rule() {
    assert!(GlMultiDraw::new(vec![indexed(u32::MAX, 1, 1)]).is_ok());
}

#[test]
fn a_batch_keeps_its_draws_in_submission_order() {
    let batch = GlMultiDraw::new(vec![indexed(6, 3, 1), indexed(0, 3, 1), indexed(9, 3, 1)])
        .expect("a homogeneous batch");
    assert_eq!(batch.draws().len(), 3);
    assert_eq!(batch.draws()[0], indexed(6, 3, 1));
    assert_eq!(batch.draws()[1], indexed(0, 3, 1));
    assert_eq!(batch.draws()[2], indexed(9, 3, 1));
}

/// The plain combined command carries no instance count, so it is chosen
/// only when every draw already runs exactly one instance.
#[test]
fn instance_counts_are_needed_exactly_when_a_draw_runs_several_instances() {
    let single = |draws| GlMultiDraw::new(draws).expect("a valid batch");
    assert!(!single(vec![non_indexed(0, 3, 1), non_indexed(3, 3, 1)]).needs_instance_counts());
    assert!(single(vec![non_indexed(0, 3, 1), non_indexed(3, 3, 2)]).needs_instance_counts());
    assert!(!single(vec![indexed(0, 6, 1)]).needs_instance_counts());
    assert!(single(vec![indexed(0, 6, 1), indexed(6, 6, 4)]).needs_instance_counts());
}
