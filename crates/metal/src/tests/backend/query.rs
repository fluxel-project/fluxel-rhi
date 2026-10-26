//! Unit tests for the corresponding implementation module.

use super::*;
#[test]
fn slot_state_follows_submission_order_and_lane() {
    let set = MetalQuerySet(Arc::new(QueryState {
        statistics_pass: None,
        submitted: Mutex::new(Submitted {
            slots: vec![None; 2],
            generations: vec![None; 2],
        }),
    }));
    let reset = QueryOp::Reset(set.clone(), 0..2);
    let write = QueryOp::Write(set.clone(), 0);
    let resolve = QueryOp::Resolve(set.clone(), 0..1);
    assert!(matches!(
        validate_ops([(0, &resolve)].into_iter()),
        Err(Error::InvalidState)
    ));
    assert!(matches!(
        validate_ops([(0, &write)].into_iter()),
        Err(Error::InvalidState)
    ));
    validate_ops([(0, &reset), (0, &write), (0, &resolve)].into_iter()).unwrap();
    let first = commit_ops(&[reset.clone(), write.clone()], 0);
    assert!(validate_ops([(0, &resolve)].into_iter()).is_ok());
    assert!(validate_ops([(1, &resolve)].into_iter()).is_ok());
    invalidate_ops(&first, 0);
    assert!(matches!(
        validate_ops([(0, &resolve)].into_iter()),
        Err(Error::InvalidState)
    ));
    let older = commit_ops(&[reset.clone(), write.clone()], 0);
    let newer = commit_ops(&[reset, write], 0);
    invalidate_ops(&older, 0);
    assert!(validate_ops([(0, &resolve)].into_iter()).is_ok());
    invalidate_ops(&newer, 0);
    assert!(matches!(
        validate_ops([(0, &resolve)].into_iter()),
        Err(Error::InvalidState)
    ));
}
