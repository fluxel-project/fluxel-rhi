use super::GlQueryResult;
#[test]
fn disjoint_is_not_a_zero_elapsed_time() {
    assert_ne!(GlQueryResult::Disjoint, GlQueryResult::Available(0));
}
