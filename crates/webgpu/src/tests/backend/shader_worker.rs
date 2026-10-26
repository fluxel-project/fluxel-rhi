use super::translate;

#[test]
fn malformed_spirv_is_a_structured_error() {
    assert!(translate(&[0, 1, 2, 3]).is_err());
}
