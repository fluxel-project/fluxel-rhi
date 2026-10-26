//! Unit tests for the corresponding implementation module.

use super::*;

#[test]
fn resolve_layouts_require_conservative_scope() {
    assert!(resolve_scope(api::ImageLayout::ResolveSrc));
    assert!(resolve_scope(api::ImageLayout::ResolveDst));
    assert!(!resolve_scope(api::ImageLayout::ColorAttachment));
    assert!(!resolve_scope(api::ImageLayout::CopySrc));
}
