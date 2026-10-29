//! Raster index-strip restart compatibility validation.

use super::*;
#[test]
fn strip_restart_requires_matching_index_width() {
    assert!(index_format_matches(
        Some(api::IndexFormat::Uint16),
        Some(api::IndexFormat::Uint16)
    ));
    assert!(!index_format_matches(
        Some(api::IndexFormat::Uint16),
        Some(api::IndexFormat::Uint32)
    ));
    assert!(!index_format_matches(Some(api::IndexFormat::Uint32), None));
    assert!(index_format_matches(None, Some(api::IndexFormat::Uint32)));
}
