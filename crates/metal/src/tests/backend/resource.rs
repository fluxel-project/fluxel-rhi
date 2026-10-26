//! Unit tests for the corresponding implementation module.

use super::unsupported_buffer_usage;
use fluxel_rhi_core::api::BufferUsage;

#[test]
fn buffer_support_accepts_mapping_and_rejects_shader_tables() {
    assert!(unsupported_buffer_usage(BufferUsage(0)));
    assert!(!unsupported_buffer_usage(BufferUsage::MAP_READ));
    assert!(!unsupported_buffer_usage(BufferUsage::MAP_WRITE));
    assert!(!unsupported_buffer_usage(BufferUsage::ACCEL_BUILD_INPUT));
    assert!(!unsupported_buffer_usage(BufferUsage::ACCEL_SCRATCH));
    assert!(unsupported_buffer_usage(BufferUsage::SHADER_TABLE));
    assert!(!unsupported_buffer_usage(BufferUsage::COPY_SRC));
    assert!(!unsupported_buffer_usage(BufferUsage::VERTEX));
}
