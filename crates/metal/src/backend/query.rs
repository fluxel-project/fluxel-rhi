//! Explicitly unsupported query operations for the current Metal profile.
//!
//! `QueryBackend` is implemented so callers can ask for capabilities and use
//! their documented visual fallback. No query object or result is fabricated.

use std::ops::Range;

use fluxel_rhi_core::api::{self, QueryBackend, QueryDesc, QueryKind};

use super::{Metal, MetalBuffer, MetalCommandEncoder, MetalDevice};

impl QueryBackend for Metal {
    fn query_supported(_: &MetalDevice, _: QueryKind) -> bool {
        false
    }

    fn create_query_set(_: &MetalDevice, _: &QueryDesc) -> api::Result<()> {
        Err(api::Error::Unsupported("Metal query sets"))
    }

    fn write_timestamp(_: &mut MetalCommandEncoder, _: &(), _: u32) -> api::Result<()> {
        Err(api::Error::Unsupported("Metal timestamp queries"))
    }

    fn begin_occlusion(_: &mut MetalCommandEncoder, _: &(), _: u32) -> api::Result<()> {
        Err(api::Error::Unsupported("Metal occlusion queries"))
    }

    fn reset_queries(_: &mut MetalCommandEncoder, _: &(), _: Range<u32>) -> api::Result<()> {
        Err(api::Error::Unsupported("Metal query reset"))
    }

    fn end_occlusion(_: &mut MetalCommandEncoder) -> api::Result<()> {
        Err(api::Error::Unsupported("Metal occlusion queries"))
    }

    fn begin_statistics(_: &mut MetalCommandEncoder, _: &(), _: u32) -> api::Result<()> {
        Err(api::Error::Unsupported("Metal pipeline statistics"))
    }

    fn end_statistics(_: &mut MetalCommandEncoder) -> api::Result<()> {
        Err(api::Error::Unsupported("Metal pipeline statistics"))
    }

    fn resolve_queries(
        _: &mut MetalCommandEncoder,
        _: &(),
        _: Range<u32>,
        _: &MetalBuffer,
        _: u64,
    ) -> api::Result<()> {
        Err(api::Error::Unsupported("Metal query resolve"))
    }
}
