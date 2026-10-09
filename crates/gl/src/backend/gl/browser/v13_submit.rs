//! Browser-private immediate action vocabulary.
//!
//! WebGL2 work is resolved and executed on the context owner while an encoder
//! is open. These actions carry generation-safe native ids and copied CPU data;
//! they are consumed during the encoder call and are absent from completed work.

use crate::api::error::RhiResult;

use super::driver::BrowserDriverState;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct BrowserReadbackSinkId(pub(super) u64);

pub(super) enum BrowserV13CopyPacket {
    Buffer {
        source: crate::backend::gl::api::GlBufferRange,
        destination: crate::backend::gl::api::GlBufferRange,
    },
    Texture {
        source: crate::backend::gl::api::GlTextureRegion,
        destination: crate::backend::gl::api::GlTextureRegion,
    },
}

/// Borrowed direct-copy input consumed on the browser owner thread.
pub(super) enum BrowserCopyInput<'a> {
    Buffer(&'a crate::api::command::copy::BufferCopy),
    Texture(&'a crate::api::command::copy::TextureCopy),
}

pub(super) enum BrowserV13UploadPacket {
    Buffer {
        destination: crate::backend::gl::api::GlBufferRange,
        bytes: std::sync::Arc<[u8]>,
    },
    Texture {
        destination: crate::backend::gl::api::GlTextureRegion,
        layout: crate::backend::gl::api::GlPixelLayout,
        bytes: std::sync::Arc<[u8]>,
    },
}

pub(super) enum BrowserV13ReadbackPacket {
    Buffer {
        source: crate::backend::gl::api::GlBufferRange,
        sink: BrowserReadbackSinkId,
    },
    Texture {
        source: crate::backend::gl::api::GlTextureRegion,
        layout: crate::backend::gl::api::GlPixelLayout,
        sink: BrowserReadbackSinkId,
    },
}

pub(super) enum BrowserV13QueryPacket {
    BeginOcclusion(crate::backend::gl::api::QueryId),
    EndOcclusion,
    BeginElapsed(crate::backend::gl::api::QueryId),
    EndElapsed,
    Timestamp(crate::backend::gl::api::QueryId),
}

pub(super) enum BrowserV13ActionPacket {
    Copy(BrowserV13CopyPacket),
    Upload(BrowserV13UploadPacket),
    Readback(BrowserV13ReadbackPacket),
    Query(BrowserV13QueryPacket),
}

pub(super) struct BrowserV13PacketAction(pub(super) BrowserV13ActionPacket);

impl BrowserImmediateAction for BrowserV13PacketAction {
    fn execute(self: Box<Self>, owner: &mut BrowserDriverState) -> RhiResult<()> {
        owner.execute_v13_packet(self.0)
    }
}

pub(super) trait BrowserImmediateAction {
    fn execute(self: Box<Self>, owner: &mut BrowserDriverState) -> RhiResult<()>;
}

/// Synchronous owner-table resolution used by typed encoder callbacks.
pub(super) trait BrowserObjectResolver {
    fn readback(
        &self,
        ticket: &crate::api::resource::transfer::ReadbackTicket,
    ) -> RhiResult<Box<dyn BrowserImmediateAction>>;
    fn copy(&self, copy: BrowserCopyInput<'_>) -> RhiResult<Box<dyn BrowserImmediateAction>>;
    fn upload(
        &self,
        upload: &crate::api::resource::transfer::UploadJob,
    ) -> RhiResult<Box<dyn BrowserImmediateAction>>;
    fn query_begin(
        &self,
        set: &crate::api::query::QuerySet,
        index: u32,
    ) -> RhiResult<Box<dyn BrowserImmediateAction>>;
    fn query_end(
        &self,
        set: &crate::api::query::QuerySet,
        index: u32,
    ) -> RhiResult<Box<dyn BrowserImmediateAction>>;
    fn timestamp_write(
        &self,
        set: &crate::api::query::QuerySet,
        index: u32,
    ) -> RhiResult<Box<dyn BrowserImmediateAction>>;
    fn raster_begin(
        &self,
        begin: &crate::api::command::record::RasterBegin,
    ) -> RhiResult<Box<dyn BrowserImmediateAction>>;
    fn raster_end(&self) -> RhiResult<Box<dyn BrowserImmediateAction>>;
    fn raster_draw(
        &self,
        draw: &crate::backend::gl::translate_raster::GlRasterDrawInput<'_>,
    ) -> RhiResult<Box<dyn BrowserImmediateAction>>;
}
