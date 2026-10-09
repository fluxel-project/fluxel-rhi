//! Finished native command work and typed descriptors used while an encoder is
//! open.  It deliberately contains no portable command stream.

use crate::api::binding::{BindGroup, BindGroupIndex};
use crate::api::command::IndexFormat;
use crate::api::command::ResourceUse;
use crate::api::command::attachment::{
    ColorAttachment, DepthAttachmentMode, DepthStencilAttachment, StencilAttachmentMode,
};
use crate::api::command::backend::CommandBufferBackend;
use crate::api::command::geometry::{LoadOp, StoreOp};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::pipeline::RenderTargetSignature;
use crate::api::query::QuerySet;
use crate::api::resource::buffer::BufferBinding;
use crate::api::resource::transfer::ReadbackTicket;
use crate::api::submission::LaneWorkDomains;

/// One validated write into a pipeline interface's immediate-data address
/// space. It is forwarded to the native encoder and not retained after finish.
#[derive(Clone)]
pub struct ImmediateWrite {
    pub offset: u32,
    pub bytes: Vec<u8>,
    pub visibility: crate::api::shader::ShaderStages,
}

/// The validated data needed to begin a native raster pass.
#[derive(Clone)]
pub struct RasterBegin {
    pub label: Label,
    pub colors: Vec<(u32, ColorAttachment)>,
    pub depth_stencil: Option<DepthStencilAttachment>,
    pub occlusion_query_set: Option<QuerySet>,
}

/// The validated data needed to begin a native compute pass.
pub struct ComputeBegin {
    pub label: Label,
}

/// Binding state retained only while a scope is open, so a later draw can
/// report its actual resource uses before issuing its native call.
#[derive(Clone)]
pub struct BoundGroup {
    pub index: BindGroupIndex,
    pub group: BindGroup,
    pub dynamic_offsets: Vec<u32>,
}

/// Index binding state retained while a raster scope is open.
#[derive(Clone)]
pub struct BoundIndexBuffer {
    pub binding: BufferBinding,
    pub format: IndexFormat,
}

/// Draw-only work encoded into a backend secondary command buffer.
pub struct SecondaryRasterWork {
    device: DeviceIdentity,
    signature: RenderTargetSignature,
    uses: Vec<ResourceUse>,
    native: Box<dyn CommandBufferBackend>,
}

impl SecondaryRasterWork {
    pub(crate) fn from_native(
        work: RecordedWork,
        begin: RasterBegin,
        uses: Vec<ResourceUse>,
        draw_count: usize,
    ) -> crate::api::error::RhiResult<Self> {
        validate_secondary_begin(&begin)?;
        if draw_count == 0 {
            return Err(crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "secondary raster work needs at least one draw",
            ));
        }
        Ok(Self {
            device: work.device,
            signature: signature_from_begin(&begin),
            uses,
            native: work.native,
        })
    }

    pub(crate) fn device_identity(&self) -> DeviceIdentity {
        self.device
    }
    pub fn signature(&self) -> &RenderTargetSignature {
        &self.signature
    }
    pub fn native(&self) -> &dyn CommandBufferBackend {
        self.native.as_ref()
    }
    pub fn uses(&self) -> &[ResourceUse] {
        &self.uses
    }
}

fn validate_secondary_begin(begin: &RasterBegin) -> crate::api::error::RhiResult<()> {
    for (location, color) in &begin.colors {
        if !matches!(color.load, LoadOp::Load)
            || color.store != StoreOp::Store
            || color.resolve.is_some()
        {
            return Err(crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                format!("secondary raster attachment {location} must inherit with Load, Store, and no resolve"),
            ).at("CommandRecorder::finish_secondary_raster"));
        }
    }
    if let Some(depth_stencil) = &begin.depth_stencil {
        let depth_inherits = matches!(
            depth_stencil.depth,
            None | Some(DepthAttachmentMode::ReadOnly)
                | Some(DepthAttachmentMode::ReadWrite {
                    load: LoadOp::Load,
                    store: StoreOp::Store
                })
        );
        let stencil_inherits = matches!(
            depth_stencil.stencil,
            None | Some(StencilAttachmentMode::ReadOnly)
                | Some(StencilAttachmentMode::ReadWrite {
                    load: LoadOp::Load,
                    store: StoreOp::Store
                })
        );
        if !depth_inherits || !stencil_inherits {
            return Err(crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "secondary raster depth/stencil attachments must inherit with Load and Store",
            )
            .at("CommandRecorder::finish_secondary_raster"));
        }
    }
    Ok(())
}

fn signature_from_begin(begin: &RasterBegin) -> RenderTargetSignature {
    let mut color_formats = Vec::new();
    for (location, color) in &begin.colors {
        color_formats.resize(*location as usize + 1, None);
        color_formats[*location as usize] = Some(color.view.format());
    }
    let sample_count = begin
        .colors
        .first()
        .map(|(_, color)| color.view.sample_count())
        .or_else(|| begin.depth_stencil.as_ref().map(|d| d.view.sample_count()))
        .unwrap_or(1);
    RenderTargetSignature {
        color_formats,
        depth_stencil_format: begin.depth_stencil.as_ref().map(|d| d.view.format()),
        sample_count,
    }
    .canonicalized()
}

/// Completed work: one native command buffer plus portable submission facts.
pub struct RecordedWork {
    id: ObjectId,
    device: DeviceIdentity,
    domains: LaneWorkDomains,
    uses: Vec<ResourceUse>,
    native: Box<dyn CommandBufferBackend>,
    readbacks: Vec<ReadbackTicket>,
}

impl RecordedWork {
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        domains: LaneWorkDomains,
        uses: Vec<ResourceUse>,
        native: Box<dyn CommandBufferBackend>,
        readbacks: Vec<ReadbackTicket>,
    ) -> Self {
        Self {
            id,
            device,
            domains,
            uses,
            native,
            readbacks,
        }
    }

    pub fn native(&self) -> &dyn CommandBufferBackend {
        self.native.as_ref()
    }
    pub fn readbacks(&self) -> &[ReadbackTicket] {
        &self.readbacks
    }
    pub fn id(&self) -> ObjectId {
        self.id
    }
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }
    pub fn work_domains(&self) -> LaneWorkDomains {
        self.domains
    }
    pub fn resource_uses(&self) -> &[ResourceUse] {
        &self.uses
    }
}

impl core::fmt::Debug for RecordedWork {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("RecordedWork")
            .field("id", &self.id)
            .field("device", &self.device)
            .field("work_domains", &self.domains)
            .field("resource_uses", &self.uses.len())
            .finish_non_exhaustive()
    }
}
