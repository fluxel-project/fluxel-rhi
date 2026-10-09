use std::any::Any;
use std::sync::{Arc, Mutex};
use std::task::Waker;

use crate::api::capability::CapabilityFacts;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::ObjectId;
use crate::api::platform::backend::DeviceBackend;
use crate::api::platform::provider::{AdapterInfo, BackendKind};
use crate::api::platform::{DeviceLossInfo, DeviceStatus};
use crate::api::presentation::PresentationTarget;
use crate::api::submission::{
    CompletionFailure, CompletionState, LaneWorkDomains, SubmissionCapabilities,
    SubmissionLaneClass, SubmissionLaneId, SubmissionLaneInfo,
};

/// An owner-context native object name.  Zero is GL's "no object" sentinel and
/// can never back a published RHI handle.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct GlObjectName(u32);

impl GlObjectName {
    pub(crate) fn new(value: u32, operation: &'static str) -> RhiResult<Self> {
        if value == 0 {
            return Err(RhiError::new(
                RhiErrorKind::BackendFailure,
                "GL driver returned object name zero for a successful creation",
            )
            .at(operation));
        }
        Ok(Self(value))
    }
    pub(crate) fn raw(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GlObjectKind {
    Buffer,
    Texture,
    /// A view may be a virtual GL view sharing its texture name. `destroy` must
    /// then release only driver-side view metadata, never delete the texture.
    TextureView,
    Sampler,
    QuerySet,
    Shader,
    /// GL has no descriptor-set object. A driver may use a unique packet id and
    /// release cached binding metadata here; it must not claim a no-op packet is
    /// executable unless its submit lowerer consumes that packet.
    BindGroup,
    ComputePipeline,
    RasterPipeline,
}

/// GL-private references handed to a native/browser lowerer. They carry only
/// validated GL object names, never portable handles or platform context state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlBufferRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlTextureRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlTextureViewRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlSamplerRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlShaderRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlBindGroupRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlComputePipelineRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlRasterPipelineRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlQuerySetRef {
    pub(crate) name: GlObjectName,
}

/// Pipeline creation packets pair the only native shader inputs with the
/// already-validated portable fixed-state description. Native/browser drivers
/// must consume the refs and never downcast `ShaderModule` themselves.
pub(crate) struct GlComputePipelinePacket<'a> {
    pub(crate) shader: GlShaderRef,
    pub(crate) descriptor: &'a crate::api::pipeline::ComputePipelineDescriptor,
}
pub(crate) struct GlRasterPipelinePacket<'a> {
    pub(crate) vertex: GlShaderRef,
    pub(crate) fragment: Option<GlShaderRef>,
    pub(crate) descriptor: &'a crate::api::pipeline::RasterPipelineDescriptor,
}

/// Canonical, object-name-only packet built before a GL bind-group creation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GlBindGroupPacket {
    pub(crate) entries: Vec<GlBindGroupEntry>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GlBindGroupEntry {
    pub(crate) slot: u32,
    pub(crate) resource: GlBindingResource,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum GlBindingResource {
    Buffer {
        buffer: GlBufferRef,
        offset: u64,
        size: u64,
    },
    Texture(GlTextureViewRef),
    Sampler(GlSamplerRef),
    BufferArray(Vec<(GlBufferRef, u64, u64)>),
    TextureArray(Vec<GlTextureViewRef>),
    SamplerArray(Vec<GlSamplerRef>),
}

/// Platform-side lowering view. The GL driver receives this rather than the
/// generic submission seam; command detail remains crate-private to this crate.
pub(crate) struct GlSubmissionPlan {
    pub(crate) batches: Vec<GlSubmissionBatch>,
}
pub(crate) struct GlSubmissionBatch {
    /// All GL commands have already reached the context owner while their
    /// encoder was open. Submit only associates this completed native work
    /// token with its portable plan point and establishes a completion fence.
    pub(crate) point: Option<crate::api::submission::PlanPoint>,
}

/// One validated command delivered to the context owner at recorder time.
///
/// This intentionally carries only the semantic operands. GL has no native
/// command list, so it applies every command while the caller still owns the
/// open encoder state.
pub(crate) enum GlTypedCommand<'a> {
    RasterBegin {
        begin: &'a crate::api::command::record::RasterBegin,
        uses: &'a [crate::api::command::ResourceUse],
    },
    RasterClear {
        clear: &'a crate::api::command::RasterAttachmentClear,
        uses: &'a [crate::api::command::ResourceUse],
    },
    RasterExecuteSecondary {
        work: crate::api::command::SecondaryRasterWork,
        uses: &'a [crate::api::command::ResourceUse],
    },
    RasterSetPipeline(&'a crate::api::pipeline::RasterPipeline),
    RasterSetBindGroup {
        index: crate::api::binding::BindGroupIndex,
        group: &'a crate::api::binding::BindGroup,
        dynamic_offsets: &'a [u32],
    },
    RasterSetVertexBuffer {
        slot: u32,
        binding: &'a crate::api::resource::buffer::BufferBinding,
    },
    RasterSetIndexBuffer {
        binding: &'a crate::api::resource::buffer::BufferBinding,
        format: crate::api::command::IndexFormat,
    },
    RasterSetViewport(crate::api::command::geometry::Viewport),
    RasterSetScissor(crate::api::command::geometry::Rect),
    RasterSetBlendConstant(crate::api::command::geometry::Color),
    RasterSetStencilReference(u32),
    RasterSetImmediates(&'a crate::api::command::record::ImmediateWrite),
    RasterDraw {
        vertices: core::ops::Range<u32>,
        instances: core::ops::Range<u32>,
        uses: &'a [crate::api::command::ResourceUse],
    },
    RasterDrawIndexed {
        indices: core::ops::Range<u32>,
        base_vertex: i32,
        instances: core::ops::Range<u32>,
        uses: &'a [crate::api::command::ResourceUse],
    },
    RasterDrawIndirect {
        arguments: &'a crate::api::resource::Buffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
        count: Option<(&'a crate::api::resource::Buffer, u64, u32)>,
        indexed: bool,
        uses: &'a [crate::api::command::ResourceUse],
    },
    RasterBeginQuery {
        set: &'a crate::api::query::QuerySet,
        index: u32,
    },
    RasterEndQuery {
        set: &'a crate::api::query::QuerySet,
        index: u32,
    },
    RasterWriteTimestamp {
        set: &'a crate::api::query::QuerySet,
        index: u32,
    },
    RasterPushDebugGroup(&'a str),
    RasterPopDebugGroup,
    RasterInsertDebugMarker(&'a str),
    RasterEnd,
    CopyExternalImageToTexture {
        copy: &'a crate::api::external::ExternalImageCopyDescriptor,
        uses: &'a [crate::api::command::ResourceUse],
    },
    ClearBuffer {
        buffer: &'a crate::api::resource::Buffer,
        range: crate::api::resource::BufferRange,
        uses: &'a [crate::api::command::ResourceUse],
    },
    ClearTexture {
        texture: &'a crate::api::resource::Texture,
        subresources: crate::api::resource::subresource::TextureSubresourceRange,
        uses: &'a [crate::api::command::ResourceUse],
    },
    CopyBuffer {
        copy: &'a crate::api::command::copy::BufferCopy,
        uses: &'a [crate::api::command::ResourceUse],
    },
    CopyBufferToTexture {
        copy: &'a crate::api::command::copy::BufferTextureCopy,
        uses: &'a [crate::api::command::ResourceUse],
    },
    CopyTextureToBuffer {
        copy: &'a crate::api::command::copy::BufferTextureCopy,
        uses: &'a [crate::api::command::ResourceUse],
    },
    CopyTexture {
        copy: &'a crate::api::command::copy::TextureCopy,
        uses: &'a [crate::api::command::ResourceUse],
    },
    ResolveTexture {
        resolve: &'a crate::api::command::copy::TextureResolve,
        uses: &'a [crate::api::command::ResourceUse],
    },
    BlitTexture {
        blit: &'a crate::api::command::copy::TextureBlit,
        uses: &'a [crate::api::command::ResourceUse],
    },
    Upload {
        upload: &'a crate::api::resource::transfer::UploadJob,
        uses: &'a [crate::api::command::ResourceUse],
    },
    Readback {
        ticket: &'a crate::api::resource::transfer::ReadbackTicket,
        uses: &'a [crate::api::command::ResourceUse],
    },
    EncoderWriteTimestamp {
        set: &'a crate::api::query::QuerySet,
        index: u32,
        uses: &'a [crate::api::command::ResourceUse],
    },
    ResolveQuerySet {
        set: &'a crate::api::query::QuerySet,
        first_query: u32,
        query_count: u32,
        destination: &'a crate::api::resource::Buffer,
        destination_offset: u64,
        uses: &'a [crate::api::command::ResourceUse],
    },
    EncoderPushDebugGroup(&'a str),
    EncoderPopDebugGroup,
    EncoderInsertDebugMarker(&'a str),
    ComputeBegin(&'a crate::api::command::record::ComputeBegin),
    ComputeSetPipeline(&'a crate::api::pipeline::ComputePipeline),
    ComputeSetBindGroup {
        index: crate::api::binding::BindGroupIndex,
        group: &'a crate::api::binding::BindGroup,
        dynamic_offsets: &'a [u32],
    },
    ComputeSetImmediates(&'a crate::api::command::record::ImmediateWrite),
    ComputeDispatch {
        x: u32,
        y: u32,
        z: u32,
        uses: &'a [crate::api::command::ResourceUse],
    },
    ComputeDispatchIndirect {
        arguments: &'a crate::api::resource::Buffer,
        offset: u64,
        uses: &'a [crate::api::command::ResourceUse],
    },
    ComputeBeginQuery {
        set: &'a crate::api::query::QuerySet,
        index: u32,
    },
    ComputeEndQuery {
        set: &'a crate::api::query::QuerySet,
        index: u32,
    },
    ComputeWriteTimestamp {
        set: &'a crate::api::query::QuerySet,
        index: u32,
    },
    ComputePushDebugGroup(&'a str),
    ComputePopDebugGroup,
    ComputeInsertDebugMarker(&'a str),
    ComputeEnd,
    RasterSetMeshPipeline(&'a crate::api::pipeline::MeshPipeline),
    RasterDispatchMesh {
        x: u32,
        y: u32,
        z: u32,
        uses: &'a [crate::api::command::ResourceUse],
    },
    RasterDispatchMeshIndirect {
        arguments: &'a crate::api::resource::Buffer,
        offset: u64,
        count: Option<(&'a crate::api::resource::Buffer, u64, u32)>,
        uses: &'a [crate::api::command::ResourceUse],
    },
    AccelerationBuild {
        destination: &'a crate::api::resource::AccelerationStructure,
        scratch: &'a crate::api::resource::Buffer,
        mode: crate::api::resource::AccelerationStructureBuildMode,
        uses: &'a [crate::api::command::ResourceUse],
    },
    AccelerationCopy {
        source: &'a crate::api::resource::AccelerationStructure,
        destination: &'a crate::api::resource::AccelerationStructure,
        mode: crate::api::resource::AccelerationStructureCopyMode,
        uses: &'a [crate::api::command::ResourceUse],
    },
    AccelerationWriteCompactedSize {
        source: &'a crate::api::resource::AccelerationStructure,
        destination: &'a crate::api::resource::Buffer,
        destination_offset: u64,
        uses: &'a [crate::api::command::ResourceUse],
    },
    RayBegin(&'a crate::api::command::RayTracingScopeDescriptor),
    RaySetPipeline(&'a crate::api::pipeline::RayTracingPipeline),
    RaySetBindGroup {
        index: crate::api::binding::BindGroupIndex,
        group: &'a crate::api::binding::BindGroup,
        dynamic_offsets: &'a [u32],
    },
    RaySetImmediates(&'a crate::api::command::record::ImmediateWrite),
    RayDispatch {
        table: &'a crate::api::command::RayTracingShaderTable,
        width: u32,
        height: u32,
        depth: u32,
        uses: &'a [crate::api::command::ResourceUse],
    },
    RayEnd,
}

/// Backend-private one-way notification from a context owner to the v13
/// device liveness authority.  This deliberately carries no context/session
/// object: a GL context loss invalidates one `DeviceIdentity`, full stop.
pub(crate) trait GlLossSink: Send + Sync + 'static {
    fn report_context_loss(&self, info: DeviceLossInfo);
}

/// Owner-context dispatch seam used by native GL and browser implementations.
///
/// A GL context is thread-affine while public RHI handles are `Send + Sync`.
/// Implementors marshal each call to the current WGL/EGL/GLX owner or browser
/// owner thread.  The seam carries no native context, browser session, or token
/// into the common API.  `dispatch` must not report success unless the operation
/// actually reached that owner context.
pub(crate) trait GlExecutionDriver: Send + Sync + 'static {
    /// Marshals `operation` to the context's owner. Implementations perform the
    /// real GL/browser work synchronously before returning.
    fn dispatch(&self, operation: &'static str) -> RhiResult<()>;

    /// Installs the private terminal-loss notification route for this adopted
    /// context. Drivers use it for asynchronous browser/worker observations
    /// which cannot be returned through a synchronous public Device verb.
    fn install_loss_sink(&self, _: Arc<dyn GlLossSink>) {}

    fn create_buffer(
        &self,
        _: &crate::api::resource::buffer::BufferDescriptor,
    ) -> RhiResult<GlObjectName> {
        unsupported("create_buffer")
    }
    fn create_texture(
        &self,
        _: &crate::api::resource::texture::TextureDescriptor,
    ) -> RhiResult<GlObjectName> {
        unsupported("create_texture")
    }
    fn create_texture_view(
        &self,
        _: GlTextureRef,
        _: &crate::api::resource::view::TextureViewDescriptor,
    ) -> RhiResult<GlObjectName> {
        unsupported("create_texture_view")
    }
    fn create_sampler(
        &self,
        _: &crate::api::resource::sampler::SamplerDescriptor,
    ) -> RhiResult<GlObjectName> {
        unsupported("create_sampler")
    }
    fn create_query_set(
        &self,
        _: &crate::api::query::QuerySetDescriptor,
    ) -> RhiResult<GlObjectName> {
        unsupported("create_query_set")
    }
    fn create_shader(&self, _: &crate::api::shader::ShaderArtifact) -> RhiResult<GlObjectName> {
        unsupported("create_shader")
    }
    fn create_bind_group(&self, _: &GlBindGroupPacket) -> RhiResult<GlObjectName> {
        unsupported("create_bind_group")
    }
    fn create_compute_pipeline(&self, _: GlComputePipelinePacket<'_>) -> RhiResult<GlObjectName> {
        unsupported("create_compute_pipeline")
    }
    fn create_raster_pipeline(&self, _: GlRasterPipelinePacket<'_>) -> RhiResult<GlObjectName> {
        unsupported("create_raster_pipeline")
    }
    fn map_buffer(
        &self,
        _: GlBufferRef,
        _: crate::api::resource::MapMode,
        _: crate::api::resource::BufferRange,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::MappingRequestBackend>> {
        unsupported("map_buffer")
    }
    fn submit(
        &self,
        _: GlSubmissionPlan,
    ) -> RhiResult<crate::api::submission::backend::SubmissionOutcome> {
        unsupported("submit")
    }

    fn encode_typed(&self, _: GlTypedCommand<'_>) -> RhiResult<()> {
        unsupported("typed command encoding")
    }
    fn completion(&self, serial: u64) -> CompletionState {
        CompletionState::Failed(CompletionFailure::new(format!(
            "GL completion serial {serial} was never accepted"
        )))
    }
    fn completion_or_register_waker(&self, serial: u64, _: &Waker) -> CompletionState {
        self.completion(serial)
    }
    /// Destruction is called by wrapper `Drop`; implementations must schedule it
    /// onto the owner context and tolerate loss, where native deletion is moot.
    fn destroy(&self, _: GlObjectKind, _: GlObjectName) {}

    fn poll(&self) -> RhiResult<()> {
        self.dispatch("GlExecutionDriver::poll")
    }
    fn wait_idle(&self) -> RhiResult<()> {
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "the adopted GL driver has no verified idle wait route",
        ))
    }

    fn supports_presentation(&self, _target: &PresentationTarget) -> RhiResult<bool> {
        Ok(false)
    }
    fn presentation_capabilities(
        &self,
        _: crate::api::identity::ObjectId,
    ) -> RhiResult<crate::api::presentation::PresentationTargetCapabilities> {
        unsupported("presentation_capabilities")
    }
    fn configure_presentation(
        &self,
        _: crate::api::identity::ObjectId,
        _: &crate::api::presentation::PresentationConfiguration,
    ) -> RhiResult<super::presentation::GlPresentationLease> {
        unsupported("configure_presentation")
    }
    fn lease_capabilities(
        &self,
        _: super::presentation::GlPresentationLease,
    ) -> RhiResult<crate::api::presentation::PresentationTargetCapabilities> {
        unsupported("lease_capabilities")
    }
    fn reconfigure_or_register_waker(
        &self,
        _: super::presentation::GlPresentationLease,
        _: &crate::api::presentation::PresentationConfiguration,
        _: &Waker,
    ) -> std::task::Poll<RhiResult<()>> {
        std::task::Poll::Ready(unsupported("reconfigure_presentation"))
    }
    fn try_acquire(
        &self,
        _: super::presentation::GlPresentationLease,
    ) -> Result<
        Option<super::presentation::GlAcquiredFramebuffer>,
        crate::api::presentation::AcquireError,
    > {
        Err(crate::api::presentation::AcquireError::new(
            crate::api::presentation::AcquireErrorKind::NotReady,
            "the adopted GL driver has no acquired framebuffer",
        ))
    }
    fn acquire_or_register_waker(
        &self,
        _: super::presentation::GlPresentationLease,
        _: &Waker,
    ) -> std::task::Poll<
        Result<super::presentation::GlAcquiredFramebuffer, crate::api::presentation::AcquireError>,
    > {
        std::task::Poll::Ready(Err(crate::api::presentation::AcquireError::new(
            crate::api::presentation::AcquireErrorKind::NotReady,
            "the adopted GL driver has no acquired framebuffer",
        )))
    }
    fn abandon(
        &self,
        _: super::presentation::GlPresentationLease,
        _: crate::api::presentation::AcquiredFrameId,
    ) -> RhiResult<()> {
        unsupported("abandon_frame")
    }
    fn abandon_no_throw(
        &self,
        _: super::presentation::GlPresentationLease,
        _: crate::api::presentation::AcquiredFrameId,
    ) {
    }
    fn release_presentation(&self, _: super::presentation::GlPresentationLease) {}
    fn present(
        &self,
        _: super::presentation::GlAcquiredFramebuffer,
        _: crate::api::presentation::PresentReceiptId,
    ) {
    }
    fn terminate_present(
        &self,
        _: super::presentation::GlAcquiredFramebuffer,
        _: crate::api::presentation::PresentReceiptId,
        _: crate::api::presentation::PresentState,
    ) {
    }
    fn present_state(
        &self,
        _: crate::api::presentation::PresentReceiptId,
    ) -> RhiResult<crate::api::presentation::PresentState> {
        unsupported("present_state")
    }
    fn present_state_or_register_waker(
        &self,
        _: crate::api::presentation::PresentReceiptId,
        _: &Waker,
    ) -> RhiResult<crate::api::presentation::PresentState> {
        unsupported("present_state_or_register_waker")
    }
    /// Called once by the GL device terminal-loss authority. Implementations wake
    /// all pending acquire and present futures before returning.
    fn device_lost(&self, _: &DeviceLossInfo) {}
}

impl crate::api::presentation::backend::PresentationBackend for GlDevice {
    fn capabilities(
        &self,
        target: ObjectId,
    ) -> RhiResult<crate::api::presentation::PresentationTargetCapabilities> {
        self.active("GlDevice::presentation_capabilities")?;
        self.observe(
            self.driver.presentation_capabilities(target),
            "GlDevice::presentation_capabilities",
        )
    }
    fn configure(
        &self,
        _: crate::api::identity::DeviceIdentity,
        target: ObjectId,
        config: &crate::api::presentation::PresentationConfiguration,
    ) -> RhiResult<Box<dyn crate::api::presentation::backend::ConfiguredPresentationBackend>> {
        self.active("GlDevice::configure_presentation")?;
        let lease = self.observe(
            self.driver.configure_presentation(target, config),
            "GlDevice::configure_presentation",
        )?;
        Ok(super::presentation::configured(
            Arc::clone(&self.driver),
            Arc::clone(&self.loss_sink),
            lease,
        ))
    }
    fn present_state(
        &self,
        receipt: crate::api::presentation::PresentReceiptId,
    ) -> RhiResult<crate::api::presentation::PresentState> {
        self.active("GlDevice::present_state")?;
        self.observe(
            self.driver.present_state(receipt),
            "GlDevice::present_state",
        )
    }
    fn present_state_or_register_waker(
        &self,
        receipt: crate::api::presentation::PresentReceiptId,
        waker: &Waker,
    ) -> RhiResult<crate::api::presentation::PresentState> {
        self.active("GlDevice::present_state_or_register_waker")?;
        self.observe(
            self.driver.present_state_or_register_waker(receipt, waker),
            "GlDevice::present_state_or_register_waker",
        )
    }
}

fn unsupported<T>(operation: &'static str) -> RhiResult<T> {
    Err(RhiError::new(
        RhiErrorKind::Unsupported,
        "the adopted GL driver did not install this v13 lowering",
    )
    .at(operation))
}

macro_rules! native_wrapper {
    ($name:ident, $kind:ident, $trait:path) => {
        struct $name {
            driver: Arc<dyn GlExecutionDriver>,
            name: GlObjectName,
        }
        impl Drop for $name {
            fn drop(&mut self) {
                self.driver.destroy(GlObjectKind::$kind, self.name);
            }
        }
        impl $trait for $name {
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
    };
}

native_wrapper!(
    GlBuffer,
    Buffer,
    crate::api::resource::backend::BufferBackend
);
native_wrapper!(
    GlTexture,
    Texture,
    crate::api::resource::backend::TextureBackend
);
native_wrapper!(
    GlTextureView,
    TextureView,
    crate::api::resource::backend::TextureViewBackend
);
native_wrapper!(
    GlSampler,
    Sampler,
    crate::api::resource::backend::SamplerBackend
);
native_wrapper!(
    GlQuerySet,
    QuerySet,
    crate::api::resource::backend::QuerySetBackend
);
native_wrapper!(
    GlShader,
    Shader,
    crate::api::shader::backend::ShaderModuleBackend
);
native_wrapper!(
    GlBindGroup,
    BindGroup,
    crate::api::binding::backend::BindGroupBackend
);
native_wrapper!(
    GlComputePipeline,
    ComputePipeline,
    crate::api::pipeline::backend::ComputePipelineBackend
);
native_wrapper!(
    GlRasterPipeline,
    RasterPipeline,
    crate::api::pipeline::backend::RasterPipelineBackend
);

struct Liveness {
    status: DeviceStatus,
    info: Option<DeviceLossInfo>,
    waiters: Vec<Waker>,
}

impl Liveness {
    fn report(&mut self, info: DeviceLossInfo) -> Option<Vec<Waker>> {
        if matches!(self.status, DeviceStatus::Lost) {
            return None;
        }
        self.status = DeviceStatus::Lost;
        self.info = Some(info);
        Some(core::mem::take(&mut self.waiters))
    }
}

struct GlLossAuthority {
    liveness: Arc<Mutex<Liveness>>,
}

impl GlLossAuthority {
    fn report(&self, info: DeviceLossInfo) -> bool {
        let waiters = self
            .liveness
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .report(info);
        let Some(waiters) = waiters else {
            return false;
        };
        for waker in waiters {
            waker.wake();
        }
        true
    }
}

impl GlLossSink for GlLossAuthority {
    fn report_context_loss(&self, info: DeviceLossInfo) {
        self.report(info);
    }
}

/// Context-owned GL device adapter.  Lowering is intentionally fail-closed
/// until the state-machine command executor is wired through `GlExecutionDriver`.
pub(crate) struct GlDevice {
    backend: BackendKind,
    adapter: AdapterInfo,
    facts: CapabilityFacts,
    object: ObjectId,
    driver: Arc<dyn GlExecutionDriver>,
    liveness: Arc<Mutex<Liveness>>,
    loss_sink: Arc<dyn GlLossSink>,
    immediate_order: Arc<Mutex<GlImmediateOrder>>,
}

/// GL calls are committed while recording.  One open encoder at a time keeps
/// that physical order representable by the later portable submission plan.
struct GlImmediateOrder {
    open: bool,
    next_encode: u64,
    next_submit: u64,
}

/// GL's finished command-buffer token.
///
/// OpenGL executes against the current context while a recorder is being
/// encoded. Submit therefore only advances the completion token;
/// this token only proves that those owner-thread calls completed and makes the
/// resulting work single-submit like every other backend command buffer.
struct GlImmediateCommandBuffer {
    available: Mutex<bool>,
    order: u64,
}

impl GlImmediateCommandBuffer {
    fn new(order: u64) -> Self {
        Self {
            available: Mutex::new(true),
            order,
        }
    }

    fn take(&self) -> RhiResult<()> {
        let mut available = self
            .available
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if !*available {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "an immediate GL command buffer may be submitted only once",
            )
            .at("GlDevice::submit"));
        }
        *available = false;
        Ok(())
    }
}

impl crate::api::command::backend::CommandBufferBackend for GlImmediateCommandBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Bridges the portable recorder to GL's context-affine immediate stream.
/// The driver synchronously marshals every `encode` call to the context owner;
/// no command payload is retained by this encoder or by the finished token.
struct GlImmediateEncoder {
    driver: Arc<dyn GlExecutionDriver>,
    order: u64,
    state: Arc<Mutex<GlImmediateOrder>>,
}

impl Drop for GlImmediateEncoder {
    fn drop(&mut self) {
        // `finish` normally clears this lease.  A poisoned or abandoned
        // recorder must not permanently prevent a later encoder from using
        // the immediate context.
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.open = false;
    }
}

impl crate::api::command::backend::CommandEncoderBackend for GlImmediateEncoder {
    fn raster_begin(
        &mut self,
        begin: &crate::api::command::record::RasterBegin,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterBegin { begin, uses })
    }
    fn raster_clear(
        &mut self,
        clear: &crate::api::command::RasterAttachmentClear,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterClear { clear, uses })
    }
    fn raster_execute_secondary(
        &mut self,
        work: crate::api::command::SecondaryRasterWork,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterExecuteSecondary { work, uses })
    }
    fn raster_set_pipeline(
        &mut self,
        pipeline: &crate::api::pipeline::RasterPipeline,
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterSetPipeline(pipeline))
    }
    fn raster_set_bind_group(
        &mut self,
        index: crate::api::binding::BindGroupIndex,
        group: &crate::api::binding::BindGroup,
        dynamic_offsets: &[u32],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterSetBindGroup {
                index,
                group,
                dynamic_offsets,
            })
    }
    fn raster_set_vertex_buffer(
        &mut self,
        slot: u32,
        binding: &crate::api::resource::buffer::BufferBinding,
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterSetVertexBuffer { slot, binding })
    }
    fn raster_set_index_buffer(
        &mut self,
        binding: &crate::api::resource::buffer::BufferBinding,
        format: crate::api::command::IndexFormat,
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterSetIndexBuffer { binding, format })
    }
    fn raster_set_viewport(
        &mut self,
        value: crate::api::command::geometry::Viewport,
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterSetViewport(value))
    }
    fn raster_set_scissor(&mut self, value: crate::api::command::geometry::Rect) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterSetScissor(value))
    }
    fn raster_set_blend_constant(
        &mut self,
        value: crate::api::command::geometry::Color,
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterSetBlendConstant(value))
    }
    fn raster_set_stencil_reference(&mut self, value: u32) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterSetStencilReference(value))
    }
    fn raster_set_immediates(
        &mut self,
        write: &crate::api::command::record::ImmediateWrite,
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterSetImmediates(write))
    }
    fn raster_draw(
        &mut self,
        vertices: core::ops::Range<u32>,
        instances: core::ops::Range<u32>,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver.encode_typed(GlTypedCommand::RasterDraw {
            vertices,
            instances,
            uses,
        })
    }
    fn raster_draw_indexed(
        &mut self,
        indices: core::ops::Range<u32>,
        base_vertex: i32,
        instances: core::ops::Range<u32>,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver.encode_typed(GlTypedCommand::RasterDrawIndexed {
            indices,
            base_vertex,
            instances,
            uses,
        })
    }
    fn raster_draw_indirect(
        &mut self,
        arguments: &crate::api::resource::Buffer,
        arguments_offset: u64,
        draw_count: u32,
        stride: u32,
        count: Option<(&crate::api::resource::Buffer, u64, u32)>,
        indexed: bool,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterDrawIndirect {
                arguments,
                offset: arguments_offset,
                draw_count,
                stride,
                count,
                indexed,
                uses,
            })
    }
    fn raster_begin_query(
        &mut self,
        set: &crate::api::query::QuerySet,
        index: u32,
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterBeginQuery { set, index })
    }
    fn raster_end_query(&mut self, set: &crate::api::query::QuerySet, index: u32) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterEndQuery { set, index })
    }
    fn raster_write_timestamp(
        &mut self,
        set: &crate::api::query::QuerySet,
        index: u32,
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterWriteTimestamp { set, index })
    }
    fn raster_push_debug_group(&mut self, label: &str) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterPushDebugGroup(label))
    }
    fn raster_pop_debug_group(&mut self) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterPopDebugGroup)
    }
    fn raster_insert_debug_marker(&mut self, label: &str) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::RasterInsertDebugMarker(label))
    }
    fn raster_end(&mut self) -> RhiResult<()> {
        self.driver.encode_typed(GlTypedCommand::RasterEnd)
    }

    fn copy_external_image_to_texture(
        &mut self,
        copy: &crate::api::external::ExternalImageCopyDescriptor,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::CopyExternalImageToTexture { copy, uses })
    }
    fn clear_buffer(
        &mut self,
        buffer: &crate::api::resource::Buffer,
        range: crate::api::resource::BufferRange,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver.encode_typed(GlTypedCommand::ClearBuffer {
            buffer,
            range,
            uses,
        })
    }
    fn clear_texture(
        &mut self,
        texture: &crate::api::resource::Texture,
        subresources: crate::api::resource::subresource::TextureSubresourceRange,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver.encode_typed(GlTypedCommand::ClearTexture {
            texture,
            subresources,
            uses,
        })
    }
    fn copy_buffer(
        &mut self,
        copy: &crate::api::command::copy::BufferCopy,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::CopyBuffer { copy, uses })
    }
    fn copy_buffer_to_texture(
        &mut self,
        copy: &crate::api::command::copy::BufferTextureCopy,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::CopyBufferToTexture { copy, uses })
    }
    fn copy_texture_to_buffer(
        &mut self,
        copy: &crate::api::command::copy::BufferTextureCopy,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::CopyTextureToBuffer { copy, uses })
    }
    fn copy_texture(
        &mut self,
        copy: &crate::api::command::copy::TextureCopy,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::CopyTexture { copy, uses })
    }
    fn resolve_texture(
        &mut self,
        resolve: &crate::api::command::copy::TextureResolve,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::ResolveTexture { resolve, uses })
    }
    fn blit_texture(
        &mut self,
        blit: &crate::api::command::copy::TextureBlit,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::BlitTexture { blit, uses })
    }
    fn encode_upload(
        &mut self,
        upload: &crate::api::resource::transfer::UploadJob,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::Upload { upload, uses })
    }
    fn encode_readback(
        &mut self,
        ticket: &crate::api::resource::transfer::ReadbackTicket,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::Readback { ticket, uses })
    }
    fn encoder_write_timestamp(
        &mut self,
        set: &crate::api::query::QuerySet,
        index: u32,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver
            .encode_typed(GlTypedCommand::EncoderWriteTimestamp { set, index, uses })
    }
    fn resolve_query_set(
        &mut self,
        set: &crate::api::query::QuerySet,
        first_query: u32,
        query_count: u32,
        destination: &crate::api::resource::Buffer,
        destination_offset: u64,
        uses: &[crate::api::command::ResourceUse],
    ) -> RhiResult<()> {
        self.driver.encode_typed(GlTypedCommand::ResolveQuerySet {
            set,
            first_query,
            query_count,
            destination,
            destination_offset,
            uses,
        })
    }

    fn finish(
        self: Box<Self>,
    ) -> RhiResult<Box<dyn crate::api::command::backend::CommandBufferBackend>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.open = false;
        Ok(Box::new(GlImmediateCommandBuffer::new(self.order)))
    }
}

impl GlDevice {
    pub(super) fn new(
        backend: BackendKind,
        adapter: AdapterInfo,
        facts: CapabilityFacts,
        driver: Arc<dyn GlExecutionDriver>,
    ) -> Self {
        let liveness = Arc::new(Mutex::new(Liveness {
            status: DeviceStatus::Active,
            info: None,
            waiters: Vec::new(),
        }));
        let immediate_order = Arc::new(Mutex::new(GlImmediateOrder {
            open: false,
            next_encode: 1,
            next_submit: 1,
        }));
        let loss_sink: Arc<dyn GlLossSink> = Arc::new(GlLossAuthority {
            liveness: Arc::clone(&liveness),
        });
        driver.install_loss_sink(Arc::clone(&loss_sink));
        Self {
            backend,
            adapter,
            facts,
            object: ObjectId::next(),
            driver,
            liveness,
            loss_sink,
            immediate_order,
        }
    }

    /// Native/browser glue calls this once on `GL_CONTEXT_LOST`, WebGL
    /// `webglcontextlost`, or an equivalent terminal driver failure.  It wakes
    /// all completion waiters; later public verbs observe the stable loss state.
    pub(crate) fn mark_lost(&self, message: impl Into<String>) {
        let info = DeviceLossInfo::new(message.into());
        let authority = GlLossAuthority {
            liveness: Arc::clone(&self.liveness),
        };
        if authority.report(info.clone()) {
            self.driver.device_lost(&info);
        }
    }

    fn active(&self, operation: &'static str) -> RhiResult<()> {
        if let Some(info) = self.loss_info() {
            return Err(RhiError::new(RhiErrorKind::DeviceLost, info.message()).at(operation));
        }
        Ok(())
    }

    fn observe<T>(&self, result: RhiResult<T>, operation: &'static str) -> RhiResult<T> {
        if let Err(error) = &result {
            if error.kind() == RhiErrorKind::DeviceLost {
                self.mark_lost(error.to_string());
            }
        }
        result.map_err(|error| error.at(operation))
    }

    pub(crate) fn buffer_ref(buffer: &crate::api::resource::Buffer) -> RhiResult<GlBufferRef> {
        buffer
            .native()
            .as_any()
            .downcast_ref::<GlBuffer>()
            .map(|native| GlBufferRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(RhiErrorKind::Unsupported, "buffer has no GL native backing")
            })
    }
    pub(crate) fn texture_ref(texture: &crate::api::resource::Texture) -> RhiResult<GlTextureRef> {
        texture
            .native()
            .as_any()
            .downcast_ref::<GlTexture>()
            .map(|native| GlTextureRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "texture has no GL native backing",
                )
            })
    }
    pub(crate) fn view_ref(
        view: &crate::api::resource::TextureView,
    ) -> RhiResult<GlTextureViewRef> {
        view.native()
            .as_any()
            .downcast_ref::<GlTextureView>()
            .map(|native| GlTextureViewRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "texture view has no GL native backing",
                )
            })
    }
    pub(crate) fn sampler_ref(sampler: &crate::api::resource::Sampler) -> RhiResult<GlSamplerRef> {
        sampler
            .native()
            .as_any()
            .downcast_ref::<GlSampler>()
            .map(|native| GlSamplerRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "sampler has no GL native backing",
                )
            })
    }
    pub(crate) fn shader_ref(shader: &crate::api::shader::ShaderModule) -> RhiResult<GlShaderRef> {
        shader
            .native()
            .as_any()
            .downcast_ref::<GlShader>()
            .map(|native| GlShaderRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "shader module has no GL native backing",
                )
            })
    }
    pub(crate) fn bind_group_ref(
        group: &crate::api::binding::BindGroup,
    ) -> RhiResult<GlBindGroupRef> {
        group
            .native()
            .as_any()
            .downcast_ref::<GlBindGroup>()
            .map(|native| GlBindGroupRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "bind group has no GL native backing",
                )
            })
    }
    pub(crate) fn compute_pipeline_ref(
        pipeline: &crate::api::pipeline::ComputePipeline,
    ) -> RhiResult<GlComputePipelineRef> {
        pipeline
            .native()
            .as_any()
            .downcast_ref::<GlComputePipeline>()
            .map(|native| GlComputePipelineRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "compute pipeline has no GL native backing",
                )
            })
    }
    pub(crate) fn raster_pipeline_ref(
        pipeline: &crate::api::pipeline::RasterPipeline,
    ) -> RhiResult<GlRasterPipelineRef> {
        pipeline
            .native()
            .as_any()
            .downcast_ref::<GlRasterPipeline>()
            .map(|native| GlRasterPipelineRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "raster pipeline has no GL native backing",
                )
            })
    }
    pub(crate) fn query_set_ref(set: &crate::api::query::QuerySet) -> RhiResult<GlQuerySetRef> {
        set.native()
            .as_any()
            .downcast_ref::<GlQuerySet>()
            .map(|native| GlQuerySetRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "query set has no GL native backing",
                )
            })
    }
    fn bind_packet(
        descriptor: &crate::api::binding::BindGroupDescriptor,
    ) -> RhiResult<GlBindGroupPacket> {
        use crate::api::binding::BindingResource;
        let entries = descriptor.entries.iter().map(|entry| {
            let resource = match &entry.resource {
                BindingResource::Buffer(binding) => GlBindingResource::Buffer { buffer: Self::buffer_ref(&binding.buffer)?, offset: binding.range.offset, size: binding.range.size },
                BindingResource::Texture(view) => GlBindingResource::Texture(Self::view_ref(view)?),
                BindingResource::Sampler(sampler) => GlBindingResource::Sampler(Self::sampler_ref(sampler)?),
                BindingResource::BufferArray(bindings) => GlBindingResource::BufferArray(bindings.iter().map(|binding| Ok((Self::buffer_ref(&binding.buffer)?, binding.range.offset, binding.range.size))).collect::<RhiResult<_>>()?),
                BindingResource::TextureArray(views) => GlBindingResource::TextureArray(views.iter().map(Self::view_ref).collect::<RhiResult<_>>()?),
                BindingResource::SamplerArray(samplers) => GlBindingResource::SamplerArray(samplers.iter().map(Self::sampler_ref).collect::<RhiResult<_>>()?),
                BindingResource::AccelerationStructure(_) | BindingResource::AccelerationStructureArray(_) | BindingResource::ExternalTexture(_) => return Err(RhiError::new(RhiErrorKind::Unsupported, "this GL backend does not lower acceleration-structure or external-texture bindings")),
                _ => return Err(RhiError::new(RhiErrorKind::Unsupported, "this GL backend does not lower an unknown binding resource")),
            };
            Ok(GlBindGroupEntry { slot: entry.slot.get(), resource })
        }).collect::<RhiResult<_>>()?;
        Ok(GlBindGroupPacket { entries })
    }
    fn submission_plan(
        &self,
        request: &crate::api::submission::backend::SubmissionRequest<'_>,
    ) -> RhiResult<GlSubmissionPlan> {
        let mut batches = Vec::with_capacity(request.batches.len());
        for batch in request.batches {
            // Native GL work has already reached the context owner.
            // Consume only its completion token here; lowering a
            // portable command sequence at submit would execute every
            // draw/copy/dispatch twice.
            let all_immediate = batch.work.iter().all(|work| {
                work.native()
                    .as_any()
                    .downcast_ref::<GlImmediateCommandBuffer>()
                    .is_some()
            });
            if all_immediate {
                let mut order = self
                    .immediate_order
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                for work in &batch.work {
                    let token = work
                        .native()
                        .as_any()
                        .downcast_ref::<GlImmediateCommandBuffer>()
                        .expect("immediate GL work was type-checked above");
                    if token.order != order.next_submit {
                        return Err(RhiError::new(
                            RhiErrorKind::InvalidUsage,
                            "GL immediate work must be submitted in the order it was encoded",
                        )
                        .at("GlDevice::submit"));
                    }
                    token.take()?;
                    order.next_submit = order.next_submit.checked_add(1).ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::BackendFailure,
                            "GL command order space exhausted",
                        )
                        .at("GlDevice::submit")
                    })?;
                }
            } else {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "GL submission accepts only immediate native command buffers",
                )
                .at("GlDevice::submit"));
            }
            batches.push(GlSubmissionBatch {
                point: Some(batch.point),
            });
        }
        Ok(GlSubmissionPlan { batches })
    }
}

impl DeviceBackend for GlDevice {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn create_command_encoder(
        &self,
        _: &crate::api::command::RecorderDescriptor,
    ) -> RhiResult<Box<dyn crate::api::command::backend::CommandEncoderBackend>> {
        self.active("GlDevice::create_command_encoder")?;
        let order = {
            let mut state = self
                .immediate_order
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if state.open {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "GL immediate encoding permits only one open command recorder per device",
                )
                .at("GlDevice::create_command_encoder"));
            }
            state.open = true;
            let order = state.next_encode;
            state.next_encode = state.next_encode.checked_add(1).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "GL command order space exhausted",
                )
                .at("GlDevice::create_command_encoder")
            })?;
            order
        };
        Ok(Box::new(GlImmediateEncoder {
            driver: Arc::clone(&self.driver),
            order,
            state: Arc::clone(&self.immediate_order),
        }))
    }

    fn create_secondary_raster_encoder(
        &self,
        _: &crate::api::command::RecorderDescriptor,
    ) -> RhiResult<Box<dyn crate::api::command::backend::CommandEncoderBackend>> {
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "GL does not expose secondary raster command buffers",
        )
        .at("GlDevice::create_secondary_raster_encoder"))
    }

    fn backend_kind(&self) -> BackendKind {
        self.backend
    }
    fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter
    }
    fn capability_facts(&self) -> CapabilityFacts {
        self.facts.clone()
    }
    fn submission_capabilities(&self) -> SubmissionCapabilities {
        // GL has one ordered command stream. A discovered native compute route
        // shares that stream with raster and copy work; the portable layer
        // filters compute before this point when the context did not publish it.
        SubmissionCapabilities::new(vec![SubmissionLaneInfo::new(
            SubmissionLaneId::unscoped(0),
            SubmissionLaneClass::General,
            LaneWorkDomains::RASTER
                .union(LaneWorkDomains::COPY)
                .union(LaneWorkDomains::COMPUTE),
        )])
    }
    fn object_id(&self) -> ObjectId {
        self.object
    }
    fn status(&self) -> DeviceStatus {
        self.liveness
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .status
    }
    fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.liveness
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .info
            .clone()
    }
    fn poll(&self) -> RhiResult<()> {
        self.active("GlDevice::poll")?;
        self.observe(self.driver.poll(), "GlDevice::poll")
    }
    fn wait_idle(&self) -> RhiResult<()> {
        self.active("GlDevice::wait_idle")?;
        self.observe(self.driver.wait_idle(), "GlDevice::wait_idle")
    }
    fn presentation(&self) -> Option<&dyn crate::api::presentation::backend::PresentationBackend> {
        Some(self)
    }

    fn map_buffer(
        &self,
        buffer: &crate::api::resource::Buffer,
        mode: crate::api::resource::MapMode,
        range: crate::api::resource::BufferRange,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::MappingRequestBackend>> {
        self.active("GlDevice::map_buffer")?;
        let native = Self::buffer_ref(buffer).map_err(|error| error.at("GlDevice::map_buffer"))?;
        self.observe(
            self.driver.map_buffer(native, mode, range),
            "GlDevice::map_buffer",
        )
    }

    fn create_buffer(
        &self,
        descriptor: &crate::api::resource::buffer::BufferDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::BufferBackend>> {
        self.active("GlDevice::create_buffer")?;
        self.observe(
            self.driver.create_buffer(descriptor),
            "GlDevice::create_buffer",
        )
        .map(|name| {
            Box::new(GlBuffer {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn create_query_set(
        &self,
        descriptor: &crate::api::query::QuerySetDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::QuerySetBackend>> {
        self.active("GlDevice::create_query_set")?;
        self.observe(
            self.driver.create_query_set(descriptor),
            "GlDevice::create_query_set",
        )
        .map(|name| {
            Box::new(GlQuerySet {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn create_texture(
        &self,
        descriptor: &crate::api::resource::texture::TextureDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureBackend>> {
        self.active("GlDevice::create_texture")?;
        self.observe(
            self.driver.create_texture(descriptor),
            "GlDevice::create_texture",
        )
        .map(|name| {
            Box::new(GlTexture {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn create_texture_view(
        &self,
        texture: &crate::api::resource::Texture,
        descriptor: &crate::api::resource::view::TextureViewDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureViewBackend>> {
        self.active("GlDevice::create_texture_view")?;
        let native_texture = Self::texture_ref(texture)
            .map_err(|error| error.at("GlDevice::create_texture_view"))?;
        self.observe(
            self.driver.create_texture_view(native_texture, descriptor),
            "GlDevice::create_texture_view",
        )
        .map(|name| {
            Box::new(GlTextureView {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn create_sampler(
        &self,
        descriptor: &crate::api::resource::sampler::SamplerDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::SamplerBackend>> {
        self.active("GlDevice::create_sampler")?;
        self.observe(
            self.driver.create_sampler(descriptor),
            "GlDevice::create_sampler",
        )
        .map(|name| {
            Box::new(GlSampler {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn create_shader_request(
        &self,
        artifact: &crate::api::shader::ShaderArtifact,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::shader::backend::ShaderModuleBackend,
                >,
        >,
    > {
        self.active("GlDevice::create_shader")?;
        self.observe(
            self.driver.create_shader(artifact),
            "GlDevice::create_shader",
        )
        .map(|name| {
            Box::new(GlShader {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
        .map(crate::api::platform::backend::ready_creation_request)
    }
    fn create_bind_group(
        &self,
        descriptor: &crate::api::binding::BindGroupDescriptor,
    ) -> RhiResult<Box<dyn crate::api::binding::backend::BindGroupBackend>> {
        self.active("GlDevice::create_bind_group")?;
        let packet = Self::bind_packet(descriptor)
            .map_err(|error| error.at("GlDevice::create_bind_group"))?;
        self.observe(
            self.driver.create_bind_group(&packet),
            "GlDevice::create_bind_group",
        )
        .map(|name| {
            Box::new(GlBindGroup {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn create_compute_pipeline_request(
        &self,
        descriptor: &crate::api::pipeline::ComputePipelineDescriptor,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::pipeline::backend::ComputePipelineBackend,
                >,
        >,
    > {
        self.active("GlDevice::create_compute_pipeline")?;
        let packet = GlComputePipelinePacket {
            shader: Self::shader_ref(&descriptor.shader)
                .map_err(|error| error.at("GlDevice::create_compute_pipeline"))?,
            descriptor,
        };
        self.observe(
            self.driver.create_compute_pipeline(packet),
            "GlDevice::create_compute_pipeline",
        )
        .map(|name| {
            Box::new(GlComputePipeline {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
        .map(crate::api::platform::backend::ready_creation_request)
    }
    fn create_raster_pipeline_request(
        &self,
        descriptor: &crate::api::pipeline::RasterPipelineDescriptor,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::pipeline::backend::RasterPipelineBackend,
                >,
        >,
    > {
        self.active("GlDevice::create_raster_pipeline")?;
        let packet = GlRasterPipelinePacket {
            vertex: Self::shader_ref(&descriptor.vertex)
                .map_err(|error| error.at("GlDevice::create_raster_pipeline"))?,
            fragment: descriptor
                .fragment
                .as_ref()
                .map(Self::shader_ref)
                .transpose()
                .map_err(|error| error.at("GlDevice::create_raster_pipeline"))?,
            descriptor,
        };
        self.observe(
            self.driver.create_raster_pipeline(packet),
            "GlDevice::create_raster_pipeline",
        )
        .map(|name| {
            Box::new(GlRasterPipeline {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
        .map(crate::api::platform::backend::ready_creation_request)
    }
    fn submit(
        &self,
        request: &crate::api::submission::backend::SubmissionRequest<'_>,
    ) -> RhiResult<crate::api::submission::backend::SubmissionOutcome> {
        self.active("GlDevice::submit")?;
        let outcome = self.observe(
            self.driver.submit(self.submission_plan(request)?),
            "GlDevice::submit",
        )?;
        // The driver has now accepted the ordered GL stream and published its
        // completion fence. Presentation is consequently post-acceptance: a
        // browser/WGL failure must complete the receipt terminally through the
        // attachment rather than turn this accepted submission into `Err`.
        for present in request.presents {
            present.attachment.native().present(present.receipt);
        }
        Ok(outcome)
    }
    fn completion(&self, serial: u64) -> CompletionState {
        match self.loss_info() {
            Some(info) => CompletionState::DeviceLost(info),
            None => {
                let state = self.driver.completion(serial);
                if let CompletionState::DeviceLost(info) = &state {
                    self.mark_lost(info.message().to_owned());
                    return CompletionState::DeviceLost(
                        self.loss_info().expect("loss authority stored info"),
                    );
                }
                state
            }
        }
    }
    fn completion_or_register_waker(&self, serial: u64, waker: &Waker) -> CompletionState {
        if let Some(info) = self.loss_info() {
            return CompletionState::DeviceLost(info);
        }
        let state = self.driver.completion_or_register_waker(serial, waker);
        if let CompletionState::DeviceLost(info) = &state {
            self.mark_lost(info.message().to_owned());
            return CompletionState::DeviceLost(
                self.loss_info().expect("loss authority stored info"),
            );
        }
        if matches!(state, CompletionState::Pending) {
            self.liveness
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .waiters
                .push(waker.clone());
        }
        state
    }
}
