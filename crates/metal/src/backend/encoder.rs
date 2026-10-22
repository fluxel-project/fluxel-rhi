//! Direct Metal command recording.
//!
//! Every SPI call immediately reaches an `MTLCommandBuffer` or one active
//! native encoder. This module stores no CPU command representation and does
//! not infer resource hazards; Metal's tracked resources provide its native
//! same-queue ordering while Core supplies explicit transition intent.

use fluxel_rhi_core::api::{self, CommandBackend, LaneId, ResourceTransition, TextureRegion};

use super::{Metal, MetalBuffer, MetalDevice, MetalRasterPipeline};

mod clear;
pub(super) mod indirect_count;
mod raster;

#[cfg(target_vendor = "apple")]
type NativeIndexBuffer = (
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    u64,
    api::IndexFormat,
    u64,
);

pub struct MetalCommandEncoder {
    pub(crate) lane: u32,
    pub(super) query_ops: Vec<super::query::QueryOp>,
    #[cfg(target_vendor = "apple")]
    pub(super) external_buffers: Vec<super::MetalBuffer>,
    #[cfg(target_vendor = "apple")]
    pub(super) argument_arrays: Vec<std::sync::Arc<super::argument::ArgumentArray>>,
    #[cfg(target_vendor = "apple")]
    pub(super) external_textures: Vec<super::MetalTexture>,
    #[cfg(target_vendor = "apple")]
    pub(super) raster_end_timestamp: Option<(super::query::MetalQuerySet, u32)>,
    #[cfg(target_vendor = "apple")]
    pub(super) occlusion_set: Option<super::query::MetalQuerySet>,
    #[cfg(target_vendor = "apple")]
    pub(super) occlusion_active: bool,
    pub(super) pending_statistics: Option<(super::query::MetalQuerySet, u32, api::StatisticsPass)>,
    pub(crate) accelerations: Vec<super::ray::MetalAcceleration>,
    #[cfg(target_vendor = "apple")]
    pub(super) ray_owner: u64,
    #[cfg(target_vendor = "apple")]
    pub(super) ray_builds: Vec<std::sync::Arc<super::ray::BuildReservation>>,
    #[cfg(target_vendor = "apple")]
    pub(crate) compute_layout: Option<super::bindings::MetalPipelineLayout>,
    #[cfg(target_vendor = "apple")]
    pub(crate) compute_push: Vec<u8>,
    #[cfg(target_vendor = "apple")]
    pub(crate) compute_bindings: Vec<super::shader::variant::StageBindings>,
    #[cfg(target_vendor = "apple")]
    pub(crate) compute_groups: std::collections::BTreeMap<u32, (super::MetalBindGroup, Vec<u32>)>,
    #[cfg(target_vendor = "apple")]
    pub(crate) raster_group_count: usize,
    #[cfg(target_vendor = "apple")]
    pub(crate) raster_vertex_base: usize,
    #[cfg(target_vendor = "apple")]
    pub(crate) raster_vertex_buffers: Vec<Option<(MetalBuffer, u64, u64)>>,
    #[cfg(target_vendor = "apple")]
    pub(crate) raster_push_ranges: Vec<api::PushConstantRange>,
    #[cfg(target_vendor = "apple")]
    pub(crate) raster_push: [Vec<u8>; 4],
    #[cfg(target_vendor = "apple")]
    pub(crate) raster_read_only: (bool, bool),
    #[cfg(target_vendor = "apple")]
    pub(crate) raster_bindings: Vec<super::shader::variant::StageBindings>,
    #[cfg(target_vendor = "apple")]
    pub(crate) raster_groups: std::collections::BTreeMap<u32, (super::MetalBindGroup, Vec<u32>)>,
    #[cfg(target_vendor = "apple")]
    pub(crate) mesh_threads: Option<([u32; 3], [u32; 3])>,
    #[cfg(target_vendor = "apple")]
    pub(crate) command:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
    #[cfg(target_vendor = "apple")]
    pub(crate) prior_commands:
        Vec<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>>,
    #[cfg(target_vendor = "apple")]
    pub(crate) indirect_prepass: Option<
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
    >,
    #[cfg(target_vendor = "apple")]
    pub(super) prepared_draws: Vec<indirect_count::PreparedDraws>,
    #[cfg(target_vendor = "apple")]
    pub(crate) raster_pipeline_bound: bool,
    #[cfg(target_vendor = "apple")]
    pub(crate) indirect_count_enabled: bool,
    #[cfg(target_vendor = "apple")]
    pub(crate) render: Option<
        objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        >,
    >,
    #[cfg(target_vendor = "apple")]
    pub(crate) compute: Option<
        objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        >,
    >,
    #[cfg(target_vendor = "apple")]
    pub(crate) workgroup_size: Option<[u32; 3]>,
    #[cfg(target_vendor = "apple")]
    pub(crate) indirect_arguments:
        Vec<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    #[cfg(target_vendor = "apple")]
    pub(crate) indirect_dispatch_supported: bool,
    #[cfg(target_vendor = "apple")]
    pub(crate) raster_area: Option<api::Rect>,
    #[cfg(target_vendor = "apple")]
    pub(crate) raster_view_mask: Option<u32>,
    #[cfg(target_vendor = "apple")]
    pub(crate) raster_resolves: Vec<(
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>,
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>,
        api::Rect,
        Option<u32>,
    )>,
    #[cfg(target_vendor = "apple")]
    pub(crate) raster_signature: Option<(
        usize,
        Vec<Option<objc2_metal::MTLPixelFormat>>,
        Option<objc2_metal::MTLPixelFormat>,
    )>,
    #[cfg(target_vendor = "apple")]
    index_buffer: Option<NativeIndexBuffer>,
    #[cfg(target_vendor = "apple")]
    primitive_type: objc2_metal::MTLPrimitiveType,
    #[cfg(target_vendor = "apple")]
    strip_index_format: Option<api::IndexFormat>,
}

impl CommandBackend for Metal {
    fn begin_encoder(device: &MetalDevice, lane: LaneId) -> api::Result<MetalCommandEncoder> {
        if lane.ordinal > 1 {
            return Err(api::Error::WrongLane);
        }
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLCommandQueue as _;
            // Metal's `commandBuffer` API returns a buffer that retains every
            // native object encoded into it. Do not replace this with the
            // unretained-reference queue constructor without adding explicit
            // GPU-completion ownership for encoder resources and uploads.
            let queue = if lane.ordinal == 0 {
                &device._state._queue
            } else {
                &device._state.compute_queue
            };
            let command = queue.commandBuffer().ok_or(api::Error::OutOfMemory)?;
            Ok(MetalCommandEncoder {
                lane: lane.ordinal,
                query_ops: Vec::new(),
                external_buffers: Vec::new(),
                argument_arrays: Vec::new(),
                external_textures: Vec::new(),
                pending_statistics: None,
                raster_end_timestamp: None,
                occlusion_set: None,
                occlusion_active: false,
                accelerations: Vec::new(),
                ray_owner: {
                    static NEXT: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(1);
                    NEXT.fetch_update(
                        std::sync::atomic::Ordering::Relaxed,
                        std::sync::atomic::Ordering::Relaxed,
                        |n| n.checked_add(1),
                    )
                    .map_err(|_| api::Error::InvalidState)?
                },
                ray_builds: Vec::new(),
                mesh_threads: None,
                compute_layout: None,
                compute_push: Vec::new(),
                compute_bindings: Vec::new(),
                compute_groups: std::collections::BTreeMap::new(),
                raster_group_count: 0,
                raster_vertex_base: 31,
                raster_vertex_buffers: Vec::new(),
                raster_push_ranges: Vec::new(),
                raster_push: std::array::from_fn(|_| Vec::new()),
                raster_read_only: (false, false),
                raster_bindings: Vec::new(),
                raster_groups: std::collections::BTreeMap::new(),
                command,
                prior_commands: Vec::new(),
                indirect_prepass: None,
                prepared_draws: Vec::new(),
                raster_pipeline_bound: false,
                indirect_count_enabled: device._state.indirect_count_supported,
                render: None,
                compute: None,
                workgroup_size: None,
                indirect_arguments: Vec::new(),
                indirect_dispatch_supported: device._state.indirect_dispatch_supported,
                raster_signature: None,
                raster_area: None,
                raster_view_mask: None,
                raster_resolves: Vec::new(),
                index_buffer: None,
                primitive_type: objc2_metal::MTLPrimitiveType::Triangle,
                strip_index_format: None,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = device;
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn transition(
        encoder: &mut MetalCommandEncoder,
        ops: &[ResourceTransition<'_, Self>],
    ) -> api::Result<()> {
        for op in ops {
            if let ResourceTransition::Acceleration { before, after, .. } = op {
                if !acceleration_scope_supported(*before) || !acceleration_scope_supported(*after) {
                    return Err(api::Error::Unsupported(
                        "Metal acceleration transition scope",
                    ));
                }
            }
        }
        if ops.is_empty() {
            return Ok(());
        }
        // A command-buffer event forms a full native memory boundary. This is
        // needed for heap aliasing and argument-buffer resources, for which
        // Metal's automatic hazard tracking alone is insufficient. Metal has no
        // queue-family ownership; cross-queue order comes from the submitted plan.
        #[cfg(target_vendor = "apple")]
        {
            use objc2::runtime::NSObjectProtocol as _;
            use objc2_metal::{MTLCommandBuffer as _, MTLDevice as _};
            let device = encoder.command.device();
            if !device.respondsToSelector(objc2::sel!(newEvent)) {
                return Err(api::Error::Unsupported("Metal events unavailable"));
            }
            let event = device.newEvent().ok_or(api::Error::OutOfMemory)?;
            encoder.command.encodeSignalEvent_value(&event, 1);
            encoder.command.encodeWaitForEvent_value(&event, 1);
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = encoder;
            return Err(api::Error::Unsupported("Metal requires an Apple target"));
        }

        Ok(())
    }

    fn copy_buffer(
        encoder: &mut MetalCommandEncoder,
        src: &MetalBuffer,
        src_offset: u64,
        dst: &MetalBuffer,
        dst_offset: u64,
        bytes: u64,
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{
                MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _,
            };
            if !src.usage.contains(api::BufferUsage::COPY_SRC)
                || !dst.usage.contains(api::BufferUsage::COPY_DST)
                || bytes == 0
                || src_offset
                    .checked_add(bytes)
                    .is_none_or(|end| end > src.size())
                || dst_offset
                    .checked_add(bytes)
                    .is_none_or(|end| end > dst.size())
            {
                return Err(api::Error::InvalidRange);
            }
            if std::ptr::eq(&*src.raw, &*dst.raw)
                && src_offset < dst_offset + bytes
                && dst_offset < src_offset + bytes
            {
                return Err(api::Error::InvalidRange);
            }
            if (src_offset | dst_offset | bytes) & 3 != 0 {
                let source = usize::try_from(src_offset).map_err(|_| api::Error::InvalidRange)?;
                let destination =
                    usize::try_from(dst_offset).map_err(|_| api::Error::InvalidRange)?;
                let length = usize::try_from(bytes).map_err(|_| api::Error::InvalidRange)?;
                return super::copy::copy_bytes(
                    &encoder.command,
                    &src.raw,
                    source,
                    &dst.raw,
                    destination,
                    length,
                );
            }
            let blit = encoder
                .command
                .blitCommandEncoder()
                .ok_or(api::Error::OutOfMemory)?;
            // SAFETY: offsets and byte count were range-checked against both
            // retained native buffers above; Metal retains encoded resources.
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &src.raw,
                    src_offset as usize,
                    &dst.raw,
                    dst_offset as usize,
                    bytes as usize,
                );
            }
            blit.endEncoding();
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, src, src_offset, dst, dst_offset, bytes);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn resolve(
        encoder: &mut MetalCommandEncoder,
        src: TextureRegion<'_, Metal>,
        dst: TextureRegion<'_, Metal>,
    ) -> api::Result<()> {
        super::copy::resolve(encoder, src, dst)
    }

    fn copy_texture(
        encoder: &mut MetalCommandEncoder,
        src: TextureRegion<'_, Self>,
        dst: TextureRegion<'_, Self>,
    ) -> api::Result<()> {
        super::copy::copy_texture(encoder, src, dst)
    }

    fn buffer_to_texture(
        encoder: &mut MetalCommandEncoder,
        src: &MetalBuffer,
        layout: api::TexelCopyBufferLayout,
        dst: TextureRegion<'_, Self>,
    ) -> api::Result<()> {
        super::copy::buffer_to_texture(encoder, src, layout, dst)
    }

    fn texture_to_buffer(
        encoder: &mut MetalCommandEncoder,
        src: TextureRegion<'_, Self>,
        dst: &MetalBuffer,
        layout: api::TexelCopyBufferLayout,
    ) -> api::Result<()> {
        super::copy::texture_to_buffer(encoder, src, dst, layout)
    }

    fn write_buffer(
        encoder: &mut MetalCommandEncoder,
        dst: &MetalBuffer,
        offset: u64,
        bytes: &[u8],
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{
                MTLBlitCommandEncoder as _, MTLBuffer as _, MTLCommandBuffer as _,
                MTLCommandEncoder as _, MTLDevice as _, MTLResourceOptions,
            };
            if offset
                .checked_add(bytes.len() as u64)
                .is_none_or(|end| end > dst.size())
            {
                return Err(api::Error::InvalidRange);
            }
            if bytes.is_empty() {
                return Ok(());
            }
            let staging = encoder
                .command
                .device()
                .newBufferWithLength_options(bytes.len(), MTLResourceOptions::StorageModeShared)
                .ok_or(api::Error::OutOfMemory)?;
            let pointer = staging.contents().cast::<u8>().as_ptr();
            // SAFETY: the shared staging allocation is at least bytes.len()
            // bytes, and both pointers are valid and nonoverlapping.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer, bytes.len()) };
            if offset & 3 != 0 || bytes.len() & 3 != 0 {
                let destination = usize::try_from(offset).map_err(|_| api::Error::InvalidRange)?;
                return super::copy::copy_bytes(
                    &encoder.command,
                    &staging,
                    0,
                    &dst.raw,
                    destination,
                    bytes.len(),
                );
            }
            let blit = encoder
                .command
                .blitCommandEncoder()
                .ok_or(api::Error::OutOfMemory)?;
            // SAFETY: the staging buffer length and destination range were
            // checked above; Metal retains the staging buffer through execution.
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &staging,
                    0,
                    &dst.raw,
                    offset as usize,
                    bytes.len(),
                );
            }
            blit.endEncoding();
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, dst, offset, bytes);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn write_texture(
        encoder: &mut MetalCommandEncoder,
        dst: TextureRegion<'_, Self>,
        layout: api::TexelCopyBufferLayout,
        bytes: &[u8],
    ) -> api::Result<()> {
        super::copy::write_texture(encoder, dst, layout, bytes)
    }

    fn clear_texture(
        encoder: &mut MetalCommandEncoder,
        texture: &super::MetalTexture,
        mips: std::ops::Range<u32>,
        layers: std::ops::Range<u32>,
        value: api::TextureClearValue,
    ) -> api::Result<()> {
        clear::clear_texture(encoder, texture, mips, layers, value)
    }
}

fn acceleration_scope_supported(scope: api::AccessScope) -> bool {
    // Native event synchronization is a full boundary and accepts broad scopes
    // such as ALL_COMMANDS / ALL without requiring stage-specific lowering.
    scope.accesses.0 == 0 || scope.stages.0 != 0
}

#[cfg(test)]
mod acceleration_transition_tests {
    use super::{acceleration_scope_supported, api};

    #[test]
    fn accepts_build_and_supported_shader_read_scopes() {
        let scope = |stages, accesses| api::AccessScope { stages, accesses };
        assert!(acceleration_scope_supported(scope(
            api::StageMask::ACCELERATION,
            api::AccessMask::ACCEL_WRITE
        )));
        assert!(acceleration_scope_supported(scope(
            api::StageMask::COMPUTE,
            api::AccessMask::ACCEL_READ
        )));
        assert!(acceleration_scope_supported(scope(
            api::StageMask::FRAGMENT,
            api::AccessMask::ACCEL_READ
        )));
        assert!(!acceleration_scope_supported(scope(
            api::StageMask::EMPTY,
            api::AccessMask::ACCEL_WRITE
        )));
    }
}
