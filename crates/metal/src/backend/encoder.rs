//! Direct Metal command recording.
//!
//! Every SPI call immediately reaches an `MTLCommandBuffer` or one active
//! native encoder. This module stores no CPU command representation and does
//! not infer resource hazards; Metal's tracked resources provide its native
//! same-queue ordering while Core supplies explicit transition intent.

use fluxel_rhi_core::api::{self, CommandBackend, LaneId, ResourceTransition, TextureRegion};

use super::{Metal, MetalBuffer, MetalDevice, MetalRasterPipeline};

mod clear;
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
    #[cfg(target_vendor = "apple")]
    pub(super) wait_slots:
        Vec<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>>>,
    pub(super) query_ops: Vec<super::query::QueryOp>,
    pub(crate) accelerations: Vec<super::ray::MetalAcceleration>,
    #[cfg(target_vendor = "apple")]
    pub(crate) mesh_threads: Option<([u32; 3], [u32; 3])>,
    #[cfg(target_vendor = "apple")]
    pub(crate) command:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
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
    pub(crate) raster_signature: Option<(
        usize,
        objc2_metal::MTLPixelFormat,
        Option<objc2_metal::MTLPixelFormat>,
    )>,
    #[cfg(target_vendor = "apple")]
    index_buffer: Option<NativeIndexBuffer>,
}

impl CommandBackend for Metal {
    fn begin_encoder(device: &MetalDevice, lane: LaneId) -> api::Result<MetalCommandEncoder> {
        if lane.ordinal > 1 {
            return Err(api::Error::WrongLane);
        }
        #[cfg(target_vendor = "apple")]
        {
            use objc2::runtime::NSObjectProtocol as _;
            use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _, MTLDevice as _};
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
            let mut wait_slots = Vec::new();
            if device
                ._state
                ._device
                .respondsToSelector(objc2::sel!(newSharedEvent))
            {
                for _ in 0..4 {
                    let event = device
                        ._state
                        ._device
                        .newSharedEvent()
                        .ok_or(api::Error::OutOfMemory)?;
                    let native =
                        objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(
                            &*event,
                        );
                    command.encodeWaitForEvent_value(native, 1);
                    wait_slots.push(event);
                }
            }
            Ok(MetalCommandEncoder {
                lane: lane.ordinal,
                wait_slots,
                query_ops: Vec::new(),
                accelerations: Vec::new(),
                mesh_threads: None,
                command,
                render: None,
                compute: None,
                workgroup_size: None,
                indirect_arguments: Vec::new(),
                indirect_dispatch_supported: device._state.indirect_dispatch_supported,
                raster_signature: None,
                index_buffer: None,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = device;
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn transition(
        _: &mut MetalCommandEncoder,
        ops: &[ResourceTransition<'_, Self>],
    ) -> api::Result<()> {
        // Metal's default tracked resources perform same-queue hazard ordering
        // natively. Explicit Core transitions therefore need no state mutation
        // here; ownership transfers and aliasing remain unsupported. Acceleration
        // structures are allocated as tracked resources, and Metal orders their
        // builds and subsequent uses across encoders on the same queue.
        for op in ops {
            match op {
                ResourceTransition::Global { .. } => {}
                ResourceTransition::Buffer(barrier) if barrier.ownership.is_none() => {}
                ResourceTransition::Texture(barrier) if barrier.ownership.is_none() => {}
                ResourceTransition::Acceleration { before, after, .. }
                    if acceleration_scope_supported(*before)
                        && acceleration_scope_supported(*after) => {}
                _ => return Err(api::Error::Unsupported("Metal ownership/alias transition")),
            }
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
            if src_offset
                .checked_add(bytes)
                .is_none_or(|end| end > src.size())
                || dst_offset
                    .checked_add(bytes)
                    .is_none_or(|end| end > dst.size())
            {
                return Err(api::Error::InvalidRange);
            }
            if bytes == 0 {
                return Ok(());
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
    let stages = api::StageMask::ACCELERATION.0 | api::StageMask::COMPUTE.0;
    let accesses = api::AccessMask::ACCEL_READ.0
        | api::AccessMask::ACCEL_WRITE.0
        | api::AccessMask::SHADER_READ.0;
    scope.stages.0 & !stages == 0
        && scope.accesses.0 & !accesses == 0
        && (scope.accesses.0 == 0 || scope.stages.0 != 0)
}

#[cfg(test)]
mod acceleration_transition_tests {
    use super::{acceleration_scope_supported, api};

    #[test]
    fn accepts_build_and_compute_read_scopes_only() {
        let scope = |stages, accesses| api::AccessScope { stages, accesses };
        assert!(acceleration_scope_supported(scope(
            api::StageMask::ACCELERATION,
            api::AccessMask::ACCEL_WRITE
        )));
        assert!(acceleration_scope_supported(scope(
            api::StageMask::COMPUTE,
            api::AccessMask::ACCEL_READ
        )));
        assert!(!acceleration_scope_supported(scope(
            api::StageMask::FRAGMENT,
            api::AccessMask::ACCEL_READ
        )));
        assert!(!acceleration_scope_supported(scope(
            api::StageMask::EMPTY,
            api::AccessMask::ACCEL_WRITE
        )));
    }
}
