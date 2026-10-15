//! Direct Metal command recording.
//!
//! Every SPI call immediately reaches an `MTLCommandBuffer` or one active
//! native encoder. This module stores no CPU command representation and does
//! not infer resource hazards; Metal's tracked resources provide its native
//! same-queue ordering while Core supplies explicit transition intent.

use fluxel_rhi_core::api::{self, CommandBackend, LaneId, ResourceTransition, TextureRegion};

use super::{Metal, MetalBuffer, MetalDevice, MetalRasterPipeline};

mod raster;

#[cfg(target_vendor = "apple")]
type NativeIndexBuffer = (
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    u64,
    api::IndexFormat,
    u64,
);

pub struct MetalCommandEncoder {
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
        if lane.ordinal != 0 {
            return Err(api::Error::WrongLane);
        }
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLCommandQueue as _;
            // Metal's `commandBuffer` API returns a buffer that retains every
            // native object encoded into it. Do not replace this with the
            // unretained-reference queue constructor without adding explicit
            // GPU-completion ownership for encoder resources and uploads.
            let command = device
                ._state
                ._queue
                .commandBuffer()
                .ok_or(api::Error::OutOfMemory)?;
            Ok(MetalCommandEncoder {
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
        // here; ownership transfers and acceleration/aliasing remain unsupported.
        for op in ops {
            match op {
                ResourceTransition::Global { .. } => {}
                ResourceTransition::Buffer(barrier) if barrier.ownership.is_none() => {}
                ResourceTransition::Texture(barrier) if barrier.ownership.is_none() => {}
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
}
