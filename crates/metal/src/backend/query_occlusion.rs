//! Preserve logical occlusion queries across native render-encoder segments.
//!
//! Metal resets visibility results per render encoder. Only a lazy GPU-count
//! split allocates another zeroed native visibility buffer. At logical pass
//! end, a GPU OR folds segment results into the original query buffer; Core's
//! zero/nonzero contract is preserved without host reads or draw replay.
use super::{Error, MetalCommandEncoder, Result};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLBlitCommandEncoder as _, MTLBuffer as _, MTLCommandBuffer as _,
    MTLCommandEncoder as _, MTLComputeCommandEncoder as _, MTLComputePipelineState as _,
    MTLDevice as _, MTLLibrary as _, MTLRenderPassDescriptor, MTLResourceOptions, MTLSize,
};

/// Device-owned immutable native helper; never stores recorded commands.
pub(in crate::backend) type VisibilityMergeCache = std::sync::Arc<
    std::sync::Mutex<
        Option<
            objc2::rc::Retained<
                objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
            >,
        >,
    >,
>;

pub(in crate::backend) fn split_occlusion(
    encoder: &mut MetalCommandEncoder,
    descriptor: &MTLRenderPassDescriptor,
) -> Result<()> {
    let Some(set) = &encoder.occlusion_set else {
        return Ok(());
    };
    let original = set.0.visibility.as_ref().ok_or(Error::InvalidState)?;
    let buffer = encoder
        .command
        .device()
        .newBufferWithLength_options(original.length(), MTLResourceOptions::StorageModePrivate)
        .ok_or(Error::OutOfMemory)?;
    let blit = encoder
        .command
        .blitCommandEncoder()
        .ok_or(Error::OutOfMemory)?;
    // A fresh segment must not contain stale results for queries inactive in it.
    blit.fillBuffer_range_value(
        &buffer,
        objc2_foundation::NSRange::new(0, original.length()),
        0,
    );
    blit.endEncoding();
    descriptor.setVisibilityResultBuffer(Some(&buffer));
    encoder.occlusion_segments.push(buffer);
    Ok(())
}

pub(in crate::backend) fn finish_occlusion_segments(
    encoder: &mut MetalCommandEncoder,
) -> Result<()> {
    if encoder.occlusion_segments.is_empty() {
        return Ok(());
    }
    let set = encoder.occlusion_set.as_ref().ok_or(Error::InvalidState)?;
    let original = set.0.visibility.as_ref().ok_or(Error::InvalidState)?;
    let words = u32::try_from(original.length() / 4).map_err(|_| Error::InvalidRange)?;
    let device = encoder.command.device();
    let pipeline = {
        let mut cache = encoder
            .visibility_merge_pipeline
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if let Some(pipeline) = &*cache {
            pipeline.clone()
        } else {
            let library = device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(
                        "#include <metal_stdlib>\nusing namespace metal;\n\
                 kernel void merge_visibility(device uint* dst [[buffer(0)]], \
                 const device uint* src [[buffer(1)]], constant uint& size [[buffer(2)]], \
                 uint i [[thread_position_in_grid]]) { if(i < size) dst[i] |= src[i]; }",
                    ),
                    None,
                )
                .map_err(|error| Error::Shader(format!("Metal visibility merge: {error:?}")))?;
            let function = library
                .newFunctionWithName(&NSString::from_str("merge_visibility"))
                .ok_or_else(|| Error::Shader("Metal visibility merge entry missing".into()))?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| {
                    Error::Shader(format!("Metal visibility merge pipeline: {error:?}"))
                })?;
            *cache = Some(pipeline.clone());
            pipeline
        }
    };
    let compute = encoder
        .command
        .computeCommandEncoder()
        .ok_or(Error::OutOfMemory)?;
    compute.setComputePipelineState(&pipeline);
    // SAFETY: the original and every segment have the same nonempty word-aligned
    // allocation; Metal retains the bound resources and copies the size bytes.
    unsafe {
        compute.setBuffer_offset_atIndex(Some(original), 0, 0);
        compute.setBytes_length_atIndex(std::ptr::NonNull::from(&words).cast(), 4, 2);
    }
    let group = pipeline
        .threadExecutionWidth()
        .max(1)
        .min(pipeline.maxTotalThreadsPerThreadgroup());
    for segment in &encoder.occlusion_segments {
        // SAFETY: this segment has exactly original.length() bytes and stays
        // live throughout encoding; the kernel bounds checks every word.
        unsafe { compute.setBuffer_offset_atIndex(Some(segment), 0, 1) };
        compute.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: words as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: group,
                height: 1,
                depth: 1,
            },
        );
        // Dispatches in one compute encoder need an explicit dependency for
        // their common read/write result buffer. All resources here are tracked.
        compute.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    }
    compute.endEncoding();
    encoder.occlusion_segments.clear();
    Ok(())
}
