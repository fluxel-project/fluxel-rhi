//! Resume a raster pass around Metal's compute-only indirect-count lowering.
//!
//! This retains native encoder state only. It never retains or replays draws:
//! earlier draws remain in the first render encoder, the ICB converter runs on
//! the same command buffer, and subsequent draws use the resumed encoder.

use fluxel_rhi_core::api::{self, RasterBackend as _};

use crate::backend::{Metal, MetalCommandEncoder};

/// A logical pass-statistics interval may span native render encoders.
/// Serialize their counter stores, including the final store before resolution.
/// Without this boundary, independent vertex stages could overlap and make
/// repeated writes to the logical end slot race despite attachment tracking.
pub(in crate::backend::encoder) fn statistics_boundary(
    encoder: &mut MetalCommandEncoder,
) -> api::Result<()> {
    if !encoder.raster_statistics || !encoder.raster_split {
        return Ok(());
    }
    use objc2::runtime::NSObjectProtocol as _;
    let Some(descriptor) = encoder.raster_descriptor.as_ref() else {
        return Ok(());
    };
    if !descriptor.respondsToSelector(objc2::sel!(sampleBufferAttachments)) {
        return Ok(());
    }
    // SAFETY: render pass descriptors always contain counter attachment zero.
    let counters = unsafe {
        descriptor
            .sampleBufferAttachments()
            .objectAtIndexedSubscript(0)
    };
    if counters.sampleBuffer().is_some() {
        super::super::transition::command_boundary(encoder)?;
    }
    Ok(())
}

pub(in crate::backend::encoder) fn end_for_indirect_count(
    encoder: &mut MetalCommandEncoder,
) -> api::Result<()> {
    use objc2_metal::{
        MTLCommandEncoder as _, MTLCounterDontSample, MTLLoadAction, MTLRenderCommandEncoder as _,
        MTLResource as _, MTLStorageMode, MTLStoreAction,
    };

    let descriptor = encoder
        .raster_descriptor
        .as_ref()
        .ok_or(api::Error::InvalidState)?
        .clone();
    // A memoryless attachment has no storage between render encoders. This is
    // an actual Metal restriction of render -> compute -> render lowering; a
    // framebuffer-only drawable remains valid and is deliberately accepted.
    let colors = descriptor.colorAttachments();
    for index in 0..8 {
        // SAFETY: Metal render-pass descriptors always expose eight color slots.
        let attachment = unsafe { colors.objectAtIndexedSubscript(index) };
        if attachment
            .texture()
            .is_some_and(|texture| texture.storageMode() == MTLStorageMode::Memoryless)
        {
            return Err(api::Error::Unsupported(
                "Metal indirect-count cannot resume a memoryless render attachment",
            ));
        }
    }
    if descriptor
        .depthAttachment()
        .texture()
        .is_some_and(|texture| texture.storageMode() == MTLStorageMode::Memoryless)
        || descriptor
            .stencilAttachment()
            .texture()
            .is_some_and(|texture| texture.storageMode() == MTLStorageMode::Memoryless)
    {
        return Err(api::Error::Unsupported(
            "Metal indirect-count cannot resume a memoryless depth/stencil attachment",
        ));
    }
    let render = encoder.render.take().ok_or(api::Error::InvalidState)?;
    for index in 0..8 {
        // SAFETY: Metal render-pass descriptors always expose eight color slots.
        let attachment = unsafe { colors.objectAtIndexedSubscript(index) };
        if attachment.texture().is_some() {
            // SAFETY: the attachment index is bounded by Metal's eight slots.
            unsafe { render.setColorStoreAction_atIndex(MTLStoreAction::Store, index) };
        }
    }
    if descriptor.depthAttachment().texture().is_some() {
        render.setDepthStoreAction(MTLStoreAction::Store);
    }
    if descriptor.stencilAttachment().texture().is_some() {
        render.setStencilStoreAction(MTLStoreAction::Store);
    }
    render.endEncoding();
    encoder.raster_split = true;
    statistics_boundary(encoder)?;

    // The query helper moves an active visibility result into a fresh segment
    // before the resumed descriptor is opened; the final pass combines every
    // segment back into Core's original query buffer.
    super::super::super::query::split_occlusion(encoder, &descriptor)?;

    for index in 0..8 {
        // SAFETY: Metal render-pass descriptors always expose eight color slots.
        let attachment = unsafe { colors.objectAtIndexedSubscript(index) };
        if attachment.texture().is_some() {
            attachment.setLoadAction(MTLLoadAction::Load);
        }
    }
    if descriptor.depthAttachment().texture().is_some() {
        descriptor
            .depthAttachment()
            .setLoadAction(MTLLoadAction::Load);
    }
    if descriptor.stencilAttachment().texture().is_some() {
        descriptor
            .stencilAttachment()
            .setLoadAction(MTLLoadAction::Load);
    }
    use objc2::runtime::NSObjectProtocol as _;
    if encoder.raster_statistics
        && descriptor.respondsToSelector(objc2::sel!(sampleBufferAttachments))
    {
        // Counter start samples belong to the first encoder. Retaining the end
        // sample lets the final resumed encoder overwrite the logical pass end.
        // Do not install counter indices when this pass has no statistics query.
        // SAFETY: attachment zero exists in every Metal render descriptor.
        let counters = unsafe {
            descriptor
                .sampleBufferAttachments()
                .objectAtIndexedSubscript(0)
        };
        if counters.sampleBuffer().is_some() {
            unsafe {
                counters.setStartOfVertexSampleIndex(MTLCounterDontSample);
                counters.setStartOfFragmentSampleIndex(MTLCounterDontSample);
            }
        }
    }
    Ok(())
}

pub(in crate::backend::encoder) fn resume_after_indirect_count(
    encoder: &mut MetalCommandEncoder,
) -> api::Result<()> {
    use objc2_metal::{MTLCommandBuffer as _, MTLRenderCommandEncoder as _};
    let descriptor = encoder
        .raster_descriptor
        .as_ref()
        .ok_or(api::Error::InvalidState)?;
    let resumed = encoder
        .command
        .renderCommandEncoderWithDescriptor(descriptor)
        .ok_or(api::Error::OutOfMemory)?;
    encoder.render = Some(resumed);

    if let Some(slot) = encoder.occlusion_slot {
        encoder
            .render
            .as_ref()
            .expect("resumed render encoder")
            .setVisibilityResultMode_offset(
                objc2_metal::MTLVisibilityResultMode::Boolean,
                slot as usize * 8,
            );
    }

    let pipeline = encoder
        .raster_pipeline
        .clone()
        .ok_or(api::Error::InvalidState)?;
    Metal::set_pipeline(encoder, &pipeline)?;
    if let Some(viewport) = encoder.raster_viewport {
        Metal::viewport(encoder, viewport)?;
    }
    if let Some(scissor) = encoder.raster_scissor {
        Metal::scissor(encoder, scissor)?;
    }
    Metal::blend_constant(encoder, encoder.raster_blend_constant)?;
    Metal::stencil_reference(encoder, encoder.raster_stencil_reference)?;
    Ok(())
}
