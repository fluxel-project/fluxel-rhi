//! Explicit Core-transition validation and narrow Metal synchronization.
//!
//! Ordered encoders in one command buffer preserve dependencies for tracked
//! Metal resources. Placement heaps default to *untracked*, however, so this
//! module queries each transitioned native resource and keeps a command-buffer
//! event boundary for untracked, global, and aliasing transitions. It does not
//! recreate resource state tracking.

use super::MetalCommandEncoder;
use fluxel_rhi_core::api::{self, Error, ResourceTransition, Result};

pub(super) fn record(
    encoder: &mut MetalCommandEncoder,
    ops: &[ResourceTransition<'_, super::Metal>],
) -> Result<()> {
    // Do not issue a partial native barrier sequence when a later element is
    // malformed. Core intentionally supplies independent explicit barriers;
    // validating the full batch first preserves the encoder's all-or-error
    // recording behavior.
    validate_all(ops)?;
    if ops.is_empty() {
        return Ok(());
    }

    let global_or_alias = ops.iter().any(|op| {
        matches!(
            op,
            ResourceTransition::Global { .. } | ResourceTransition::Aliasing { .. }
        )
    });

    #[cfg(target_vendor = "apple")]
    {
        if encoder.render.is_some() || encoder.compute.is_some() {
            return Err(Error::InvalidState);
        }
        let needs_command_boundary = global_or_alias || contains_untracked_resource(ops);
        if !needs_command_boundary {
            // Native command-buffer ordering covers tracked resource-specific
            // transitions. Cross-queue execution order is installed by
            // MultiQueue submission, not by this local barrier lowering.
            return Ok(());
        }
        command_boundary(encoder)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (encoder, global_or_alias);
        Err(Error::Unsupported("Metal requires an Apple target"))
    }
}

fn validate_all(ops: &[ResourceTransition<'_, super::Metal>]) -> Result<()> {
    for op in ops {
        match op {
            ResourceTransition::Global { .. } | ResourceTransition::Aliasing { .. } => {}
            ResourceTransition::Buffer(barrier) => {
                // SAFETY: CommandEncoder validated resource/device provenance
                // before invoking the backend; this borrow is used only for
                // immutable range validation.
                let buffer = unsafe { barrier.target.buffer.native_ref() };
                validate_buffer_range(buffer.size(), &barrier.target.range)?;
            }
            ResourceTransition::Texture(barrier) => {
                // SAFETY: same Core provenance guarantee as buffer barriers.
                let texture = unsafe { barrier.target.texture.native_ref() };
                validate_texture_range(
                    texture.desc(),
                    barrier.target.aspects,
                    &barrier.target.mips,
                    &barrier.target.layers,
                )?;
            }
            ResourceTransition::Acceleration { before, after, .. } => {
                if !acceleration_scope_supported(*before) || !acceleration_scope_supported(*after) {
                    return Err(Error::Unsupported("Metal acceleration transition scope"));
                }
            }
        }
    }
    Ok(())
}

fn validate_buffer_range(size: u64, range: &std::ops::Range<u64>) -> Result<()> {
    if range.start >= range.end || range.end > size {
        Err(Error::InvalidRange)
    } else {
        Ok(())
    }
}

fn validate_texture_range(
    desc: &api::TextureDesc,
    aspects: api::TextureAspects,
    mips: &std::ops::Range<u32>,
    layers: &std::ops::Range<u32>,
) -> Result<()> {
    let available = texture_aspects(desc.format);
    if aspects.0 == 0
        || aspects.0 & !available.0 != 0
        || mips.start >= mips.end
        || mips.end > desc.mip_levels
        || layers.start >= layers.end
        || layers.end > desc.array_layers
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

const fn texture_aspects(format: api::TextureFormat) -> api::TextureAspects {
    match format {
        api::TextureFormat::Depth16Unorm
        | api::TextureFormat::Depth24Plus
        | api::TextureFormat::Depth32Float => api::TextureAspects::DEPTH,
        api::TextureFormat::Stencil8 => api::TextureAspects::STENCIL,
        api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8 => {
            api::TextureAspects(api::TextureAspects::DEPTH.0 | api::TextureAspects::STENCIL.0)
        }
        _ => api::TextureAspects::COLOR,
    }
}

fn acceleration_scope_supported(scope: api::AccessScope) -> bool {
    // Command-buffer events use a full memory boundary. A nonempty access
    // still needs a graph-declared producing/consuming stage.
    scope.accesses.0 == 0 || scope.stages.0 != 0
}

/// Returns true conservatively if any resource cannot prove tracked hazard
/// mode. Device-created resources normally report `Tracked`; placement heap
/// resources normally report `Untracked`, and must retain explicit ordering.
#[cfg(target_vendor = "apple")]
fn contains_untracked_resource(ops: &[ResourceTransition<'_, super::Metal>]) -> bool {
    use objc2::runtime::NSObjectProtocol as _;
    use objc2_metal::{MTLHazardTrackingMode, MTLResource as _};

    let tracked = |resource: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLResource>| {
        resource.respondsToSelector(objc2::sel!(hazardTrackingMode))
            && resource.hazardTrackingMode() == MTLHazardTrackingMode::Tracked
    };
    ops.iter().any(|op| match op {
        ResourceTransition::Buffer(barrier) => {
            // SAFETY: full batch validation already established the buffer's
            // Core provenance; conversion only borrows its native resource.
            let buffer = unsafe { barrier.target.buffer.native_ref() };
            let resource = objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLResource>::from_ref(
                &*buffer.raw,
            );
            !tracked(resource)
        }
        ResourceTransition::Texture(barrier) => {
            // SAFETY: same provenance and immutable borrow as buffer above.
            let texture = unsafe { barrier.target.texture.native_ref() };
            let resource = objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLResource>::from_ref(
                &*texture.raw,
            );
            !tracked(resource)
        }
        ResourceTransition::Acceleration { target, .. } => {
            // SAFETY: Core checked the acceleration structure's device.
            let acceleration = unsafe { target.native_ref() };
            let resource = objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLResource>::from_ref(
                &*acceleration.raw,
            );
            !tracked(resource)
        }
        ResourceTransition::Global { .. } | ResourceTransition::Aliasing { .. } => false,
    })
}

#[cfg(target_vendor = "apple")]
pub(super) fn command_boundary(encoder: &mut MetalCommandEncoder) -> Result<()> {
    use objc2::runtime::NSObjectProtocol as _;
    use objc2_metal::{MTLCommandBuffer as _, MTLDevice as _};

    let value = encoder
        .transition_value
        .checked_add(1)
        .ok_or(Error::InvalidState)?;
    if encoder.transition_event.is_none() {
        let device = encoder.command.device();
        if !device.respondsToSelector(objc2::sel!(newEvent)) {
            return Err(Error::Unsupported("Metal events unavailable"));
        }
        encoder.transition_event = Some(device.newEvent().ok_or(Error::OutOfMemory)?);
    }
    let event = encoder
        .transition_event
        .as_deref()
        .expect("created transition event");
    encoder.command.encodeSignalEvent_value(event, value);
    encoder.command.encodeWaitForEvent_value(event, value);
    encoder.transition_value = value;
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/backend/encoder/transition.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/backend/encoder/transition/acceleration_transition_tests.rs"]
mod acceleration_transition_tests;
