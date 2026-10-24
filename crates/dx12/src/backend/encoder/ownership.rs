//! Queue ownership handoff lowering. Submission waits remain the caller's responsibility.
use super::*;

/// A matched release/acquire pair transitions through COMMON on its respective
/// source/destination queue. Core and the submission plan own pairing and waits.
pub(super) fn states(
    encoder: &Dx12Encoder,
    transfer: api::OwnershipTransfer,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
    before_stages: api::StageMask,
    after_stages: api::StageMask,
) -> Result<(D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES), Error> {
    let (from, to, release) = match transfer {
        api::OwnershipTransfer::Release { from, to } => (from, to, true),
        api::OwnershipTransfer::Acquire { from, to } => (from, to, false),
    };
    if from.device != to.device
        || from.ordinal > 1
        || to.ordinal > 1
        || from == to
        || (release && (from.ordinal == 1) != encoder.is_compute)
        || (!release && (to.ordinal == 1) != encoder.is_compute)
    {
        return Err(Error::WrongLane);
    }
    // DXR dispatch and acceleration builds are native compute-list operations
    // too. Their inputs use the same NON_PIXEL/UAV states admitted below.
    let allowed_stages = api::StageMask::COPY.0
        | api::StageMask::INDIRECT.0
        | api::StageMask::COMPUTE.0
        | api::StageMask::RAY.0
        | api::StageMask::ACCELERATION.0;
    let allowed_states = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER.0
        | D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0
        | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0
        | D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT.0
        | D3D12_RESOURCE_STATE_COPY_SOURCE.0
        | D3D12_RESOURCE_STATE_COPY_DEST.0;
    for (lane, stage, state) in [(from, before_stages, before), (to, after_stages, after)] {
        if lane.ordinal == 1 && (stage.0 & !allowed_stages != 0 || state.0 & !allowed_states != 0) {
            return Err(Error::Unsupported("DX12 compute queue ownership scope"));
        }
    }
    let common = D3D12_RESOURCE_STATE_COMMON;
    Ok(if release {
        (before, common)
    } else {
        (common, after)
    })
}
