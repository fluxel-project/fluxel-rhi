//! Lock-free immutable packet checks for DX12 submission.
//!
//! This module only inspects a packet and clones native fence references used
//! to model declared waits. Query admission, device-removal observation, and
//! every native handoff remain under `query_submission_gate` in `submit.rs`.

use super::*;

#[cfg(all(test, windows, feature = "hardware-tests"))]
#[path = "../../../tests/backend/encoder/submit/preflight_bench.rs"]
mod benchmark;

pub(super) struct PreparedPacket {
    pub(super) coverage: Vec<[Option<usize>; 2]>,
    pub(super) external_waits: Vec<Vec<(ID3D12Fence, u64)>>,
}

pub(super) fn packet(
    device: &Dx12Device,
    packet: &api::MultiQueuePacket<Dx12>,
) -> Result<PreparedPacket, Error> {
    for (index, step) in packet.work.steps.iter().enumerate() {
        let desc = &packet.work.plan.steps()[index];
        if desc.lane.device != packet.work.plan.id().device || desc.lane.ordinal > 1 {
            return Err(Error::WrongLane);
        }
        for wait in &desc.waits {
            if wait.previous.plan != packet.work.plan.id() || wait.previous.index as usize >= index
            {
                return Err(Error::InvalidState);
            }
        }
        match &step.payload {
            StepPayload::Commands(work) => {
                for record in work {
                    // SAFETY: Core checked ownership and device identity; a
                    // finished recorded work is immutable until handoff.
                    let native = unsafe { record.native_ref() };
                    if !native.encoder.closed
                        || native.lane != desc.lane
                        || native.encoder.closed_segments.len()
                            != native.encoder.queue_boundaries.len()
                    {
                        return Err(Error::InvalidState);
                    }
                }
            }
            StepPayload::SparseBind(binds) => {
                if desc.lane.ordinal != 0 {
                    return Err(Error::Unsupported(
                        "DX12 sparse mappings require direct lane",
                    ));
                }
                if !device.0.sparse_supported || binds.is_empty() {
                    return Err(Error::InvalidRange);
                }
                binds
                    .iter()
                    .try_for_each(super::super::super::sparse::validate_bind)?;
            }
            StepPayload::Present(_) if desc.lane.ordinal != 0 || index == 0 => {
                return Err(Error::InvalidState);
            }
            _ => {}
        }
    }

    // Track only declared order. This never derives new synchronization: the
    // result is later supplied to mutable query admission under the gate.
    let mut coverage = Vec::<[Option<usize>; 2]>::new();
    let mut external_waits = Vec::<Vec<(ID3D12Fence, u64)>>::new();
    let mut lanes: [Option<usize>; 2] = [None; 2];
    for (index, desc) in packet.work.plan.steps().iter().enumerate() {
        let lane = desc.lane.ordinal as usize;
        let mut frontier = lanes[lane]
            .map(|previous| coverage[previous])
            .unwrap_or([None; 2]);
        let mut native_frontier = lanes[lane]
            .map(|previous| external_waits[previous].clone())
            .unwrap_or_default();
        for wait in &desc.waits {
            for (slot, imported) in frontier
                .iter_mut()
                .zip(coverage[wait.previous.index as usize])
            {
                *slot = (*slot).max(imported);
            }
            for (fence, value) in &external_waits[wait.previous.index as usize] {
                if let Some((_, current)) = native_frontier
                    .iter_mut()
                    .find(|(known, _)| known.as_raw() == fence.as_raw())
                {
                    *current = (*current).max(*value);
                } else {
                    native_frontier.push((fence.clone(), *value));
                }
            }
        }
        for wait in packet
            .waits
            .iter()
            .filter(|wait| wait.signal.consumer().index as usize == index)
        {
            // SAFETY: Core validated token ownership; only its immutable native
            // fence/value are cloned into this local preflight representation.
            let signal = unsafe { wait.signal.native_ref() };
            if let Some((_, value)) = native_frontier
                .iter_mut()
                .find(|(known, _)| known.as_raw() == signal.fence.as_raw())
            {
                *value = (*value).max(signal.value);
            } else {
                native_frontier.push((signal.fence.clone(), signal.value));
            }
        }
        frontier[lane] = Some(index);
        coverage.push(frontier);
        external_waits.push(native_frontier);
        lanes[lane] = Some(index);
    }
    Ok(PreparedPacket {
        coverage,
        external_waits,
    })
}
