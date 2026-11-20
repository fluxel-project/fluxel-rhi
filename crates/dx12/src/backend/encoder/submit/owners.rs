//! Boundary between issued DX12 work and fence-retired or quarantined owners.

use super::*;

fn retain_indeterminate(device: &Dx12Device, issued: Vec<Dx12RecordedWork>, error: &Error) {
    if !issued.is_empty() {
        for record in &issued {
            crate::backend::encoder::query::invalidate_ops(&record.encoder.query_ops);
            for receipt in &record.encoder.readback_receipts {
                receipt.fail(error);
            }
        }
        let mut direct = Vec::new();
        let mut compute = Vec::new();
        for record in issued {
            match record.lane.ordinal {
                0 => direct.push(record),
                1 => compute.push(record),
                _ => unreachable!("submission preflight validates DX12 lanes"),
            }
        }
        if !direct.is_empty() {
            device
                .0
                .in_flight
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push((u64::MAX, direct));
        }
        if !compute.is_empty() {
            device
                .0
                .multi_queue
                .in_flight
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push((u64::MAX, compute));
        }
    }
}

pub(super) fn retain_submitted(
    device: &Dx12Device,
    values: [Option<u64>; 2],
    issued: &mut Vec<Dx12RecordedWork>,
) -> [bool; 2] {
    let mut direct = Vec::new();
    let mut compute = Vec::new();
    for record in issued.drain(..) {
        let ordinal = record.lane.ordinal;
        let fence = crate::backend::queue::fence(device, ordinal).expect("validated lane");
        let value = values[ordinal as usize].expect("issued work has a terminal lane fence");
        crate::backend::ray::seal_compaction_receipts(&record, fence, value);
        match record.lane.ordinal {
            0 => direct.push(record),
            1 => compute.push(record),
            _ => unreachable!("submission preflight validates DX12 lanes"),
        }
    }
    let retained = [!direct.is_empty(), !compute.is_empty()];
    if retained[0] {
        let value = values[0].expect("direct work has a terminal direct fence");
        device
            .0
            .in_flight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((value, direct));
    }
    if retained[1] {
        let value = values[1].expect("compute work has a terminal compute fence");
        device
            .0
            .multi_queue
            .in_flight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((value, compute));
    }
    retained
}

pub(super) fn retain_sparse(
    device: &Dx12Device,
    value: u64,
    retained: Vec<crate::backend::sparse::SparseRetained>,
) {
    if !retained.is_empty() {
        device
            .0
            .sparse_in_flight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((value, retained));
        if value == u64::MAX {
            // Sparse mapping owners have no encoder to retain the Device.
            // Without a completion fence they need a separate owner in
            // quarantine even if the caller drops every public handle.
            crate::backend::encoder::fence_wait::retain_quarantined_owner(
                device,
                device.0.fence.clone(),
                value,
                None,
            );
        }
    }
}

pub(super) fn register_retirements(
    device: &Dx12Device,
    values: [Option<u64>; 2],
    retained_lanes: [bool; 2],
    retained_sparse: bool,
) -> Result<(), (u32, Error)> {
    let mut first_error = None;
    for ordinal in 0..2 {
        if retained_lanes[ordinal] || (ordinal == 0 && retained_sparse) {
            if let Err(error) = super::super::retirement::register(
                device,
                ordinal as u32,
                values[ordinal].expect("retained lane has a fence"),
            ) {
                // A failed registration cannot suppress the other lane's
                // independent completion observation.
                first_error.get_or_insert((ordinal as u32, error));
            }
        }
    }
    first_error.map_or(Ok(()), Err)
}

/// End indices of the owners covered by each lane's most recent signal.
#[derive(Default)]
pub(super) struct SignaledOwners {
    commands: [usize; 2],
    sparse: usize,
    pub(super) frontiers: [Option<(LaneId, api::StepId, u64)>; 2],
}

impl SignaledOwners {
    pub(super) fn record(
        &mut self,
        lane: LaneId,
        step: api::StepId,
        value: u64,
        commands: usize,
        sparse: usize,
    ) {
        let ordinal = lane.ordinal as usize;
        self.frontiers[ordinal] = Some((lane, step, value));
        self.commands[ordinal] = commands;
        if ordinal == 0 {
            self.sparse = sparse;
        }
    }
}

pub(super) fn retain_failed_handoff(
    device: &Dx12Device,
    values: [Option<u64>; 2],
    signaled: &SignaledOwners,
    issued: Vec<Dx12RecordedWork>,
    mut sparse: Vec<crate::backend::sparse::SparseRetained>,
    error: &Error,
) {
    let mut covered = Vec::new();
    let mut uncertain = Vec::new();
    for (index, record) in issued.into_iter().enumerate() {
        if index < signaled.commands[record.lane.ordinal as usize] {
            covered.push(record);
        } else {
            uncertain.push(record);
        }
    }
    let uncertain_sparse = sparse.split_off(signaled.sparse);
    retain_indeterminate(device, uncertain, error);
    retain_sparse(device, u64::MAX, uncertain_sparse);
    let retained_lanes = retain_submitted(device, values, &mut covered);
    let retained_sparse = !sparse.is_empty();
    if retained_sparse {
        retain_sparse(
            device,
            values[0].expect("signaled sparse work has a fence"),
            sparse,
        );
    }
    // Report the original handoff failure. Registration retains its own
    // fallback owner if observing a valid fence cannot be arranged.
    let _ = register_retirements(device, values, retained_lanes, retained_sparse);
}
