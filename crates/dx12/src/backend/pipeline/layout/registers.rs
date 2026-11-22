//! Native register assignment and stage visibility validation.

use fluxel_rhi_core::api::{self, Error, Result};
use std::collections::BTreeMap;

use super::{IMMEDIATE_REGISTER, IMMEDIATE_SPACE, PipelineLayoutState};

impl PipelineLayoutState {
    /// DXR uses the compute root signature as its global root signature.
    pub(in crate::backend) fn global_ray_cbv_occupied(&self, space: u32, binding: u32) -> bool {
        self.register_map
            .iter()
            .any(|&(group, _, register, class, count)| {
                space == group && class == 0 && register <= binding && binding - register < count
            })
            || (space == u32::from(IMMEDIATE_SPACE)
                && binding == IMMEDIATE_REGISTER
                && self
                    .push_constants
                    .as_ref()
                    .is_some_and(|root| root.compute.is_some()))
    }

    pub(crate) fn requires_native_binding_validation(&self, stage: api::ShaderStage) -> bool {
        self.explicit_native_registers
            || self
                .binding_entries
                .iter()
                .any(|(_, entry)| !entry.visibility.contains(&stage))
    }

    /// An AS shader register span must fit a visible, typed acceleration slot.
    /// Using a generic SRV range can otherwise admit an unrelated view type.
    pub(crate) fn validate_native_acceleration_binding(
        &self,
        stage: api::ShaderStage,
        space: u32,
        registers: std::ops::Range<u64>,
    ) -> Result<()> {
        let unbounded = registers.end == u64::from(u32::MAX) + 1;
        let covered = self
            .register_map
            .iter()
            .any(|&(group, binding, start, class, count)| {
                let start = u64::from(start);
                if group != space || class != 1 || registers.start < start {
                    return false;
                }
                self.binding_entries.iter().any(|(group, entry)| {
                    *group == space
                        && entry.binding == binding
                        && matches!(entry.ty, api::BindingType::AccelerationStructure)
                        && entry.visibility.contains(&stage)
                        && if unbounded {
                            registers.start == start
                                && matches!(entry.count, api::BindingCount::RuntimeSized { .. })
                        } else {
                            registers.end <= start + u64::from(count)
                        }
                })
            });
        if !covered {
            return Err(Error::Shader(format!(
                "DX12 acceleration register range {registers:?} in space {space} lacks a matching {stage:?} layout binding"
            )));
        }
        Ok(())
    }

    pub(crate) fn validate_native_binding(
        &self,
        stage: api::ShaderStage,
        space: u32,
        class: u8,
        mut registers: std::ops::Range<u64>,
    ) -> Result<()> {
        if self.explicit_native_registers && registers.end == u64::from(u32::MAX) + 1 {
            let capacity = self
                .runtime_binding_capacities
                .iter()
                .find_map(|&(group, binding, cap)| {
                    self.register_map
                        .iter()
                        .any(|&(g, b, register, register_class, _)| {
                            g == group
                                && b == binding
                                && g == space
                                && register_class == class
                                && u64::from(register) == registers.start
                        })
                        .then_some(cap)
                })
                .ok_or_else(|| {
                    Error::Shader(
                        "DX12 native runtime register range requires a mapped capacity".into(),
                    )
                })?;
            registers.end = registers.start + u64::from(capacity);
        }
        let immediate = u64::from(IMMEDIATE_REGISTER);
        let mut covered = u64::from(
            class == 0
                && space == u32::from(IMMEDIATE_SPACE)
                && registers.contains(&immediate)
                && self
                    .push_constants
                    .as_ref()
                    .is_some_and(|root| root.has_stage(stage)),
        );
        for &(group, binding, register, register_class, count) in &self.register_map {
            let register = u64::from(register);
            if group != space || register_class != class {
                continue;
            }
            let start = register.max(registers.start);
            let end = (register + u64::from(count)).min(registers.end);
            if start >= end {
                continue;
            }
            if self.binding_entries.iter().any(|(space, entry)| {
                *space == group && entry.binding == binding && !entry.visibility.contains(&stage)
            }) {
                return Err(Error::Shader(format!(
                    "DX12 binding {group}:{binding} is not visible to {stage:?}"
                )));
            }
            covered += end - start;
        }
        if self.explicit_native_registers && covered != registers.end - registers.start {
            return Err(Error::Shader(format!(
                "DX12 native register range {:?} in space {space}, class {class} is not fully mapped",
                registers
            )));
        }
        Ok(())
    }
}
pub(super) fn collect_native_registers(
    assignments: &[api::NativeBindingRegister],
) -> Result<BTreeMap<(u32, u32), u32>> {
    let mut registers = BTreeMap::new();
    for item in assignments {
        if registers
            .insert((item.group, item.binding), item.register)
            .is_some()
        {
            return Err(Error::InvalidState);
        }
    }
    Ok(registers)
}

pub(super) fn validate_native_register_ranges(
    registers: &[(u32, u32, u32, u8, u32)],
) -> Result<()> {
    let mut ranges: Vec<_> = registers
        .iter()
        .map(|&(space, _, register, class, count)| {
            (
                (space, class),
                u64::from(register),
                u64::from(register) + u64::from(count),
            )
        })
        .collect();
    ranges.sort_unstable();
    if ranges
        .windows(2)
        .any(|pair| pair[0].0 == pair[1].0 && pair[1].1 < pair[0].2)
    {
        return Err(Error::Unsupported("DX12 native register ranges overlap"));
    }
    Ok(())
}
