//! Native register assignment and stage visibility validation.

use fluxel_rhi_core::api::{self, Error, Result};
use std::collections::BTreeMap;

use super::PipelineLayoutState;

impl PipelineLayoutState {
    pub(crate) fn has_stage_restricted_bindings(&self, stage: api::ShaderStage) -> bool {
        self.binding_entries
            .iter()
            .any(|(_, entry)| !entry.visibility.contains(&stage))
    }

    pub(crate) fn validate_register_visibility(
        &self,
        stage: api::ShaderStage,
        space: u32,
        class: u8,
        registers: std::ops::Range<u64>,
    ) -> Result<()> {
        for &(group, binding, register, register_class, count) in &self.register_map {
            let register = u64::from(register);
            if group == space
                && register_class == class
                && register < registers.end
                && registers.start < register + u64::from(count)
                && self.binding_entries.iter().any(|(space, entry)| {
                    *space == group
                        && entry.binding == binding
                        && !entry.visibility.contains(&stage)
                })
            {
                return Err(Error::Shader(format!(
                    "DX12 binding {group}:{binding} is not visible to {stage:?}"
                )));
            }
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
