//! Merges shader reflection into explicit Core pipeline layouts.

use super::*;

impl<B: ResourceBackend + ShaderBackend> Device<B> {
    /// Creates a layout for the selected shader entry points. Non-shader state
    /// such as dynamic offsets and partial binding uses the explicit layout API.
    pub fn infer_pipeline_layout(
        &self,
        stages: &[(ShaderStage, &PipelineStage<B>)],
        runtime_arrays: &[RuntimeBindingCapacity],
        push_constants: &[PushConstantRange],
    ) -> Result<PipelineLayout<B>> {
        use std::collections::BTreeMap;
        if !self.caps.shader_layout_inference {
            return Err(Error::Unsupported("shader layout inference"));
        }
        if stages.is_empty() {
            return Err(Error::InvalidRange);
        }
        let mut capacities = BTreeMap::new();
        for entry in runtime_arrays {
            if entry.max_count == 0 || entry.group >= self.caps.limits.max_bind_groups {
                return Err(Error::InvalidRange);
            }
            if capacities
                .insert((entry.group, entry.binding), entry.max_count)
                .is_some()
            {
                return Err(Error::InvalidState);
            }
        }
        let mut bindings = BTreeMap::<(u32, u32), BindGroupLayoutEntry>::new();
        for &(stage, entry) in stages {
            if entry.module.device != self.id {
                return Err(Error::WrongDevice);
            }
            for (group, binding) in
                B::reflect_stage_layout(&self.native, stage, entry, runtime_arrays)?
            {
                if group >= self.caps.limits.max_bind_groups {
                    return Err(Error::InvalidRange);
                }
                match bindings.entry((group, binding.binding)) {
                    std::collections::btree_map::Entry::Vacant(slot) => {
                        slot.insert(binding);
                    }
                    std::collections::btree_map::Entry::Occupied(mut slot) => {
                        merge_inferred_binding(slot.get_mut(), binding)?
                    }
                }
            }
        }
        for (key, max_count) in capacities {
            if !bindings
                .get(&key)
                .is_some_and(|entry| entry.count == BindingCount::RuntimeSized { max_count })
            {
                return Err(Error::InvalidState);
            }
        }
        let mut groups = Vec::<BindGroupLayoutDesc>::new();
        for ((group, _), entry) in bindings {
            if groups.len() <= group as usize {
                groups.resize(
                    group as usize + 1,
                    BindGroupLayoutDesc {
                        label: None,
                        entries: Vec::new(),
                    },
                );
            }
            groups[group as usize].entries.push(entry);
        }
        let owned = groups
            .iter()
            .map(|group| self.create_bind_group_layout(group))
            .collect::<Result<Vec<_>>>()?;
        let layout = self.create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &owned.iter().collect::<Vec<_>>(),
            push_constants,
        })?;
        // SAFETY: the layout and all selected stage modules belong to this device.
        B::validate_inferred_pipeline_layout(unsafe { layout.native_ref() }, stages)?;
        Ok(layout)
    }
}

fn merge_inferred_binding(
    current: &mut BindGroupLayoutEntry,
    incoming: BindGroupLayoutEntry,
) -> Result<()> {
    if current.count != incoming.count {
        return Err(Error::Shader("conflicting inferred binding counts".into()));
    }
    match (&mut current.ty, incoming.ty) {
        (current, incoming) if *current == incoming => {}
        (
            BindingType::UniformBuffer { min_size, .. },
            BindingType::UniformBuffer {
                min_size: other, ..
            },
        ) => {
            *min_size = (*min_size).max(other);
        }
        (
            BindingType::StorageBuffer {
                min_size, access, ..
            },
            BindingType::StorageBuffer {
                min_size: other,
                access: other_access,
                ..
            },
        ) if *access == other_access => {
            *min_size = (*min_size).max(other);
        }
        (
            BindingType::Sampler {
                filtering,
                comparison,
            },
            BindingType::Sampler {
                filtering: other,
                comparison: other_compare,
            },
        ) if *comparison == other_compare => {
            *filtering |= other;
        }
        (
            BindingType::SampledTexture {
                sample,
                dimension,
                multisampled,
            },
            BindingType::SampledTexture {
                sample: other,
                dimension: other_dimension,
                multisampled: other_samples,
            },
        ) if *dimension == other_dimension
            && *multisampled == other_samples
            && matches!(
                sample,
                TextureSampleType::Float | TextureSampleType::UnfilterableFloat
            )
            && matches!(
                other,
                TextureSampleType::Float | TextureSampleType::UnfilterableFloat
            ) =>
        {
            *sample = TextureSampleType::Float;
        }
        _ => return Err(Error::Shader("conflicting inferred binding types".into())),
    }
    current.nonuniform_indexing |= incoming.nonuniform_indexing;
    for stage in incoming.visibility {
        if !current.visibility.contains(&stage) {
            current.visibility.push(stage);
        }
    }
    Ok(())
}
