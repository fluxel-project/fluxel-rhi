//! Validates the descriptor layout and its register ABI.

use super::*;

/// Validates descriptor ABI and the DXIL register ranges for a group.
pub(crate) fn create_bind_group_layout(
    _: &Dx12Device,
    desc: &api::BindGroupLayoutDesc,
) -> Result<Dx12BindGroupLayout> {
    let mut slots = Vec::with_capacity(desc.entries.len());
    let mut previous = None;
    for entry in &desc.entries {
        if previous.is_some_and(|value| value >= entry.binding) {
            return Err(Error::InvalidState);
        }
        previous = Some(entry.binding);
        let descriptor_count = descriptor_array::count(entry.count)?;
        match entry.ty {
            api::BindingType::UniformBuffer {
                dynamic_offset,
                min_size,
            } if min_size > 0
                && (!dynamic_offset || matches!(entry.count, api::BindingCount::One)) =>
            {
                slots.push(BindingSlot {
                    binding_type: entry.ty.clone(),
                    binding: entry.binding,
                    min_size,
                    class: DescriptorClass::Cbv,
                    dynamic_offset,
                    descriptor_count,
                    runtime_sized: matches!(entry.count, api::BindingCount::RuntimeSized { .. }),
                    view_dimension: None,
                    texture_sample: None,
                    texture_format: None,
                    multisampled: false,
                });
            }
            api::BindingType::StorageBuffer {
                dynamic_offset,
                min_size,
                access,
            } if min_size > 0
                && (!dynamic_offset || matches!(entry.count, api::BindingCount::One)) =>
            {
                slots.push(BindingSlot {
                    binding_type: entry.ty.clone(),
                    binding: entry.binding,
                    min_size,
                    class: if matches!(access, api::StorageAccess::Read) {
                        DescriptorClass::Srv
                    } else {
                        DescriptorClass::Uav
                    },
                    dynamic_offset,
                    descriptor_count,
                    runtime_sized: matches!(entry.count, api::BindingCount::RuntimeSized { .. }),
                    view_dimension: None,
                    texture_sample: None,
                    texture_format: None,
                    multisampled: false,
                });
            }
            api::BindingType::SampledTexture {
                sample,
                dimension,
                multisampled,
            } => {
                slots.push(BindingSlot {
                    binding_type: entry.ty.clone(),
                    binding: entry.binding,
                    min_size: 0,
                    class: DescriptorClass::Srv,
                    dynamic_offset: false,
                    descriptor_count,
                    runtime_sized: matches!(entry.count, api::BindingCount::RuntimeSized { .. }),
                    view_dimension: Some(dimension),
                    texture_sample: Some(sample),
                    texture_format: None,
                    multisampled,
                });
            }
            api::BindingType::StorageTexture {
                format,
                dimension,
                access: _,
            } => {
                if !matches!(
                    dimension,
                    api::TextureViewDimension::D1
                        | api::TextureViewDimension::D2
                        | api::TextureViewDimension::D2Array
                        | api::TextureViewDimension::D3
                ) {
                    return Err(Error::Unsupported("DX12 storage texture view dimension"));
                }
                slots.push(BindingSlot {
                    binding_type: entry.ty.clone(),
                    binding: entry.binding,
                    min_size: 0,
                    class: DescriptorClass::Uav,
                    dynamic_offset: false,
                    descriptor_count,
                    runtime_sized: matches!(entry.count, api::BindingCount::RuntimeSized { .. }),
                    view_dimension: Some(dimension),
                    texture_sample: None,
                    texture_format: Some(format),
                    multisampled: false,
                });
            }
            api::BindingType::Sampler {
                filtering: _,
                comparison: false,
            } => {
                slots.push(BindingSlot {
                    binding_type: entry.ty.clone(),
                    binding: entry.binding,
                    min_size: 0,
                    class: DescriptorClass::Sampler,
                    dynamic_offset: false,
                    descriptor_count,
                    runtime_sized: matches!(entry.count, api::BindingCount::RuntimeSized { .. }),
                    view_dimension: None,
                    texture_sample: None,
                    texture_format: None,
                    multisampled: false,
                });
            }
            api::BindingType::AccelerationStructure => {
                slots.push(BindingSlot {
                    binding_type: entry.ty.clone(),
                    binding: entry.binding,
                    min_size: 0,
                    class: DescriptorClass::Srv,
                    dynamic_offset: false,
                    descriptor_count,
                    runtime_sized: matches!(entry.count, api::BindingCount::RuntimeSized { .. }),
                    view_dimension: None,
                    texture_sample: None,
                    texture_format: None,
                    multisampled: false,
                });
            }
            _ => return Err(Error::Unsupported("DX12 bind-group binding type")),
        }
    }
    Ok(Dx12BindGroupLayout(Arc::new(BindGroupLayoutState {
        slots,
    })))
}
