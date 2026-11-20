//! Limits for the shared shader-visible descriptor tables on an opened device.

use crate::backend::descriptor::{SAMPLER_CAPACITY, VIEW_CAPACITY};
use fluxel_rhi_core::api::DescriptorTableLimits;
use windows::Win32::Graphics::{
    Direct3D::{D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1},
    Direct3D12::*,
};

pub(super) fn query(
    device: &ID3D12Device,
    binding_tier: D3D12_RESOURCE_BINDING_TIER,
) -> DescriptorTableLimits {
    let tier2 = binding_tier.0 >= D3D12_RESOURCE_BINDING_TIER_2.0;
    let tier3 = binding_tier.0 >= D3D12_RESOURCE_BINDING_TIER_3.0;
    let writable_resources = if tier3 {
        VIEW_CAPACITY
    } else if tier2 || supports_feature_level_11_1(device) {
        64
    } else {
        D3D12_PS_CS_UAV_REGISTER_COUNT
    };
    DescriptorTableLimits {
        uniform_buffers: if tier3 {
            VIEW_CAPACITY
        } else {
            D3D12_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT
        },
        read_only_resources: if tier2 {
            VIEW_CAPACITY
        } else {
            D3D12_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT
        },
        writable_resources,
        samplers: if tier2 {
            SAMPLER_CAPACITY
        } else {
            D3D12_COMMONSHADER_SAMPLER_SLOT_COUNT
        },
        total_resource_descriptors: VIEW_CAPACITY,
    }
}

fn supports_feature_level_11_1(device: &ID3D12Device) -> bool {
    let requested = [D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0];
    let mut levels = D3D12_FEATURE_DATA_FEATURE_LEVELS {
        NumFeatureLevels: requested.len() as u32,
        pFeatureLevelsRequested: requested.as_ptr(),
        MaxSupportedFeatureLevel: D3D_FEATURE_LEVEL_11_0,
    };
    // SAFETY: the feature selector matches the output structure and the
    // input feature levels live through this synchronous call.
    let result = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_FEATURE_LEVELS,
            (&mut levels as *mut D3D12_FEATURE_DATA_FEATURE_LEVELS).cast(),
            std::mem::size_of_val(&levels) as u32,
        )
    };
    result.is_ok() && levels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_11_1
}
