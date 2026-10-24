//! Naga capabilities admitted by the DX12 WGSL route.

use fluxel_rhi_core::api;

pub(super) fn validation_capabilities(desc: &api::ShaderDesc) -> naga::valid::Capabilities {
    let mut caps = naga::valid::Capabilities::SAMPLER_NON_UNIFORM_INDEXING
        | naga::valid::Capabilities::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    if desc
        .required_features
        .iter()
        .any(|x| x == api::SHADER_FEATURE_RAY_QUERY)
    {
        caps |= naga::valid::Capabilities::RAY_QUERY;
    }
    if desc
        .required_features
        .iter()
        .any(|x| x == "shaderClipDistance")
    {
        caps |= naga::valid::Capabilities::CLIP_DISTANCE;
    }
    if desc.required_features.iter().any(|x| x == "subgroups") {
        caps |= naga::valid::Capabilities::SUBGROUP;
    }
    caps
}
