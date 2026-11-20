//! Admits WGSL constants whose layout is preserved by the HLSL root CBV ABI.

use fluxel_rhi_core::api::{self, Error, Result};

use crate::backend::pipeline::{MAX_IMMEDIATE_BYTES, PushConstantsRoot};

pub(super) fn block(
    module: &naga::Module,
) -> Result<Option<(naga::Handle<naga::GlobalVariable>, u32)>> {
    let Some((handle, global)) = module
        .global_variables
        .iter()
        .find(|(_, global)| global.space == naga::AddressSpace::PushConstant)
    else {
        return Ok(None);
    };
    let naga::TypeInner::Struct { ref members, span } = module.types[global.ty].inner else {
        return Err(Error::Unsupported(
            "DX12 WGSL push constant type must be a flat structure",
        ));
    };
    if span == 0 || span % 4 != 0 || span > MAX_IMMEDIATE_BYTES {
        return Err(Error::InvalidRange);
    }
    for member in members {
        let inner = &module.types[member.ty].inner;
        let word = match inner {
            naga::TypeInner::Scalar(scalar) | naga::TypeInner::Vector { scalar, .. } => *scalar,
            _ => return Err(Error::Unsupported("DX12 WGSL push constant member type")),
        };
        if !matches!(
            word,
            naga::Scalar::I32 | naga::Scalar::U32 | naga::Scalar::F32
        ) {
            return Err(Error::Unsupported("DX12 WGSL push constant member scalar"));
        }
        // Naga fills explicit gaps with 32-bit words. Each value must fit in
        // its existing 16-byte cbuffer row or HLSL would move it to the next row.
        if member.offset % 4 != 0 || member.offset % 16 + inner.size(module.to_ctx()) > 16 {
            return Err(Error::Unsupported("DX12 WGSL push constant member layout"));
        }
    }
    Ok(Some((handle, span)))
}

pub(super) fn validate_range(
    root: Option<&PushConstantsRoot>,
    stage: api::ShaderStage,
    bytes: u32,
) -> Result<()> {
    let visibility = match stage {
        api::ShaderStage::Vertex => api::StageMask::VERTEX,
        api::ShaderStage::Fragment => api::StageMask::FRAGMENT,
        api::ShaderStage::Compute => api::StageMask::COMPUTE,
        _ => return Err(Error::Unsupported("DX12 WGSL push constant shader stage")),
    };
    if !root.is_some_and(|root| root.covers(visibility, 0, bytes)) {
        return Err(Error::Shader(
            "WGSL push constant block is outside the declared stage ranges".into(),
        ));
    }
    Ok(())
}
