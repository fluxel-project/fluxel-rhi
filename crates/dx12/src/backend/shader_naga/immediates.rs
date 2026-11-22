//! Admits WGSL constants whose layout is preserved by the HLSL root CBV ABI.

use fluxel_rhi_core::api::{self, Error, Result};

use crate::backend::pipeline::{MAX_IMMEDIATE_BYTES, PushConstantsRoot};

const STANDARD_IMMEDIATE_BYTES: u32 = 64;

pub(super) fn required(desc: &api::ShaderDesc) -> bool {
    desc.required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_IMMEDIATE_DATA)
}

pub(super) fn blocks(
    module: &naga::Module,
    standard: bool,
) -> Result<Vec<(naga::Handle<naga::GlobalVariable>, u32)>> {
    if !standard {
        return Ok(block(module)?
            .into_iter()
            .map(|(handle, bytes)| (handle, u32::MAX >> (32 - bytes / 4)))
            .collect());
    }
    module
        .global_variables
        .iter()
        .filter(|(_, global)| global.space == naga::AddressSpace::PushConstant)
        .map(|(handle, global)| {
            if global.init.is_some() || global.binding.is_some() {
                return Err(Error::Shader(
                    "WGSL immediate data cannot have an initializer or binding".into(),
                ));
            }
            let bytes = module.types[global.ty].inner.size(module.to_ctx());
            if bytes == 0 || bytes > STANDARD_IMMEDIATE_BYTES {
                return Err(Error::InvalidRange);
            }
            Ok((handle, accessible_slots(module, global.ty, 0)?))
        })
        .collect()
}

fn accessible_slots(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    offset: u32,
) -> Result<u32> {
    use naga::{Scalar, TypeInner, VectorSize};
    let words = |offset: u32, count: u32| ((1u32 << count) - 1) << (offset / 4);
    Ok(match module.types[ty].inner {
        TypeInner::Scalar(Scalar::I32 | Scalar::U32 | Scalar::F32) => words(offset, 1),
        TypeInner::Vector {
            size,
            scalar: Scalar::I32 | Scalar::U32 | Scalar::F32,
        } => words(offset, size as u32),
        TypeInner::Matrix {
            columns,
            rows,
            scalar: Scalar::F32,
        } => {
            let stride = if rows == VectorSize::Bi { 8 } else { 16 };
            (0..columns as u32).fold(0, |slots, column| {
                slots | words(offset + column * stride, rows as u32)
            })
        }
        TypeInner::Struct { ref members, .. } => members.iter().try_fold(0, |slots, member| {
            Ok::<_, Error>(slots | accessible_slots(module, member.ty, offset + member.offset)?)
        })?,
        TypeInner::Array { .. } => {
            return Err(Error::Shader(
                "WGSL immediate store types cannot contain arrays".into(),
            ));
        }
        _ => return Err(Error::Unsupported("DX12 WGSL immediate store type")),
    })
}

fn block(module: &naga::Module) -> Result<Option<(naga::Handle<naga::GlobalVariable>, u32)>> {
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
    slots: u32,
) -> Result<()> {
    let visibility = match stage {
        api::ShaderStage::Vertex => api::StageMask::VERTEX,
        api::ShaderStage::Fragment => api::StageMask::FRAGMENT,
        api::ShaderStage::Compute => api::StageMask::COMPUTE,
        _ => return Err(Error::Unsupported("DX12 WGSL push constant shader stage")),
    };
    if !root.is_some_and(|root| {
        (0..32)
            .filter(|word| slots & (1 << word) != 0)
            .all(|word| root.covers(visibility, word * 4, word * 4 + 4))
    }) {
        return Err(Error::Shader(
            "WGSL push constant block is outside the declared stage ranges".into(),
        ));
    }
    Ok(())
}
