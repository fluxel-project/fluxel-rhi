//! Naga capabilities admitted by the DX12 WGSL route.

use fluxel_rhi_core::api::{self, Error, Result};

pub(in crate::backend) fn requires_f16(desc: &api::ShaderDesc) -> bool {
    desc.required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_F16)
}

pub(super) fn requires_view_index(desc: &api::ShaderDesc) -> bool {
    desc.required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_VIEW_INDEX)
}

pub(super) fn uses_entry_builtin(
    module: &naga::Module,
    entry: &naga::EntryPoint,
    builtins: &[naga::BuiltIn],
) -> bool {
    fn uses(
        module: &naga::Module,
        ty: naga::Handle<naga::Type>,
        binding: Option<&naga::Binding>,
        builtins: &[naga::BuiltIn],
    ) -> bool {
        if matches!(binding, Some(naga::Binding::BuiltIn(builtin)) if builtins.contains(builtin)) {
            return true;
        }
        matches!(&module.types[ty].inner, naga::TypeInner::Struct { members, .. }
            if members.iter().any(|member| uses(module, member.ty, member.binding.as_ref(), builtins)))
    }
    entry
        .function
        .arguments
        .iter()
        .any(|arg| uses(module, arg.ty, arg.binding.as_ref(), builtins))
}

pub(super) fn validate_f16_bindings(module: &naga::Module, standard_immediate: bool) -> Result<()> {
    if module.global_variables.iter().any(|(_, global)| {
        (global.space == naga::AddressSpace::Uniform
            || (!standard_immediate && global.space == naga::AddressSpace::PushConstant))
            && contains_f16(&module.types, global.ty)
    }) {
        return Err(Error::Unsupported(
            "DX12 WGSL f16 uniform and immediate layout",
        ));
    }
    Ok(())
}

fn contains_f16(types: &naga::UniqueArena<naga::Type>, ty: naga::Handle<naga::Type>) -> bool {
    match &types[ty].inner {
        naga::TypeInner::Scalar(scalar)
        | naga::TypeInner::Vector { scalar, .. }
        | naga::TypeInner::Matrix { scalar, .. } => *scalar == naga::Scalar::F16,
        naga::TypeInner::Array { base, .. } | naga::TypeInner::BindingArray { base, .. } => {
            contains_f16(types, *base)
        }
        naga::TypeInner::Struct { members, .. } => {
            members.iter().any(|member| contains_f16(types, member.ty))
        }
        _ => false,
    }
}

pub(super) fn uses_packed_dot(module: &naga::Module) -> bool {
    let uses_dot = |arena: &naga::Arena<naga::Expression>| {
        arena.iter().any(|(_, expr)| {
            matches!(
                expr,
                naga::Expression::Math {
                    fun: naga::MathFunction::Dot4I8Packed | naga::MathFunction::Dot4U8Packed,
                    ..
                }
            )
        })
    };
    uses_dot(&module.global_expressions)
        || module
            .functions
            .iter()
            .any(|(_, function)| uses_dot(&function.expressions))
        || module
            .entry_points
            .iter()
            .any(|entry| uses_dot(&entry.function.expressions))
}

pub(super) fn validation_capabilities(desc: &api::ShaderDesc) -> naga::valid::Capabilities {
    let mut caps = naga::valid::Capabilities::MULTISAMPLED_SHADING
        | naga::valid::Capabilities::UNIFORM_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | naga::valid::Capabilities::SAMPLER_NON_UNIFORM_INDEXING
        | naga::valid::Capabilities::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | naga::valid::Capabilities::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING;
    if desc.required_features.iter().any(|feature| {
        matches!(
            feature.as_str(),
            api::SHADER_FEATURE_PUSH_CONSTANTS | api::SHADER_FEATURE_IMMEDIATE_DATA
        )
    }) {
        caps |= naga::valid::Capabilities::PUSH_CONSTANT;
    }
    if requires_f16(desc) {
        caps |= naga::valid::Capabilities::SHADER_FLOAT16;
        if super::immediates::required(desc) {
            caps |= naga::valid::Capabilities::FLOAT16_IMMEDIATE_DATA;
        }
    }
    if requires_view_index(desc) {
        caps |= naga::valid::Capabilities::MULTIVIEW;
    }
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
    if desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_PRIMITIVE_INDEX)
    {
        caps |= naga::valid::Capabilities::PRIMITIVE_INDEX;
    }
    if desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_EARLY_DEPTH_TEST)
    {
        caps |= naga::valid::Capabilities::EARLY_DEPTH_TEST;
    }
    if desc
        .required_features
        .iter()
        .any(|feature| feature == super::super::SUBGROUP_FEATURE)
    {
        caps |= naga::valid::Capabilities::SUBGROUP;
    }
    if desc.required_features.iter().any(|x| {
        x == api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_ADD
            || x == api::SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_ADD
            || x == api::SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MIN
            || x == api::SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MAX
            || x == api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MIN
            || x == api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MAX
            || x == api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_XOR
            || x == api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_AND
            || x == api::SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_OR
    }) {
        caps |= naga::valid::Capabilities::TEXTURE_ATOMIC;
    }
    caps
}

pub(super) fn validate_entry(module: &naga::Module, entry: &naga::EntryPoint) -> Result<()> {
    if matches!(
        entry.early_depth_test,
        Some(naga::EarlyDepthTest::Allow { .. })
    ) {
        return Err(Error::Unsupported("DX12 WGSL conservative fragment depth"));
    }
    // A compute wave need not contain consecutive local invocation indices.
    // Naga's formulas based on a contiguous partition cannot identify waves.
    if entry
        .function
        .arguments
        .iter()
        .any(|arg| uses_workgroup_subgroup_identity(module, arg.ty, arg.binding.as_ref()))
    {
        return Err(Error::Unsupported("DX12 WGSL subgroup ID and count"));
    }
    Ok(())
}

fn uses_workgroup_subgroup_identity(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    binding: Option<&naga::Binding>,
) -> bool {
    if matches!(
        binding,
        Some(naga::Binding::BuiltIn(
            naga::BuiltIn::SubgroupId | naga::BuiltIn::NumSubgroups
        ))
    ) {
        return true;
    }
    if let naga::TypeInner::Struct { members, .. } = &module.types[ty].inner {
        return members.iter().any(|member| {
            uses_workgroup_subgroup_identity(module, member.ty, member.binding.as_ref())
        });
    }
    false
}

pub(super) fn texture_atomic_only(
    module: &naga::Module,
    expected: naga::AtomicFunction,
    expected_format: naga::StorageFormat,
) -> bool {
    module.functions.iter().all(|(_, function)| {
        atomic_only_block(
            module,
            &function.expressions,
            &function.arguments,
            &function.body,
            expected,
            expected_format,
        )
    }) && module.entry_points.iter().all(|entry| {
        atomic_only_block(
            module,
            &entry.function.expressions,
            &entry.function.arguments,
            &entry.function.body,
            expected,
            expected_format,
        )
    })
}

fn atomic_only_block(
    module: &naga::Module,
    expressions: &naga::Arena<naga::Expression>,
    arguments: &[naga::FunctionArgument],
    block: &naga::Block,
    expected: naga::AtomicFunction,
    expected_format: naga::StorageFormat,
) -> bool {
    block.iter().all(|statement| match statement {
        naga::Statement::ImageAtomic { image, fun, .. } => {
            *fun == expected
                && image_storage_format(module, expressions, arguments, *image)
                    == Some(expected_format)
        }
        // Buffer and workgroup atomics require a different public route.
        naga::Statement::Atomic { .. } => false,
        naga::Statement::Block(block) => atomic_only_block(
            module,
            expressions,
            arguments,
            block,
            expected,
            expected_format,
        ),
        naga::Statement::If { accept, reject, .. } => {
            atomic_only_block(
                module,
                expressions,
                arguments,
                accept,
                expected,
                expected_format,
            ) && atomic_only_block(
                module,
                expressions,
                arguments,
                reject,
                expected,
                expected_format,
            )
        }
        naga::Statement::Switch { cases, .. } => cases.iter().all(|case| {
            atomic_only_block(
                module,
                expressions,
                arguments,
                &case.body,
                expected,
                expected_format,
            )
        }),
        naga::Statement::Loop {
            body, continuing, ..
        } => {
            atomic_only_block(
                module,
                expressions,
                arguments,
                body,
                expected,
                expected_format,
            ) && atomic_only_block(
                module,
                expressions,
                arguments,
                continuing,
                expected,
                expected_format,
            )
        }
        _ => true,
    })
}

fn image_storage_format(
    module: &naga::Module,
    expressions: &naga::Arena<naga::Expression>,
    arguments: &[naga::FunctionArgument],
    image: naga::Handle<naga::Expression>,
) -> Option<naga::StorageFormat> {
    image_storage_type(module, expressions, arguments, image)
        .and_then(|ty| storage_format(module, ty))
}

fn image_storage_type(
    module: &naga::Module,
    expressions: &naga::Arena<naga::Expression>,
    arguments: &[naga::FunctionArgument],
    image: naga::Handle<naga::Expression>,
) -> Option<naga::Handle<naga::Type>> {
    use naga::Expression as E;
    let ty = match &expressions[image] {
        E::GlobalVariable(global) => module.global_variables[*global].ty,
        E::FunctionArgument(index) => arguments.get(*index as usize)?.ty,
        E::Access { base, .. } | E::AccessIndex { base, .. } => {
            let ty = image_storage_type(module, expressions, arguments, *base)?;
            return Some(match &module.types[ty].inner {
                naga::TypeInner::BindingArray { base, .. } => *base,
                _ => ty,
            });
        }
        _ => return None,
    };
    match &module.types[ty].inner {
        naga::TypeInner::BindingArray { base, .. } => Some(*base),
        _ => Some(ty),
    }
}

fn storage_format(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
) -> Option<naga::StorageFormat> {
    match module.types[ty].inner {
        naga::TypeInner::Image {
            class: naga::ImageClass::Storage { format, .. },
            ..
        } => Some(format),
        _ => None,
    }
}

#[cfg(test)]
#[path = "../../tests/shader_naga/validation.rs"]
mod tests;
