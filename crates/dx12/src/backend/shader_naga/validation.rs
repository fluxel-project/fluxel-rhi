//! Naga capabilities admitted by the DX12 WGSL route.

use fluxel_rhi_core::api;

pub(super) fn validation_capabilities(desc: &api::ShaderDesc) -> naga::valid::Capabilities {
    let mut caps = naga::valid::Capabilities::SAMPLER_NON_UNIFORM_INDEXING
        | naga::valid::Capabilities::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | naga::valid::Capabilities::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING;
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
