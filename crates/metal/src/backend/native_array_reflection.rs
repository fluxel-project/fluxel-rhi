//! Validate the argument-buffer wrapper ABI used by native Metal functions.
//!
//! Reflection describes the resource inside each wrapper but does not describe
//! the number of wrappers reachable through a pointer. The shader author must
//! therefore uphold the binding layout's declared array capacity. This module
//! checks only the shape required by `argument::materialize`; it does not
//! infer stage usage, texture compatibility, access rights, or buffer sizes.

use fluxel_rhi_core::api::{Error, Result};

#[cfg(target_vendor = "apple")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NativeArrayElement {
    Buffer,
    Texture,
    Sampler,
    AccelerationStructure,
}

/// Verify that a reflected argument is a pointer to a wrapper holding exactly
/// one resource at argument ID zero, as required by the array encoder.
///
/// Obtain `argument` from `newArgumentEncoderWithBufferIndex:reflection:` for
/// the actual function and slot. Pipeline reflection must first establish that
/// the function uses that slot as an argument buffer: invoking that selector
/// for an arbitrary layout slot is not guaranteed safe by Metal. The caller
/// additionally validates texture shape/access and buffer compatibility, and
/// retains the shader's declared array capacity in the pipeline ABI.
#[cfg(target_vendor = "apple")]
#[allow(deprecated)]
pub(super) fn validate_wrapper(
    argument: &objc2_metal::MTLArgument,
    expected: NativeArrayElement,
) -> Result<()> {
    use objc2_metal::{MTLArgumentType, MTLDataType};
    let mismatch = || {
        Error::Shader("Metal native resource array ABI does not match the pipeline layout".into())
    };
    if argument.r#type() != MTLArgumentType::Buffer {
        return Err(mismatch());
    }
    let pointer = argument.bufferPointerType().ok_or_else(mismatch)?;
    if pointer.elementType() != MTLDataType::Struct || !pointer.elementIsArgumentBuffer() {
        return Err(mismatch());
    }
    let structure = pointer.elementStructType().ok_or_else(mismatch)?;
    let members = structure.members();
    if members.len() != 1 {
        return Err(mismatch());
    }
    let member = members.objectAtIndex(0);
    let ty = member.dataType();
    let matches = match expected {
        NativeArrayElement::Buffer => ty == MTLDataType::Pointer,
        NativeArrayElement::Texture => ty == MTLDataType::Texture,
        NativeArrayElement::Sampler => ty == MTLDataType::Sampler,
        NativeArrayElement::AccelerationStructure => {
            ty == MTLDataType::InstanceAccelerationStructure
        }
    };
    if member.argumentIndex() != 0 || !matches {
        return Err(mismatch());
    }
    Ok(())
}

/// Inspect an actual function slot. A missing argument encoder means the slot
/// is unused by this function. Array capacity remains a shader ABI promise.
#[cfg(target_vendor = "apple")]
pub(super) fn inspect_slot(
    function: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLFunction>,
    slot: usize,
    expected: NativeArrayElement,
) -> Result<bool> {
    use objc2::runtime::NSObjectProtocol as _;
    if !function.respondsToSelector(objc2::sel!(newArgumentEncoderWithBufferIndex:reflection:)) {
        return Err(Error::Unsupported("Metal argument buffer reflection"));
    }
    let mut reflection: *mut objc2_metal::MTLArgument = std::ptr::null_mut();
    // SAFETY: the selector is checked, the output pointer is valid for this
    // synchronous call, and Metal returns nil for a slot without an argument buffer.
    let encoder: Option<
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLArgumentEncoder>>,
    > = unsafe {
        objc2::msg_send![function, newArgumentEncoderWithBufferIndex: slot, reflection: &mut reflection]
    };
    if encoder.is_none() {
        return Ok(false);
    }
    // SAFETY: Metal fills this autoreleased pointer during the call, and it
    // remains live for the current autorelease scope.
    let reflection = unsafe { reflection.as_ref() }
        .ok_or_else(|| Error::Shader("Metal argument buffer reflection missing".into()))?;
    validate_wrapper(reflection, expected)?;
    Ok(true)
}

/// Extract shader-required offset alignment for active buffer bindings. Slots
/// listed in `array_slots` are the wrapper ABI validated by `validate_wrapper`;
/// their relevant alignment belongs to the wrapper's pointed-to resource,
/// rather than the argument-buffer allocation itself. Reflection must have
/// been requested with `MTLPipelineOptionBufferTypeInfo`.
#[cfg(target_vendor = "apple")]
pub(super) fn buffer_alignments(
    bindings: &objc2_foundation::NSArray<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBinding>,
    >,
    array_slots: &[usize],
) -> Result<Vec<(usize, u64)>> {
    use objc2_metal::{MTLBinding as _, MTLBindingType, MTLDataType};
    let mut alignments = Vec::new();
    for i in 0..bindings.len() {
        let binding = bindings.objectAtIndex(i);
        if binding.r#type() != MTLBindingType::Buffer || !binding.isUsed() {
            continue;
        }
        let slot = binding.index();
        // SAFETY: a reflected binding of Buffer type implements the buffer
        // binding protocol; both selectors return values owned by reflection.
        let alignment = unsafe {
            if array_slots.contains(&slot) {
                let pointer: Option<objc2::rc::Retained<objc2_metal::MTLPointerType>> =
                    objc2::msg_send![&*binding, bufferPointerType];
                let pointer = pointer.ok_or_else(|| {
                    Error::Shader("Metal array pointer reflection missing".into())
                })?;
                let structure = pointer.elementStructType().ok_or_else(|| {
                    Error::Shader("Metal array wrapper reflection missing".into())
                })?;
                let members = structure.members();
                if members.len() != 1 {
                    return Err(Error::Shader(
                        "Metal array wrapper has unexpected members".into(),
                    ));
                }
                let member = members.objectAtIndex(0);
                if member.argumentIndex() != 0 || member.dataType() != MTLDataType::Pointer {
                    return Err(Error::Shader(
                        "Metal array wrapper is not a buffer pointer".into(),
                    ));
                }
                member
                    .pointerType()
                    .ok_or_else(|| {
                        Error::Shader("Metal array member pointer reflection missing".into())
                    })?
                    .alignment()
            } else {
                let result: usize = objc2::msg_send![&*binding, bufferAlignment];
                result
            }
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::Shader(
                "Metal returned invalid buffer alignment".into(),
            ));
        }
        alignments.push((slot, alignment as u64));
    }
    Ok(alignments)
}

/// Compatibility path for systems exposing the legacy pipeline argument API.
/// Request both argument and buffer-type reflection at pipeline creation.
#[cfg(target_vendor = "apple")]
#[allow(deprecated)]
pub(super) fn legacy_buffer_alignments(
    arguments: &objc2_foundation::NSArray<objc2_metal::MTLArgument>,
    array_slots: &[usize],
) -> Result<Vec<(usize, u64)>> {
    use objc2_metal::{MTLArgumentType, MTLDataType};
    let mut alignments = Vec::new();
    for i in 0..arguments.len() {
        let argument = arguments.objectAtIndex(i);
        if argument.r#type() != MTLArgumentType::Buffer || !argument.isActive() {
            continue;
        }
        let slot = argument.index();
        let alignment = if array_slots.contains(&slot) {
            let pointer = argument
                .bufferPointerType()
                .ok_or_else(|| Error::Shader("Metal array pointer reflection missing".into()))?;
            let structure = pointer
                .elementStructType()
                .ok_or_else(|| Error::Shader("Metal array wrapper reflection missing".into()))?;
            let members = structure.members();
            if members.len() != 1 {
                return Err(Error::Shader(
                    "Metal array wrapper has unexpected members".into(),
                ));
            }
            let member = members.objectAtIndex(0);
            if member.argumentIndex() != 0 || member.dataType() != MTLDataType::Pointer {
                return Err(Error::Shader(
                    "Metal array wrapper is not a buffer pointer".into(),
                ));
            }
            member
                .pointerType()
                .ok_or_else(|| {
                    Error::Shader("Metal array member pointer reflection missing".into())
                })?
                .alignment()
        } else {
            argument.bufferAlignment()
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::Shader(
                "Metal returned invalid buffer alignment".into(),
            ));
        }
        alignments.push((slot, alignment as u64));
    }
    Ok(alignments)
}

/// Find used native argument-buffer slots before querying a function's argument
/// encoder. The type metadata requires pipeline BufferTypeInfo reflection.
#[cfg(target_vendor = "apple")]
pub(super) fn argument_buffer_slots(
    bindings: &objc2_foundation::NSArray<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBinding>,
    >,
) -> Vec<usize> {
    use objc2_metal::{MTLBinding as _, MTLBindingType, MTLDataType};
    let mut slots = Vec::new();
    for i in 0..bindings.len() {
        let binding = bindings.objectAtIndex(i);
        if binding.r#type() != MTLBindingType::Buffer || !binding.isUsed() {
            continue;
        }
        // SAFETY: Metal guarantees buffer bindings implement MTLBufferBinding;
        // reflection retains the returned metadata for this synchronous call.
        let pointer: Option<objc2::rc::Retained<objc2_metal::MTLPointerType>> =
            unsafe { objc2::msg_send![&*binding, bufferPointerType] };
        if pointer
            .as_ref()
            .is_some_and(|p| p.elementType() == MTLDataType::Struct && p.elementIsArgumentBuffer())
        {
            slots.push(binding.index());
        }
    }
    slots
}

/// Active bindings in Metal's buffer namespace, including ordinary buffers.
#[cfg(target_vendor = "apple")]
pub(super) fn active_buffer_slots(
    bindings: &objc2_foundation::NSArray<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBinding>,
    >,
) -> Vec<usize> {
    use objc2_metal::{MTLBinding as _, MTLBindingType};
    (0..bindings.len())
        .filter_map(|i| {
            let binding = bindings.objectAtIndex(i);
            (binding.isUsed() && binding.r#type() == MTLBindingType::Buffer)
                .then(|| binding.index())
        })
        .collect()
}

#[cfg(target_vendor = "apple")]
#[allow(deprecated)]
pub(super) fn legacy_active_buffer_slots(
    arguments: &objc2_foundation::NSArray<objc2_metal::MTLArgument>,
) -> Vec<usize> {
    use objc2_metal::MTLArgumentType;
    (0..arguments.len())
        .filter_map(|i| {
            let argument = arguments.objectAtIndex(i);
            (argument.isActive() && argument.r#type() == MTLArgumentType::Buffer)
                .then(|| argument.index())
        })
        .collect()
}

/// Legacy pipeline reflection equivalent of `argument_buffer_slots`.
#[cfg(target_vendor = "apple")]
#[allow(deprecated)]
pub(super) fn legacy_argument_buffer_slots(
    arguments: &objc2_foundation::NSArray<objc2_metal::MTLArgument>,
) -> Vec<usize> {
    use objc2_metal::{MTLArgumentType, MTLDataType};
    let mut slots = Vec::new();
    for i in 0..arguments.len() {
        let argument = arguments.objectAtIndex(i);
        if argument.r#type() != MTLArgumentType::Buffer || !argument.isActive() {
            continue;
        }
        if argument
            .bufferPointerType()
            .as_ref()
            .is_some_and(|p| p.elementType() == MTLDataType::Struct && p.elementIsArgumentBuffer())
        {
            slots.push(argument.index());
        }
    }
    slots
}
