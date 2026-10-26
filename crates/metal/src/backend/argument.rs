//! Materialize Naga binding arrays as Metal argument-buffer arrays.
//!
//! Naga emits `constant Wrapper<T>*`, with one resource in each wrapper.
//! Consequently each logical array element occupies one encoded argument
//! buffer, rather than a distinct ID within a single argument buffer. Every
//! materialization owns fresh shared storage so later bindings cannot modify
//! data already consumed by the GPU.
use fluxel_rhi_core::api::{Error, Result};

/// Check the exact byte extent before allocating or encoding any element.
fn encoded_array_len(
    encoded: usize,
    alignment: usize,
    count: usize,
    maximum: u64,
) -> Result<usize> {
    if encoded == 0
        || alignment == 0
        || !alignment.is_power_of_two()
        || encoded % alignment != 0
        || count == 0
    {
        return Err(Error::InvalidRange);
    }
    let bytes = encoded.checked_mul(count).ok_or(Error::InvalidRange)?;
    if bytes as u64 > maximum || bytes > isize::MAX as usize {
        return Err(Error::InvalidRange);
    }
    Ok(bytes)
}

#[cfg(target_vendor = "apple")]
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>;
#[cfg(target_vendor = "apple")]
type Texture = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>;
#[cfg(target_vendor = "apple")]
type Sampler =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSamplerState>>;
#[cfg(target_vendor = "apple")]
type Acceleration =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLAccelerationStructure>>;

/// Each array is homogeneous; `None` is an explicitly unbound element.
#[cfg(target_vendor = "apple")]
#[derive(Clone)]
pub(super) enum ArgumentElement {
    Buffer { raw: Option<Buffer>, offset: usize },
    Texture(Option<Texture>),
    Sampler(Option<Sampler>),
    Acceleration(Option<Acceleration>),
}
#[cfg(target_vendor = "apple")]
impl ArgumentElement {
    fn kind(&self) -> u8 {
        match self {
            Self::Buffer { .. } => 0,
            Self::Texture(_) => 1,
            Self::Sampler(_) => 2,
            Self::Acceleration(_) => 3,
        }
    }
    /// Native resources must be declared to the command encoder with their
    /// actual read/write usage before a shader accesses this argument buffer.
    pub(super) fn resource(
        &self,
    ) -> Option<&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLResource>> {
        use objc2::runtime::ProtocolObject;
        use objc2_metal::MTLResource;
        match self {
            Self::Buffer { raw: Some(raw), .. } => {
                Some(ProtocolObject::<dyn MTLResource>::from_ref(&**raw))
            }
            Self::Texture(Some(raw)) => Some(ProtocolObject::<dyn MTLResource>::from_ref(&**raw)),
            Self::Acceleration(Some(raw)) => {
                Some(ProtocolObject::<dyn MTLResource>::from_ref(&**raw))
            }
            _ => None,
        }
    }
}

/// Keep both encoded storage and its contents alive through GPU completion.
#[cfg(target_vendor = "apple")]
pub(super) struct ArgumentArray {
    pub(super) buffer: Buffer,
    pub(super) elements: Vec<ArgumentElement>,
}

/// Encodes one resource per wrapper and retains all objects in the result.
///
/// # Safety
/// `slot` must identify an argument buffer actually present in `function`.
/// Every element must match its reflected resource type, and `elements.len()`
/// must equal the array capacity emitted for that shader. The caller must
/// declare every returned resource with `useResource` and retain the result
/// until all GPU work reading it completes.
#[cfg(target_vendor = "apple")]
pub(super) unsafe fn materialize(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
    function: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLFunction>,
    slot: usize,
    elements: Vec<ArgumentElement>,
) -> Result<ArgumentArray> {
    use objc2::runtime::NSObjectProtocol as _;
    use objc2_metal::{
        MTLArgumentEncoder as _, MTLBuffer as _, MTLDevice as _, MTLResourceOptions,
    };
    if slot >= 31
        || elements.is_empty()
        || elements
            .iter()
            .any(|entry| entry.kind() != elements[0].kind())
    {
        return Err(Error::InvalidRange);
    }
    if !function.respondsToSelector(objc2::sel!(newArgumentEncoderWithBufferIndex:)) {
        return Err(Error::Unsupported("Metal argument buffers on this OS"));
    }
    for element in &elements {
        if let ArgumentElement::Buffer { raw, offset } = element {
            if raw.as_ref().is_some_and(|raw| *offset >= raw.length())
                || (raw.is_none() && *offset != 0)
            {
                return Err(Error::InvalidRange);
            }
        }
    }
    // SAFETY: this selector is available and accepts an NSUInteger. The
    // generated binding assumes a nonnull result, but Metal can return nil
    // for a slot that does not denote an argument buffer.
    let encoder: Option<
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLArgumentEncoder>>,
    > = unsafe { objc2::msg_send![function, newArgumentEncoderWithBufferIndex: slot] };
    let encoder = encoder.ok_or(Error::InvalidRange)?;
    if !encoder.respondsToSelector(objc2::sel!(setArgumentBuffer:startOffset:arrayElement:)) {
        return Err(Error::Unsupported(
            "Metal argument buffer arrays on this OS",
        ));
    }
    if matches!(elements[0], ArgumentElement::Acceleration(_))
        && !encoder.respondsToSelector(objc2::sel!(setAccelerationStructure:atIndex:))
    {
        return Err(Error::Unsupported(
            "Metal acceleration structure arguments on this OS",
        ));
    }
    let length = encoded_array_len(
        encoder.encodedLength(),
        encoder.alignment(),
        elements.len(),
        device.maxBufferLength() as u64,
    )?;
    let buffer = device
        .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
        .ok_or(Error::OutOfMemory)?;
    for (index, element) in elements.iter().enumerate() {
        // SAFETY: buffer has enough aligned space for every reflected wrapper;
        // caller guarantees type and array size, and returned owner retains all
        // objects through submission and the GPU completion point.
        unsafe {
            encoder.setArgumentBuffer_startOffset_arrayElement(Some(&buffer), 0, index);
            match element {
                ArgumentElement::Buffer { raw, offset } => {
                    encoder.setBuffer_offset_atIndex(raw.as_deref(), *offset, 0)
                }
                ArgumentElement::Texture(raw) => encoder.setTexture_atIndex(raw.as_deref(), 0),
                ArgumentElement::Sampler(raw) => encoder.setSamplerState_atIndex(raw.as_deref(), 0),
                ArgumentElement::Acceleration(raw) => {
                    encoder.setAccelerationStructure_atIndex(raw.as_deref(), 0)
                }
            }
        }
    }
    Ok(ArgumentArray { buffer, elements })
}

#[cfg(test)]
#[path = "../tests/backend/argument.rs"]
mod tests;
