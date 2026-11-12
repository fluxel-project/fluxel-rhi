//! Scoped native PIX debug marker recording.

use super::*;

pub(super) fn record(
    encoder: &mut Dx12Encoder,
    _: fluxel_rhi_core::backend::DebugMarkerScope,
    op: fluxel_rhi_core::backend::DebugMarkerOp<'_>,
) -> Result<(), Error> {
    use std::ffi::c_void;

    // PIX recognizes UTF-16 event data on a command list. The vector owns
    // the conversion until the synchronous native call returns.
    let label = |text: &str| text.encode_utf16().chain(Some(0)).collect::<Vec<_>>();
    match op {
        fluxel_rhi_core::backend::DebugMarkerOp::Insert(text) => {
            let label = label(text);
            // SAFETY: the command list is open and the temporary UTF-16
            // buffer remains live for SetMarker's synchronous copy.
            unsafe {
                encoder.list.SetMarker(
                    0,
                    Some(label.as_ptr().cast::<c_void>()),
                    (label.len() * std::mem::size_of::<u16>()) as u32,
                );
            }
        }
        fluxel_rhi_core::backend::DebugMarkerOp::Push(text) => {
            let label = label(text);
            // SAFETY: as above for SetMarker.
            unsafe {
                encoder.list.BeginEvent(
                    0,
                    Some(label.as_ptr().cast::<c_void>()),
                    (label.len() * std::mem::size_of::<u16>()) as u32,
                );
            }
            encoder.debug_group_depth = encoder
                .debug_group_depth
                .checked_add(1)
                .ok_or(Error::InvalidState)?;
        }
        fluxel_rhi_core::backend::DebugMarkerOp::Pop => {
            if encoder.debug_group_depth == 0 {
                return Err(Error::InvalidState);
            }
            // SAFETY: the list is open and the native depth was checked.
            unsafe { encoder.list.EndEvent() };
            encoder.debug_group_depth -= 1;
        }
    }
    Ok(())
}
