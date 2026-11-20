//! Negotiates the shader models recognized by the installed D3D12 runtime.

use windows::Win32::{Foundation::E_INVALIDARG, Graphics::Direct3D12::*};

pub(super) fn highest(device: &ID3D12Device) -> u16 {
    for minor in (0..=9).rev() {
        let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL(0x60 + minor),
        };
        // SAFETY: the feature selector matches the initialized output
        // structure, which is borrowed only for this synchronous query.
        let result = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&mut model as *mut D3D12_FEATURE_DATA_SHADER_MODEL).cast(),
                std::mem::size_of_val(&model) as u32,
            )
        };
        match result {
            Ok(()) if model.HighestShaderModel.0 >= 0x60 => {
                return 60 + (model.HighestShaderModel.0 - 0x60) as u16;
            }
            // Old runtimes reject individual shader model enumerants before
            // considering the hardware, so retry with a known lower ceiling.
            Err(error) if error.code() == E_INVALIDARG => {}
            _ => return 0,
        }
    }
    0
}
