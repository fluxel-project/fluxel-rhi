//! Narrow DXC invocation shared by Naga-generated and direct HLSL sources.
//!
//! Pipeline constants become `-D name=literal` command-line definitions. The
//! checked identifier and finite literal forms ensure they cannot alter DXC's
//! command line or inject unrelated source tokens.

use std::collections::BTreeSet;

use fluxel_rhi_core::api::{Error, Result, SpecializationValue};
use windows::Win32::Foundation::{FreeLibrary, HMODULE};
use windows::Win32::Graphics::Direct3D::Dxc::{
    CLSID_DxcCompiler, DXC_CP_UTF8, DXC_OUT_HLSL, DxcBuffer, IDxcBlob, IDxcCompiler3, IDxcResult,
};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::core::{GUID, HRESULT, Interface, PCWSTR};

struct Library(HMODULE);
impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: all COM objects from this module are dropped before this owner.
        let _ = unsafe { FreeLibrary(self.0) };
    }
}

pub(super) fn compile_hlsl(
    source: &str,
    entry: &str,
    profile: &str,
    constants: &[(String, SpecializationValue)],
) -> Result<Vec<u8>> {
    let definitions = definitions(constants)?;
    let mut arguments = vec![
        "-T".to_owned(),
        profile.to_owned(),
        "-E".to_owned(),
        entry.to_owned(),
    ];
    append_definitions(&mut arguments, definitions);
    let (_library, result) = invoke(source, &arguments)?;
    result_bytes(&result)
}

/// Expands conditionals and macro definitions in memory before the raw-HLSL
/// WaveSize lexer selects an entry. DXC emits only `#line` markers, which that
/// lexer deliberately skips; a remaining directive is rejected there.
pub(super) fn preprocess_hlsl(
    source: &str,
    constants: &[(String, SpecializationValue)],
) -> Result<String> {
    let mut arguments = vec!["-P".to_owned()];
    append_definitions(&mut arguments, definitions(constants)?);
    let (_library, result) = invoke(source, &arguments)?;
    let mut name = None;
    let mut output = None::<IDxcBlob>;
    // SAFETY: DXC fills owned COM output storage synchronously; `result` and
    // its library remain alive until the bytes are copied below.
    unsafe { result.GetOutput(DXC_OUT_HLSL, &mut name, &mut output) }
        .map_err(|e| Error::Shader(e.to_string()))?;
    let output = output.ok_or(Error::Unsupported("DXC preprocess output unavailable"))?;
    // SAFETY: DXC owns this blob until `output` is dropped after copying.
    let bytes = unsafe {
        std::slice::from_raw_parts(output.GetBufferPointer().cast(), output.GetBufferSize())
    };
    String::from_utf8(bytes.to_vec())
        .map_err(|_| Error::Shader("DXC preprocess output is not UTF-8".into()))
}

fn append_definitions(arguments: &mut Vec<String>, definitions: Vec<String>) {
    for definition in definitions {
        arguments.push("-D".into());
        arguments.push(definition);
    }
}

fn invoke(source: &str, arguments: &[String]) -> Result<(Library, IDxcResult)> {
    // SAFETY: LoadLibraryW receives a static nul-terminated DLL name.
    let library = Library(
        unsafe { LoadLibraryW(windows::core::w!("dxcompiler.dll")) }
            .map_err(|_| Error::Unsupported("DXC runtime unavailable"))?,
    );
    // SAFETY: the name is static and the module remains loaded through the COM calls.
    let proc = unsafe { GetProcAddress(library.0, windows::core::s!("DxcCreateInstance")) }
        .ok_or(Error::Unsupported("DXC entry point unavailable"))?;
    type Create =
        unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut core::ffi::c_void) -> HRESULT;
    // SAFETY: DxcCreateInstance has the documented signature for this named export.
    let create: Create = unsafe { std::mem::transmute(proc) };
    let mut object = std::ptr::null_mut();
    // SAFETY: object is a valid output pointer and both GUIDs name the requested compiler.
    unsafe { create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut object).ok() }
        .map_err(|e| Error::Shader(e.to_string()))?;
    // SAFETY: successful DxcCreateInstance returned a newly owned IDxcCompiler3.
    let compiler = unsafe { IDxcCompiler3::from_raw(object) };
    let strings: Vec<Vec<u16>> = arguments
        .iter()
        .map(|s| s.encode_utf16().chain(Some(0)).collect())
        .collect();
    let args: Vec<PCWSTR> = strings.iter().map(|s| PCWSTR(s.as_ptr())).collect();
    let buffer = DxcBuffer {
        Ptr: source.as_ptr().cast(),
        Size: source.len(),
        Encoding: DXC_CP_UTF8.0,
    };
    // SAFETY: buffer and argument strings remain live through the synchronous compilation.
    let result: IDxcResult = unsafe { compiler.Compile(&buffer, Some(&args), None) }
        .map_err(|e| Error::Shader(e.to_string()))?;
    // SAFETY: the operation result is live and owns the returned blobs.
    let status = unsafe { result.GetStatus() }.map_err(|e| Error::Shader(e.to_string()))?;
    if status.is_err() {
        // SAFETY: error blob data remains valid while the blob is alive.
        let detail = unsafe { result.GetErrorBuffer() }
            .ok()
            .map(|blob| unsafe {
                String::from_utf8_lossy(std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast(),
                    blob.GetBufferSize(),
                ))
                .into_owned()
            })
            .unwrap_or_else(|| format!("{status:?}"));
        return Err(Error::Shader(detail));
    }
    Ok((library, result))
}

fn result_bytes(result: &IDxcResult) -> Result<Vec<u8>> {
    // SAFETY: successful compilation owns an output blob; copy it before releasing COM.
    let blob = unsafe { result.GetResult() }.map_err(|e| Error::Shader(e.to_string()))?;
    let bytes =
        unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) }
            .to_vec();
    Ok(bytes)
}

fn definitions(constants: &[(String, SpecializationValue)]) -> Result<Vec<String>> {
    let mut names = BTreeSet::new();
    constants
        .iter()
        .map(|(name, value)| {
            if !identifier(name) {
                return Err(Error::Shader(format!(
                    "DX12 direct HLSL pipeline constant {name:?} is not an identifier"
                )));
            }
            if !names.insert(name.as_str()) {
                return Err(Error::Shader(format!(
                    "duplicate DX12 direct HLSL pipeline constant {name}"
                )));
            }
            // Parentheses preserve the constant's expression boundaries when
            // a caller uses it under a unary or binary operator in HLSL.
            Ok(format!("{name}=({})", literal(*value)?))
        })
        .collect()
}

fn identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn literal(value: SpecializationValue) -> Result<String> {
    let unsupported = || Error::Unsupported("DX12 direct HLSL pipeline constant value");
    match value {
        SpecializationValue::Bool(value) => Ok(u32::from(value).to_string()),
        SpecializationValue::U8(value) => Ok(format!("{value}u")),
        SpecializationValue::I8(value) => signed_literal(i64::from(value), ""),
        SpecializationValue::U16(value) => Ok(format!("{value}u")),
        SpecializationValue::I16(value) => signed_literal(i64::from(value), ""),
        SpecializationValue::U32(value) => Ok(format!("{value}u")),
        SpecializationValue::I32(value) => signed_literal(i64::from(value), ""),
        SpecializationValue::U64(value) => Ok(format!("{value}ull")),
        SpecializationValue::I64(value) => signed_literal(value, "ll"),
        SpecializationValue::F16Bits(bits) => {
            let value = f16_to_f32(bits);
            value
                .is_finite()
                .then(|| float_literal(f64::from(value)))
                .ok_or_else(unsupported)?
        }
        SpecializationValue::F32(value) if value.is_finite() => float_literal(f64::from(value)),
        SpecializationValue::F64(value) if value.is_finite() => double_literal(value),
        SpecializationValue::F32(_) | SpecializationValue::F64(_) => Err(unsupported()),
    }
}

fn signed_literal(value: i64, suffix: &str) -> Result<String> {
    if value == i64::MIN {
        // The magnitude is one beyond the largest signed literal. Form the
        // value with two representable operands instead of relying on a
        // frontend's unary-negation overflow behavior.
        Ok(format!("(-9223372036854775807{suffix}-1{suffix})"))
    } else {
        Ok(format!("{value}{suffix}"))
    }
}

fn float_literal(value: f64) -> Result<String> {
    let mut value = value.to_string();
    if !value.contains(['.', 'e', 'E']) {
        value.push_str(".0");
    }
    Ok(value)
}

fn double_literal(value: f64) -> Result<String> {
    let mut value = value.to_string();
    if !value.contains(['.', 'e', 'E']) {
        value.push_str(".0");
    }
    value.push('L');
    Ok(value)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut fraction = fraction;
            let mut exponent = 127 - 14;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((fraction & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        exponent => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
}

#[cfg(test)]
#[path = "../../tests/backend/shader_naga/dxc.rs"]
mod tests;
