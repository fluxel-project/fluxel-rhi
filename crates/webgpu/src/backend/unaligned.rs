//! Preserve bytes around unaligned buffer writes using a native GPU word patch.

use super::super::{js, resource::WebGpuBuffer};
use super::{WebGpuEncoder, error, field, number, object};
use fluxel_rhi_core::api::{Error, Result};
use js_sys::Array;
use wasm_bindgen::JsValue;

pub(super) fn patch(
    encoder: &mut WebGpuEncoder,
    dst: &WebGpuBuffer,
    word_offset: u64,
    first: usize,
    bytes: &[u8],
) -> Result<()> {
    if bytes.is_empty() || first + bytes.len() > 4 {
        return Err(Error::InvalidRange);
    }
    let mut mask = 0u32;
    let mut value = 0u32;
    for (i, byte) in bytes.iter().enumerate() {
        let shift = ((first + i) * 8) as u32;
        mask |= 0xffu32 << shift;
        value |= u32::from(*byte) << shift;
    }
    let device = &encoder.device.0.device;
    let buffer_desc = object();
    field(&buffer_desc, "size", number(4u32))?;
    field(&buffer_desc, "usage", number(4u32 | 8 | 128))?;
    let temporary = js::call1(device, "createBuffer", &buffer_desc).map_err(error)?;
    js::call(
        &encoder.raw,
        "copyBufferToBuffer",
        &[
            dst.raw.clone(),
            number(word_offset as f64),
            temporary.clone(),
            number(0u32),
            number(4u32),
        ],
    )
    .map_err(error)?;
    let module_desc = object();
    let code = format!(
        "@group(0) @binding(0) var<storage, read_write> word: array<u32>; @compute @workgroup_size(1) fn main() {{ word[0] = (word[0] & ~{mask}u) | {value}u; }}"
    );
    field(&module_desc, "code", JsValue::from_str(&code))?;
    let module = js::call1(device, "createShaderModule", &module_desc).map_err(error)?;
    let stage = object();
    field(&stage, "module", module)?;
    field(&stage, "entryPoint", JsValue::from_str("main"))?;
    let pipeline_desc = object();
    field(&pipeline_desc, "layout", JsValue::from_str("auto"))?;
    field(&pipeline_desc, "compute", stage)?;
    let pipeline = js::call1(device, "createComputePipeline", &pipeline_desc).map_err(error)?;
    let layout = js::call1(&pipeline, "getBindGroupLayout", &number(0u32)).map_err(error)?;
    let resource = object();
    field(&resource, "buffer", temporary.clone())?;
    let entry = object();
    field(&entry, "binding", number(0u32))?;
    field(&entry, "resource", resource)?;
    let entries = Array::new();
    entries.push(&entry);
    let group_desc = object();
    field(&group_desc, "layout", layout)?;
    field(&group_desc, "entries", entries.into())?;
    let group = js::call1(device, "createBindGroup", &group_desc).map_err(error)?;
    let pass = js::call1(&encoder.raw, "beginComputePass", &object()).map_err(error)?;
    js::call1(&pass, "setPipeline", &pipeline).map_err(error)?;
    js::call(&pass, "setBindGroup", &[number(0u32), group]).map_err(error)?;
    js::call1(&pass, "dispatchWorkgroups", &number(1u32)).map_err(error)?;
    js::call0(&pass, "end").map_err(error)?;
    js::call(
        &encoder.raw,
        "copyBufferToBuffer",
        &[
            temporary.clone(),
            number(0u32),
            dst.raw.clone(),
            number(word_offset as f64),
            number(4u32),
        ],
    )
    .map_err(error)?;
    encoder
        .keep_alive
        .extend([temporary, dst.raw.clone(), pipeline]);
    Ok(())
}

/// Copies a partial destination word through native storage buffers.
pub(super) fn copy_word(
    encoder: &mut WebGpuEncoder,
    src: &WebGpuBuffer,
    src_offset: u64,
    dst: &WebGpuBuffer,
    dst_offset: u64,
    count: usize,
) -> Result<()> {
    let first = (dst_offset % 4) as usize;
    if count == 0 || count > 1024 * 1024 {
        return Err(Error::InvalidRange);
    }
    let source_start = src_offset & !3;
    let source_first = (src_offset % 4) as usize;
    let source_size = ((source_first + count + 3) & !3) as u32;
    let dest_size = ((first + count + 3) & !3) as u32;
    let device = &encoder.device.0.device;
    let desc = object();
    field(&desc, "size", number(source_size + dest_size))?;
    field(&desc, "usage", number(4u32 | 8 | 128))?;
    let temp = js::call1(device, "createBuffer", &desc).map_err(error)?;
    for (buffer, offset, target, size) in [
        (src.raw.clone(), source_start, 0u32, source_size),
        (dst.raw.clone(), dst_offset & !3, source_size, dest_size),
    ] {
        js::call(
            &encoder.raw,
            "copyBufferToBuffer",
            &[
                buffer,
                number(offset as f64),
                temp.clone(),
                number(target),
                number(size),
            ],
        )
        .map_err(error)?;
    }
    let code = format!(
        "@group(0) @binding(0) var<storage, read_write> words: array<u32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{ let w = id.x; if (w >= {dest_words}u) {{ return; }} var result = words[{source_words}u + w]; for (var i = 0u; i < 4u; i++) {{ let d = w * 4u + i; if (d >= {first}u && d < {end}u) {{ let s = d - {first}u + {source_first}u; let byte = (words[s / 4u] >> ((s % 4u) * 8u)) & 255u; let shift = i * 8u; result = (result & ~(255u << shift)) | (byte << shift); }} }} words[{source_words}u + w] = result; }}",
        source_words = source_size / 4,
        dest_words = dest_size / 4,
        end = first + count
    );
    let module_desc = object();
    field(&module_desc, "code", JsValue::from_str(&code))?;
    let module = js::call1(device, "createShaderModule", &module_desc).map_err(error)?;
    let stage = object();
    field(&stage, "module", module)?;
    field(&stage, "entryPoint", JsValue::from_str("main"))?;
    let pipeline_desc = object();
    field(&pipeline_desc, "layout", JsValue::from_str("auto"))?;
    field(&pipeline_desc, "compute", stage)?;
    let pipeline = js::call1(device, "createComputePipeline", &pipeline_desc).map_err(error)?;
    let layout = js::call1(&pipeline, "getBindGroupLayout", &number(0u32)).map_err(error)?;
    let resource = object();
    field(&resource, "buffer", temp.clone())?;
    let entry = object();
    field(&entry, "binding", number(0u32))?;
    field(&entry, "resource", resource)?;
    let entries = Array::new();
    entries.push(&entry);
    let gd = object();
    field(&gd, "layout", layout)?;
    field(&gd, "entries", entries.into())?;
    let group = js::call1(device, "createBindGroup", &gd).map_err(error)?;
    let pass = js::call1(&encoder.raw, "beginComputePass", &object()).map_err(error)?;
    js::call1(&pass, "setPipeline", &pipeline).map_err(error)?;
    js::call(&pass, "setBindGroup", &[number(0u32), group]).map_err(error)?;
    js::call1(
        &pass,
        "dispatchWorkgroups",
        &number((dest_size / 4).div_ceil(64)),
    )
    .map_err(error)?;
    js::call0(&pass, "end").map_err(error)?;
    js::call(
        &encoder.raw,
        "copyBufferToBuffer",
        &[
            temp.clone(),
            number(source_size),
            dst.raw.clone(),
            number((dst_offset & !3) as f64),
            number(dest_size),
        ],
    )
    .map_err(error)?;
    encoder
        .keep_alive
        .extend([temp, pipeline, src.raw.clone(), dst.raw.clone()]);
    Ok(())
}
