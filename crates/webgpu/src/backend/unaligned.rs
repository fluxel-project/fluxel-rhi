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
    let pipeline = super::super::utility::pipeline(&encoder.device, "word-patch", PATCH)?;
    execute(encoder, &temporary, &pipeline, &[mask, value, 0, 0], 1)?;
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
    let pipeline = super::super::utility::pipeline(&encoder.device, "word-copy", COPY)?;
    execute(
        encoder,
        &temp,
        &pipeline,
        &[
            source_size / 4,
            dest_size / 4,
            first as u32,
            (first + count) as u32,
            source_first as u32,
            0,
            0,
            0,
        ],
        (dest_size / 4).div_ceil(64),
    )?;
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

// Parameter data changes per operation while these WGSL kernels stay fixed.
pub(in crate::backend) const PATCH: &str = r#"
struct Params { mask: u32, value: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read_write> word: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(1) fn main() {
    word[0] = (word[0] & ~params.mask) | params.value;
}
"#;
pub(in crate::backend) const COPY: &str = r#"
struct Params {
    source_words: u32, dest_words: u32, first: u32, end: u32,
    source_first: u32, pad0: u32, pad1: u32, pad2: u32
}
@group(0) @binding(0) var<storage, read_write> words: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = id.x;
    if (w >= params.dest_words) { return; }
    var result = words[params.source_words + w];
    for (var i = 0u; i < 4u; i++) {
        let d = w * 4u + i;
        if (d >= params.first && d < params.end) {
            let s = d - params.first + params.source_first;
            let byte = (words[s / 4u] >> ((s % 4u) * 8u)) & 255u;
            let shift = i * 8u;
            result = (result & ~(255u << shift)) | (byte << shift);
        }
    }
    words[params.source_words + w] = result;
}
"#;

fn execute(
    encoder: &mut WebGpuEncoder,
    temporary: &JsValue,
    pipeline: &JsValue,
    parameters: &[u32],
    groups: u32,
) -> Result<()> {
    let desc = object();
    field(&desc, "size", number((parameters.len() * 4) as u32))?;
    field(&desc, "usage", number(64u32))?; // GPUBufferUsage::UNIFORM
    field(&desc, "mappedAtCreation", JsValue::TRUE)?;
    let params = js::call1(&encoder.device.0.device, "createBuffer", &desc).map_err(error)?;
    let mapped = js::call0(&params, "getMappedRange").map_err(error)?;
    let bytes: Vec<_> = parameters
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    js_sys::Uint8Array::new(&mapped).copy_from(&bytes);
    js::call0(&params, "unmap").map_err(error)?;
    let entries = Array::new();
    for (binding, buffer) in [temporary, &params].into_iter().enumerate() {
        let resource = object();
        field(&resource, "buffer", buffer.clone())?;
        let entry = object();
        field(&entry, "binding", number(binding as u32))?;
        field(&entry, "resource", resource)?;
        entries.push(&entry);
    }
    let desc = object();
    field(
        &desc,
        "layout",
        js::call1(pipeline, "getBindGroupLayout", &number(0u32)).map_err(error)?,
    )?;
    field(&desc, "entries", entries.into())?;
    let group = js::call1(&encoder.device.0.device, "createBindGroup", &desc).map_err(error)?;
    let pass = js::call1(&encoder.raw, "beginComputePass", &object()).map_err(error)?;
    js::call1(&pass, "setPipeline", pipeline).map_err(error)?;
    js::call(&pass, "setBindGroup", &[number(0u32), group.clone()]).map_err(error)?;
    js::call1(&pass, "dispatchWorkgroups", &number(groups)).map_err(error)?;
    js::call0(&pass, "end").map_err(error)?;
    encoder.keep_alive.extend([params, group]);
    Ok(())
}
