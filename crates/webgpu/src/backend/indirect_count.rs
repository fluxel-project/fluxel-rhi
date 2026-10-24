//! GPU-generated indirect arguments for a GPU-count multi-draw emulation.
//!
//! The caller must end the render pass before invoking this helper. Arguments
//! and the count are snapshotted at this command position; the output contains
//! tightly packed 16-byte (non-indexed) or 20-byte (indexed) commands.
use fluxel_rhi_core::api::{self, Error, Result};
use js_sys::{Array, Uint32Array};
use wasm_bindgen::JsValue;

use super::{
    command::{WebGpuEncoder, error, field, number, object},
    js,
    resource::{WebGpuBuffer, device_limit},
};

/// One fixed conversion program serves both indexed and non-indexed records.
/// Per-chunk shape is supplied through a small uniform buffer so recording no
/// longer recompiles WGSL or a compute pipeline for each chunk.
const CONVERTER_WGSL: &str = r#"
struct Parameters {
    base: u32,
    chunk_count: u32,
    words: u32,
    _padding: u32,
};
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read> count: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let draw = gid.y * nw.x * 64u + gid.x;
    if (draw >= parameters.chunk_count) { return; }
    for (var j = 0u; j < parameters.words; j++) {
        let i = draw * parameters.words + j;
        output[i] = select(0u, source[i], draw + parameters.base < count[0]);
    }
}
"#;

/// Check the complete operation before ending the active raster pass. This
/// does not create resources or record GPU commands.
pub(super) fn validate(
    encoder: &WebGpuEncoder,
    args: &WebGpuBuffer,
    args_offset: u64,
    count: &WebGpuBuffer,
    count_offset: u64,
    max_draw_count: u32,
    stride: u32,
    indexed: bool,
) -> Result<()> {
    validate_inner(
        encoder,
        args,
        args_offset,
        count,
        count_offset,
        max_draw_count,
        stride,
        indexed,
    )
    .map(|_| ())
}

fn validate_inner(
    encoder: &WebGpuEncoder,
    args: &WebGpuBuffer,
    args_offset: u64,
    count: &WebGpuBuffer,
    count_offset: u64,
    max_draw_count: u32,
    stride: u32,
    indexed: bool,
) -> Result<Option<(u64, u32, u64)>> {
    let command_size = if indexed { 20u64 } else { 16u64 };
    if !args.usage.contains(api::BufferUsage::INDIRECT)
        || !count.usage.contains(api::BufferUsage::INDIRECT)
        || args.mapped.get()
        || count.mapped.get()
        || args_offset % 4 != 0
        || count_offset % 4 != 0
        || args_offset > args.size
        || count_offset
            .checked_add(4)
            .is_none_or(|end| end > count.size)
        || (max_draw_count != 0 && (u64::from(stride) < command_size || stride % 4 != 0))
    {
        return Err(Error::InvalidRange);
    }
    if max_draw_count == 0 {
        return Ok(None);
    }
    let source_end = args_offset
        .checked_add(u64::from(max_draw_count - 1) * u64::from(stride))
        .and_then(|n| n.checked_add(command_size))
        .ok_or(Error::InvalidRange)?;
    if source_end > args.size || source_end > (1u64 << 53) || count_offset + 4 > (1u64 << 53) {
        return Err(Error::InvalidRange);
    }
    let device = &encoder.device.0.device;
    for source in [args, count] {
        let native_usage = js::property(&source.raw, "usage")
            .map_err(error)?
            .as_f64()
            .ok_or(Error::InvalidState)? as u32;
        if native_usage & 4 == 0 {
            return Err(Error::Unsupported("WebGPU indirect source COPY_SRC"));
        }
    }
    let max_dimension =
        device_limit(device, "maxComputeWorkgroupsPerDimension").min(u64::from(u32::MAX) / 64);
    let scratch_limit = device_limit(device, "maxBufferSize")
        .min(device_limit(device, "maxStorageBufferBindingSize"));
    let dispatch_limit = if max_dimension == 0 {
        0
    } else {
        max_dimension * max_dimension.min((u64::from(u32::MAX) / 64) / max_dimension) * 64
    };
    let capacity = (scratch_limit / command_size)
        .min(u64::from(u32::MAX) / (command_size / 4))
        .min(dispatch_limit)
        .min(u64::from(u32::MAX));
    if capacity == 0 || max_dimension == 0 {
        return Err(Error::Unsupported("WebGPU indirect count device limits"));
    }
    Ok(Some((command_size, capacity as u32, max_dimension)))
}

/// Prepare GPU-masked commands. Returns no chunks when `max_draw_count` is zero.
/// Both inputs must have native COPY_SRC usage in addition to Core INDIRECT.
pub(super) fn prepare(
    encoder: &mut WebGpuEncoder,
    args: &WebGpuBuffer,
    args_offset: u64,
    count: &WebGpuBuffer,
    count_offset: u64,
    max_draw_count: u32,
    stride: u32,
    indexed: bool,
) -> Result<Vec<(JsValue, u32)>> {
    if encoder.pass.is_some() {
        return Err(Error::InvalidState);
    }
    let Some((command_size, capacity, max_dimension)) = validate_inner(
        encoder,
        args,
        args_offset,
        count,
        count_offset,
        max_draw_count,
        stride,
        indexed,
    )?
    else {
        return Ok(Vec::new());
    };
    let device = &encoder.device.0.device;
    let create_buffer = |size: u64, usage: u32| -> Result<JsValue> {
        let desc = object();
        field(&desc, "size", number(size as f64))?;
        field(&desc, "usage", number(usage))?;
        js::call1(device, "createBuffer", &desc).map_err(error)
    };
    let count_copy = create_buffer(4, 8 | 128)?;
    js::call(
        &encoder.raw,
        "copyBufferToBuffer",
        &[
            count.raw.clone(),
            number(count_offset as f64),
            count_copy.clone(),
            number(0u32),
            number(4u32),
        ],
    )
    .map_err(error)?;
    let pipeline = super::utility::pipeline(&encoder.device, "indirect-count", CONVERTER_WGSL)?;
    let layout = js::call1(&pipeline, "getBindGroupLayout", &number(0u32)).map_err(error)?;
    let mut outputs = Vec::new();
    let mut base = 0u32;
    while base < max_draw_count {
        let chunk_count = (max_draw_count - base).min(capacity);
        let packed_size = command_size * u64::from(chunk_count);
        let workgroups = u64::from(chunk_count).div_ceil(64);
        let x = workgroups.min(max_dimension);
        let y = workgroups.div_ceil(x);
        let input = create_buffer(packed_size, 8 | 128)?;
        let output = create_buffer(packed_size, 128 | 256)?;
        let parameter_desc = object();
        field(&parameter_desc, "size", number(16u32))?;
        field(&parameter_desc, "usage", number(64u32))?;
        field(&parameter_desc, "mappedAtCreation", JsValue::TRUE)?;
        let parameters = js::call1(device, "createBuffer", &parameter_desc).map_err(error)?;
        let mapped = js::call0(&parameters, "getMappedRange").map_err(error)?;
        let parameter_words = Uint32Array::new(&mapped);
        parameter_words.set_index(0, base);
        parameter_words.set_index(1, chunk_count);
        parameter_words.set_index(2, (command_size / 4) as u32);
        js::call0(&parameters, "unmap").map_err(error)?;
        let entries = Array::new();
        for (binding, buffer) in [&input, &count_copy, &output, &parameters]
            .into_iter()
            .enumerate()
        {
            let resource = object();
            field(&resource, "buffer", buffer.clone())?;
            let item = object();
            field(&item, "binding", number(binding as u32))?;
            field(&item, "resource", resource)?;
            entries.push(&item);
        }
        let group_desc = object();
        field(&group_desc, "layout", layout.clone())?;
        field(&group_desc, "entries", entries.into())?;
        let group = js::call1(device, "createBindGroup", &group_desc).map_err(error)?;
        if u64::from(stride) == command_size {
            // Tight records already match the conversion buffer's packed ABI,
            // so one native copy replaces one command-encoder call per draw.
            js::call(
                &encoder.raw,
                "copyBufferToBuffer",
                &[
                    args.raw.clone(),
                    number((args_offset + u64::from(base) * command_size) as f64),
                    input.clone(),
                    number(0u32),
                    number(packed_size as f64),
                ],
            )
            .map_err(error)?;
        } else {
            // Padding is part of the source ABI. Keep the established direct
            // copies until a separately measured GPU packing path is justified.
            for i in 0..chunk_count {
                js::call(
                    &encoder.raw,
                    "copyBufferToBuffer",
                    &[
                        args.raw.clone(),
                        number(
                            (args_offset + (u64::from(base) + u64::from(i)) * u64::from(stride))
                                as f64,
                        ),
                        input.clone(),
                        number((u64::from(i) * command_size) as f64),
                        number(command_size as f64),
                    ],
                )
                .map_err(error)?;
            }
        }
        let pass = js::call1(&encoder.raw, "beginComputePass", &object()).map_err(error)?;
        js::call1(&pass, "setPipeline", &pipeline).map_err(error)?;
        js::call(&pass, "setBindGroup", &[number(0u32), group.clone()]).map_err(error)?;
        js::call(
            &pass,
            "dispatchWorkgroups",
            &[number(x as f64), number(y as f64)],
        )
        .map_err(error)?;
        js::call0(&pass, "end").map_err(error)?;
        encoder
            .keep_alive
            .extend([input, output.clone(), parameters, group]);
        outputs.push((output, chunk_count));
        base += chunk_count;
    }
    encoder
        .keep_alive
        .extend([args.raw.clone(), count.raw.clone(), count_copy]);
    Ok(outputs)
}
