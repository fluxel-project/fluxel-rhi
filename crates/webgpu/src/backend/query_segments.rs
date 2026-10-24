//! Native query segments across split render passes, and GPU-side summation.
use super::{
    command::{WebGpuEncoder, error, field, number, object},
    js,
    query::WebGpuQuerySet,
};
use fluxel_rhi_core::api::{Error, Result};
use js_sys::{Array, Uint32Array};
use wasm_bindgen::JsValue;

/// Fixed GPU reduction kernel. Dynamic segment count and logical query slot
/// are supplied by a uniform buffer, so split passes reuse one device-local
/// compute pipeline instead of compiling WGSL for every flush.
const REDUCE_WGSL: &str = r#"
struct Parameters {
    count: u32,
    slot: u32,
    _padding0: u32,
    _padding1: u32,
};
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> parameters: Parameters;
@compute @workgroup_size(1)
fn main() {
    var lo: u32 = 0u;
    var hi: u32 = 0u;
    for (var i: u32 = 0u; i < parameters.count; i++) {
        let at = i * 64u;
        let old = lo;
        lo += src[at];
        hi += src[at + 1u] + select(0u, 1u, lo < old);
    }
    dst[parameters.slot * 2u] = lo;
    dst[parameters.slot * 2u + 1u] = hi;
}
"#;

pub(super) struct QuerySegments {
    pub(super) set: WebGpuQuerySet,
    pub(super) slot: u32,
    pub(super) native: Vec<(JsValue, u32)>,
}

pub(super) fn before_split(encoder: &mut WebGpuEncoder) -> Result<()> {
    if encoder.occlusion_active {
        let pass = encoder.pass.as_ref().ok_or(Error::InvalidState)?;
        js::call0(pass, "endOcclusionQuery").map_err(error)?;
    }
    Ok(())
}
pub(super) fn prepare_split(encoder: &mut WebGpuEncoder, descriptor: &JsValue) -> Result<()> {
    encoder.occlusion_next_index = 0;
    if let Some(active) = &mut encoder.occlusion_active_query {
        let desc = object();
        field(&desc, "type", JsValue::from_str("occlusion"))?;
        field(&desc, "count", number(4096u32))?;
        let native = js::call1(&encoder.device.0.device, "createQuerySet", &desc).map_err(error)?;
        field(descriptor, "occlusionQuerySet", native.clone())?;
        active.native.push((native.clone(), 0));
        encoder.occlusion_next_index = 1;
        encoder.keep_alive.push(native.clone());
        encoder.occlusion_query_set = Some(native);
    }
    Ok(())
}
pub(super) fn after_split(encoder: &mut WebGpuEncoder) -> Result<()> {
    if let Some(active) = &encoder.occlusion_active_query {
        let pass = encoder.pass.as_ref().ok_or(Error::InvalidState)?;
        let physical = active.native.last().ok_or(Error::InvalidState)?.1;
        js::call1(pass, "beginOcclusionQuery", &number(physical)).map_err(error)?;
    }
    Ok(())
}

fn buffer(encoder: &WebGpuEncoder, size: u64, usage: u32) -> Result<JsValue> {
    let desc = object();
    field(&desc, "size", JsValue::from_f64(size as f64))?;
    field(&desc, "usage", number(usage))?;
    js::call1(&encoder.device.0.device, "createBuffer", &desc).map_err(error)
}

pub(super) fn flush_pending(encoder: &mut WebGpuEncoder) -> Result<()> {
    // A render pass cannot resolve queries between draws. For repeated writes
    // to one logical slot, only its last result is observable after the pass.
    let mut latest: Vec<QuerySegments> = Vec::new();
    for query in std::mem::take(&mut encoder.occlusion_pending)
        .into_iter()
        .rev()
    {
        if latest.iter().any(|previous| {
            previous.slot == query.slot && js_sys::Object::is(&previous.set.raw, &query.set.raw)
        }) {
            continue;
        }
        latest.push(query);
    }
    for query in latest.into_iter().rev() {
        flush_one(encoder, query)?;
    }
    Ok(())
}
fn flush_one(encoder: &mut WebGpuEncoder, query: QuerySegments) -> Result<()> {
    let count = query.native.len();
    let count_u32 = u32::try_from(count).map_err(|_| Error::InvalidRange)?;
    let bytes = (count as u64).checked_mul(256).ok_or(Error::InvalidRange)?;
    let scratch = buffer(encoder, bytes, 512 | 128)?;
    let output = query
        .set
        .results
        .as_ref()
        .ok_or(Error::InvalidState)?
        .clone();
    for (index, (raw, physical)) in query.native.iter().enumerate() {
        js::call(
            &encoder.raw,
            "resolveQuerySet",
            &[
                raw.clone(),
                number(*physical),
                number(1u32),
                scratch.clone(),
                JsValue::from_f64((index * 256) as f64),
            ],
        )
        .map_err(error)?;
    }
    let pipeline = super::utility::pipeline(&encoder.device, "query-segment-reduce", REDUCE_WGSL)?;
    let layout = js::call1(&pipeline, "getBindGroupLayout", &number(0u32)).map_err(error)?;
    let parameter_desc = object();
    field(&parameter_desc, "size", number(16u32))?;
    field(&parameter_desc, "usage", number(64u32))?;
    field(&parameter_desc, "mappedAtCreation", JsValue::TRUE)?;
    let parameters =
        js::call1(&encoder.device.0.device, "createBuffer", &parameter_desc).map_err(error)?;
    let mapped = js::call0(&parameters, "getMappedRange").map_err(error)?;
    let words = Uint32Array::new(&mapped);
    words.set_index(0, count_u32);
    words.set_index(1, query.slot);
    js::call0(&parameters, "unmap").map_err(error)?;
    let entries_js = Array::new();
    for (binding, raw) in [(0u32, &scratch), (1u32, &output), (2u32, &parameters)] {
        let resource = object();
        field(&resource, "buffer", raw.clone())?;
        let entry = object();
        field(&entry, "binding", number(binding))?;
        field(&entry, "resource", resource)?;
        entries_js.push(&entry);
    }
    let group_desc = object();
    field(&group_desc, "layout", layout)?;
    field(&group_desc, "entries", entries_js.into())?;
    let group =
        js::call1(&encoder.device.0.device, "createBindGroup", &group_desc).map_err(error)?;
    let pass = js::call1(&encoder.raw, "beginComputePass", &object()).map_err(error)?;
    js::call1(&pass, "setPipeline", &pipeline).map_err(error)?;
    js::call(&pass, "setBindGroup", &[number(0u32), group.clone()]).map_err(error)?;
    js::call1(&pass, "dispatchWorkgroups", &number(1u32)).map_err(error)?;
    js::call0(&pass, "end").map_err(error)?;
    encoder
        .keep_alive
        .extend([scratch, output, parameters, group]);
    encoder
        .keep_alive
        .extend(query.native.into_iter().map(|(raw, _)| raw));
    Ok(())
}
