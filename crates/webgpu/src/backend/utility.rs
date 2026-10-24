//! Device-local immutable pipelines for fixed native lowering kernels.
//!
//! Each private key names one fixed WGSL program. Per-operation data belongs to
//! fresh native parameter buffers, never the source/cache key. The cache holds
//! only browser pipeline objects, with no Rust device ownership cycle.

use super::{
    command::unaligned::{COPY, PATCH},
    indirect_count::CONVERTER_WGSL,
    instance::WebGpuDevice,
    js,
    query_segments::REDUCE_WGSL,
};
use fluxel_rhi_core::api::{Error, Result};
use js_sys::Promise;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

const FIXED_PIPELINES: [(&str, &str); 4] = [
    ("word-patch", PATCH),
    ("word-copy", COPY),
    ("indirect-count", CONVERTER_WGSL),
    ("query-segment-reduce", REDUCE_WGSL),
];

/// `key` may contain native format/sample/shape variants, never per-operation
/// colors, offsets or sizes. The closure creates only immutable shader/PSO
/// objects and runs once for each supported private variant on this device.
pub(super) fn render_pipeline(
    device: &WebGpuDevice,
    key: String,
    create: impl FnOnce() -> Result<JsValue>,
) -> Result<JsValue> {
    if let Some(pipeline) = device.0.render_utility_pipelines.borrow().get(&key) {
        return Ok(pipeline.clone());
    }
    let pipeline = create()?;
    device
        .0
        .render_utility_pipelines
        .borrow_mut()
        .insert(key, pipeline.clone());
    Ok(pipeline)
}

/// Creates all finite compute utility pipelines while device opening is already
/// asynchronous. Command recording can subsequently use only immutable cached
/// native objects, never a synchronous browser pipeline creation.
pub(super) async fn prewarm_fixed(device: &WebGpuDevice) -> Result<()> {
    for (key, code) in FIXED_PIPELINES {
        if device.0.utility_pipelines.borrow().contains_key(key) {
            continue;
        }
        let pipeline = create_compute_pipeline_async(device, code).await?;
        device
            .0
            .utility_pipelines
            .borrow_mut()
            .insert(key, pipeline);
    }
    Ok(())
}

async fn create_compute_pipeline_async(device: &WebGpuDevice, code: &str) -> Result<JsValue> {
    let error = |value| Error::Backend(js::message(&value));
    // Keep validation scoped to this private cold object. The pop happens even
    // when async creation rejects, so a failed prewarm cannot poison later
    // browser validation or leave an error scope on the device stack.
    js::call1(
        &device.0.device,
        "pushErrorScope",
        &JsValue::from_str("validation"),
    )
    .map_err(error)?;
    let promise = (|| -> Result<Promise> {
        let module_desc: JsValue = js_sys::Object::new().into();
        js::set(&module_desc, "code", &JsValue::from_str(code)).map_err(error)?;
        let module =
            js::call1(&device.0.device, "createShaderModule", &module_desc).map_err(error)?;
        let stage: JsValue = js_sys::Object::new().into();
        js::set(&stage, "module", &module).map_err(error)?;
        js::set(&stage, "entryPoint", &JsValue::from_str("main")).map_err(error)?;
        let desc: JsValue = js_sys::Object::new().into();
        js::set(&desc, "layout", &JsValue::from_str("auto")).map_err(error)?;
        js::set(&desc, "compute", &stage).map_err(error)?;
        js::call1(&device.0.device, "createComputePipelineAsync", &desc)
            .map(Promise::from)
            .map_err(error)
    })();
    let created = match promise {
        Ok(promise) => JsFuture::from(promise).await.map_err(error),
        Err(error) => Err(error),
    };
    let scoped = match js::call0(&device.0.device, "popErrorScope") {
        Ok(promise) => JsFuture::from(Promise::from(promise)).await.map_err(error),
        Err(value) => Err(error(value)),
    };
    let pipeline = match created {
        Ok(pipeline) => pipeline,
        Err(error) => {
            let _ = scoped;
            return Err(error);
        }
    };
    let scoped = scoped?;
    if !scoped.is_null() && !scoped.is_undefined() {
        return Err(Error::Backend(js::message(&scoped)));
    }
    Ok(pipeline)
}

/// Returns a prepared fixed utility pipeline during command recording.
///
/// A miss indicates a device-open invariant failure. Recording cannot await a
/// browser promise and must not revive the former synchronous pipeline path.
pub(super) fn pipeline(device: &WebGpuDevice, key: &'static str, _: &str) -> Result<JsValue> {
    if let Some(pipeline) = device.0.utility_pipelines.borrow().get(key) {
        return Ok(pipeline.clone());
    }
    Err(Error::InvalidState)
}
