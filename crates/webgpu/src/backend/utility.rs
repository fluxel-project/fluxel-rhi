//! Device-local immutable pipelines for fixed native lowering kernels.
//!
//! Each private key names one fixed WGSL program. Per-operation data belongs to
//! fresh native parameter buffers, never the source/cache key. The cache holds
//! only browser pipeline objects, with no Rust device ownership cycle.

use super::{instance::WebGpuDevice, js};
use fluxel_rhi_core::api::{Error, Result};
use wasm_bindgen::JsValue;

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

pub(super) fn pipeline(device: &WebGpuDevice, key: &'static str, code: &str) -> Result<JsValue> {
    if let Some(pipeline) = device.0.utility_pipelines.borrow().get(key) {
        return Ok(pipeline.clone());
    }
    let error = |value| Error::Backend(js::message(&value));
    let module_desc: JsValue = js_sys::Object::new().into();
    js::set(&module_desc, "code", &JsValue::from_str(code)).map_err(error)?;
    let module = js::call1(&device.0.device, "createShaderModule", &module_desc).map_err(error)?;
    let stage: JsValue = js_sys::Object::new().into();
    js::set(&stage, "module", &module).map_err(error)?;
    js::set(&stage, "entryPoint", &JsValue::from_str("main")).map_err(error)?;
    let desc: JsValue = js_sys::Object::new().into();
    js::set(&desc, "layout", &JsValue::from_str("auto")).map_err(error)?;
    js::set(&desc, "compute", &stage).map_err(error)?;
    let pipeline = js::call1(&device.0.device, "createComputePipeline", &desc).map_err(error)?;
    device
        .0
        .utility_pipelines
        .borrow_mut()
        .insert(key, pipeline.clone());
    Ok(pipeline)
}
