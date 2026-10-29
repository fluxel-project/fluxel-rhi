//! Reopens native raster passes around backend GPU work or sample-count changes.
//!
//! Only native binding state is retained. Draws are never cached or replayed.
use super::{
    command::{WebGpuEncoder, error, field, number, object},
    js,
};
use fluxel_rhi_core::api::{Error, Result};
use js_sys::{Array, Object, Reflect, WeakMap};
use wasm_bindgen::{JsCast, JsValue};
thread_local! { static STATES: WeakMap = WeakMap::new(); }
fn get(pass: &JsValue) -> JsValue {
    STATES.with(|s| s.get(pass.unchecked_ref::<Object>()))
}
fn put(pass: &JsValue, state: &JsValue) {
    STATES.with(|s| {
        s.set(pass.unchecked_ref::<Object>(), state);
    });
}
pub(super) fn begin(
    pass: &JsValue,
    descriptor: &JsValue,
    width: u32,
    height: u32,
    dummy: bool,
) -> Result<()> {
    let state = object();
    field(&state, "descriptor", descriptor.clone())?;
    field(&state, "width", number(width.max(1)))?;
    field(&state, "height", number(height.max(1)))?;
    field(&state, "samples", number(1u32))?;
    field(&state, "dummy", JsValue::from_bool(dummy))?;
    field(&state, "records", Array::new().into())?;
    put(pass, &state);
    Ok(())
}
pub(super) fn record(pass: &JsValue, key: &str, method: &str, args: &[JsValue]) -> Result<()> {
    let state = get(pass);
    if state.is_undefined() {
        return Ok(());
    }
    let records = Array::from(&js::property(&state, "records").map_err(error)?);
    let value = object();
    field(&value, "key", JsValue::from_str(key))?;
    field(&value, "method", JsValue::from_str(method))?;
    let array = Array::new();
    for arg in args {
        array.push(arg);
    }
    field(&value, "args", array.into())?;
    for i in 0..records.length() {
        if js::property(&records.get(i), "key")
            .map_err(error)?
            .as_string()
            .as_deref()
            == Some(key)
        {
            records.set(i, value);
            field(&state, "records", records.into())?;
            return Ok(());
        }
    }
    records.push(&value);
    field(&state, "records", records.into())?;
    Ok(())
}
pub(super) struct Suspended {
    old: JsValue,
    state: JsValue,
}
pub(super) fn suspend(encoder: &mut WebGpuEncoder) -> Result<Suspended> {
    let old = encoder.pass.clone().ok_or(Error::InvalidState)?;
    let state = get(&old);
    if state.is_undefined() {
        return Err(Error::InvalidState);
    }
    validate_split_stores(&state)?;
    super::query_segments::before_split(encoder)?;
    // WebGPU requires each native GPURenderPassEncoder to close with a
    // balanced debug stack. Core keeps the logical groups open across this
    // internal split, so close the old native segment from innermost to
    // outermost; `resume` records the same labels on the replacement pass.
    for _ in encoder.debug_groups.iter().rev() {
        js::call0(&old, "popDebugGroup").map_err(error)?;
    }
    js::call0(&old, "end").map_err(error)?;
    encoder.pass = None;
    super::query_segments::flush_pending(encoder)?;
    Ok(Suspended { old, state })
}

pub(crate) fn validate_split_stores(state: &JsValue) -> Result<()> {
    let descriptor = js::property(state, "descriptor").map_err(error)?;
    let colors = Array::from(&js::property(&descriptor, "colorAttachments").map_err(error)?);
    for attachment in colors.iter() {
        if attachment.is_null() || attachment.is_undefined() {
            continue;
        }
        let store = js::property(&attachment, "storeOp").map_err(error)?;
        if store.as_string().as_deref() == Some("discard") {
            return Err(Error::Unsupported(
                "WebGPU internal pass split with discarded attachments",
            ));
        }
    }

    let depth =
        Reflect::get(&descriptor, &JsValue::from_str("depthStencilAttachment")).map_err(error)?;
    if depth.is_null() || depth.is_undefined() {
        return Ok(());
    }
    for (store_name, read_only_name) in [
        ("depthStoreOp", "depthReadOnly"),
        ("stencilStoreOp", "stencilReadOnly"),
    ] {
        let read_only = js::property(&depth, read_only_name).map_err(error)?;
        if read_only.as_bool().unwrap_or(false) {
            continue;
        }
        let store = js::property(&depth, store_name).map_err(error)?;
        if store.as_string().as_deref() == Some("discard") {
            return Err(Error::Unsupported(
                "WebGPU internal pass split with discarded attachments",
            ));
        }
    }
    Ok(())
}
pub(super) fn resume(
    encoder: &mut WebGpuEncoder,
    suspended: Suspended,
    samples: Option<u32>,
    restore_pipeline: bool,
) -> Result<()> {
    let state = suspended.state;
    let descriptor = js::property(&state, "descriptor").map_err(error)?;
    let colors = Array::from(&js::property(&descriptor, "colorAttachments").map_err(error)?);
    for attachment in colors.iter() {
        if !attachment.is_null() && !attachment.is_undefined() {
            field(&attachment, "loadOp", JsValue::from_str("load"))?;
        }
    }
    let attachment =
        Reflect::get(&descriptor, &JsValue::from_str("depthStencilAttachment")).map_err(error)?;
    if !attachment.is_undefined() && !attachment.is_null() {
        for (load, readonly) in [
            ("depthLoadOp", "depthReadOnly"),
            ("stencilLoadOp", "stencilReadOnly"),
        ] {
            let existing = Reflect::get(&attachment, &JsValue::from_str(load)).map_err(error)?;
            let ro = Reflect::get(&attachment, &JsValue::from_str(readonly)).map_err(error)?;
            if !existing.is_undefined() && !ro.as_bool().unwrap_or(false) {
                field(&attachment, load, JsValue::from_str("load"))?;
            }
        }
    }
    if let Some(samples) = samples {
        let texture_desc = object();
        let size = object();
        field(
            &size,
            "width",
            js::property(&state, "width").map_err(error)?,
        )?;
        field(
            &size,
            "height",
            js::property(&state, "height").map_err(error)?,
        )?;
        field(&size, "depthOrArrayLayers", number(1u32))?;
        field(&texture_desc, "size", size)?;
        field(&texture_desc, "format", JsValue::from_str("depth32float"))?;
        field(&texture_desc, "usage", number(16u32))?;
        field(&texture_desc, "sampleCount", number(samples))?;
        let texture =
            js::call1(&encoder.device.0.device, "createTexture", &texture_desc).map_err(error)?;
        let view = js::call1(&texture, "createView", &object()).map_err(error)?;
        field(&attachment, "view", view.clone())?;
        // A replacement dummy texture has no contents to load.
        field(&attachment, "depthLoadOp", JsValue::from_str("clear"))?;
        encoder.keep_alive.extend([texture, view]);
        field(&state, "samples", number(samples))?;
    }
    let timestamp =
        Reflect::get(&descriptor, &JsValue::from_str("timestampWrites")).map_err(error)?;
    if !timestamp.is_undefined() && !timestamp.is_null() {
        Reflect::delete_property(
            timestamp.unchecked_ref::<Object>(),
            &JsValue::from_str("beginningOfPassWriteIndex"),
        )
        .map_err(error)?;
    }
    super::query_segments::prepare_split(encoder, &descriptor)?;
    let new_pass = js::call1(&encoder.raw, "beginRenderPass", &descriptor).map_err(error)?;
    encoder.pass = Some(new_pass.clone());
    super::command::restore_debug_groups(&encoder.debug_groups, &new_pass)?;
    super::immediate::transfer(&suspended.old, &new_pass);
    put(&new_pass, &state);
    super::query_segments::after_split(encoder)?;
    let records = Array::from(&js::property(&state, "records").map_err(error)?);
    // Restore pipeline before bindings so its immediate ABI is installed first.
    if restore_pipeline {
        for entry in records.iter() {
            if js::property(&entry, "key")
                .map_err(error)?
                .as_string()
                .as_deref()
                == Some("pipeline")
            {
                let args = Array::from(&js::property(&entry, "args").map_err(error)?);
                let pipeline = args.get(0);
                js::call1(&new_pass, "setPipeline", &pipeline).map_err(error)?;
                super::immediate::bind_pipeline(&new_pass, &pipeline, false)?;
            }
        }
    }
    for entry in records.iter() {
        if js::property(&entry, "key")
            .map_err(error)?
            .as_string()
            .as_deref()
            == Some("pipeline")
        {
            continue;
        }
        let method = js::property(&entry, "method")
            .map_err(error)?
            .as_string()
            .ok_or(Error::InvalidState)?;
        let args = Array::from(&js::property(&entry, "args").map_err(error)?);
        let values: Vec<_> = args.iter().collect();
        js::call(&new_pass, &method, &values).map_err(error)?;
    }
    Ok(())
}
pub(super) fn split(encoder: &mut WebGpuEncoder, samples: u32) -> Result<()> {
    let old = encoder.pass.clone().ok_or(Error::InvalidState)?;
    let state = get(&old);
    if state.is_undefined()
        || !js::property(&state, "dummy")
            .map_err(error)?
            .as_bool()
            .unwrap_or(false)
    {
        return Ok(());
    }
    if js::property(&state, "samples").map_err(error)?.as_f64() == Some(samples as f64) {
        return Ok(());
    }
    let suspended = suspend(encoder)?;
    resume(encoder, suspended, Some(samples), false)
}
