//! WebGL polygon offset clamp extension entry point.
use wasm_bindgen::{JsCast, JsValue};
use web_sys::WebGl2RenderingContext as Gl;

pub(super) fn acquire(raw: &Gl) -> Option<(JsValue, js_sys::Function)> {
    let object = raw.get_extension("EXT_polygon_offset_clamp").ok()??;
    let method = js_sys::Reflect::get(&object, &JsValue::from_str("polygonOffsetClampEXT"))
        .ok()?
        .dyn_into()
        .ok()?;
    Some((object.into(), method))
}

pub(super) fn apply(
    raw: &Gl,
    command: Option<&(JsValue, js_sys::Function)>,
    factor: f32,
    units: f32,
    clamp: f32,
) -> Result<(), JsValue> {
    if clamp == 0.0 {
        raw.polygon_offset(factor, units);
        return Ok(());
    }
    let (object, method) =
        command.ok_or_else(|| JsValue::from_str("EXT_polygon_offset_clamp unavailable"))?;
    method.call3(
        object,
        &JsValue::from_f64(factor as f64),
        &JsValue::from_f64(units as f64),
        &JsValue::from_f64(clamp as f64),
    )?;
    Ok(())
}
