//! WebGL extension lowering for polygon mode and depth clamp.
use crate::backend::gl::api::{GlPolygonMode, GlRasterizationState};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::WebGl2RenderingContext as Gl;

pub(super) fn polygon_mode_command(raw: &Gl) -> Option<(JsValue, js_sys::Function)> {
    let object = raw.get_extension("WEBGL_polygon_mode").ok()??;
    let method = js_sys::Reflect::get(&object, &JsValue::from_str("polygonModeWEBGL"))
        .ok()?
        .dyn_into()
        .ok()?;
    Some((object.into(), method))
}

pub(super) fn apply(raw: &Gl, state: GlRasterizationState) -> Result<(), &'static str> {
    const LINE: u32 = 0x1B01;
    const FILL: u32 = 0x1B02;
    const DEPTH_CLAMP: u32 = 0x864F;
    if state.conservative || state.polygon_mode == GlPolygonMode::Point {
        return Err("WebGL2 rasterization state");
    }
    let polygon = polygon_mode_command(raw);
    let depth = raw
        .get_extension("EXT_depth_clamp")
        .ok()
        .flatten()
        .is_some();
    if state.polygon_mode == GlPolygonMode::Line && polygon.is_none() {
        return Err("WEBGL_polygon_mode unavailable");
    }
    if state.depth_clamp && !depth {
        return Err("EXT_depth_clamp unavailable");
    }
    if let Some((object, function)) = polygon {
        let mode = if state.polygon_mode == GlPolygonMode::Line {
            LINE
        } else {
            FILL
        };
        function
            .call2(
                &object,
                &JsValue::from_f64(Gl::FRONT_AND_BACK as f64),
                &JsValue::from_f64(mode as f64),
            )
            .map_err(|_| "polygonModeWEBGL failed")?;
    }
    if depth {
        if state.depth_clamp {
            raw.enable(DEPTH_CLAMP);
        } else {
            raw.disable(DEPTH_CLAMP);
        }
    }
    Ok(())
}

/// The extension has a distinct offset enable for line polygon mode.
pub(super) fn line_offset(raw: &Gl, enabled: bool) {
    if polygon_mode_command(raw).is_some() {
        if enabled {
            raw.enable(0x2A02);
        } else {
            raw.disable(0x2A02);
        }
    }
}
