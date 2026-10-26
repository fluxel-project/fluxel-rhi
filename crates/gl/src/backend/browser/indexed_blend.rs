//! Apply indexed blend state only when the WebGL extension is available.
use super::super::api::{GlBlendFactor as F, GlBlendOperation as O, GlColorTargetState};
use js_sys::{Array, Function, Reflect};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::WebGl2RenderingContext as Gl;
fn factor(v: F) -> u32 {
    match v {
        F::Zero => Gl::ZERO,
        F::One => Gl::ONE,
        F::Src => Gl::SRC_COLOR,
        F::OneMinusSrc => Gl::ONE_MINUS_SRC_COLOR,
        F::SrcAlpha => Gl::SRC_ALPHA,
        F::OneMinusSrcAlpha => Gl::ONE_MINUS_SRC_ALPHA,
        F::Dst => Gl::DST_COLOR,
        F::OneMinusDst => Gl::ONE_MINUS_DST_COLOR,
        F::DstAlpha => Gl::DST_ALPHA,
        F::OneMinusDstAlpha => Gl::ONE_MINUS_DST_ALPHA,
        F::Constant => Gl::CONSTANT_COLOR,
        F::OneMinusConstant => Gl::ONE_MINUS_CONSTANT_COLOR,
        F::SrcAlphaSaturated => Gl::SRC_ALPHA_SATURATE,
        F::Src1 => 0x88F9,
        F::OneMinusSrc1 => 0x88FA,
        F::Src1Alpha => 0x8589,
        F::OneMinusSrc1Alpha => 0x88FB,
    }
}
fn op(v: O) -> u32 {
    match v {
        O::Add => Gl::FUNC_ADD,
        O::Subtract => Gl::FUNC_SUBTRACT,
        O::ReverseSubtract => Gl::FUNC_REVERSE_SUBTRACT,
        O::Min => Gl::MIN,
        O::Max => Gl::MAX,
    }
}
pub(super) fn apply(raw: &Gl, targets: &[GlColorTargetState]) -> Result<(), &'static str> {
    let ext = raw
        .get_extension("OES_draw_buffers_indexed")
        .map_err(|_| "WebGL extension query failed")?
        .ok_or("OES_draw_buffers_indexed unavailable")?;
    let call = |name: &str, args: &[JsValue]| -> Result<(), &'static str> {
        let function = Reflect::get(&ext, &JsValue::from_str(name))
            .map_err(|_| "WebGL indexed blend method unavailable")?
            .dyn_into::<Function>()
            .map_err(|_| "WebGL indexed blend method unavailable")?;
        function
            .apply(&ext, &Array::from_iter(args.iter().cloned()))
            .map_err(|_| "WebGL indexed blend call failed")?;
        Ok(())
    };
    let n = |v: u32| JsValue::from_f64(f64::from(v));
    for (i, t) in targets.iter().enumerate() {
        let i = n(i as u32);
        call(
            "colorMaskiOES",
            &[
                i.clone(),
                JsValue::from_bool(t.write_mask & 1 != 0),
                JsValue::from_bool(t.write_mask & 2 != 0),
                JsValue::from_bool(t.write_mask & 4 != 0),
                JsValue::from_bool(t.write_mask & 8 != 0),
            ],
        )?;
        if let Some(b) = t.blend {
            call("enableiOES", &[n(Gl::BLEND), i.clone()])?;
            call(
                "blendFuncSeparateiOES",
                &[
                    i.clone(),
                    n(factor(b.color.src_factor)),
                    n(factor(b.color.dst_factor)),
                    n(factor(b.alpha.src_factor)),
                    n(factor(b.alpha.dst_factor)),
                ],
            )?;
            call(
                "blendEquationSeparateiOES",
                &[i, n(op(b.color.operation)), n(op(b.alpha.operation))],
            )?;
        } else {
            call("disableiOES", &[n(Gl::BLEND), i])?;
        }
    }
    Ok(())
}
