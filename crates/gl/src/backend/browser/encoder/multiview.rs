//! WebGL OVR_multiview2 attachment calls, kept at the browser boundary.
use js_sys::{Array, Function, Reflect};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::WebGl2RenderingContext as Gl;

pub(in crate::backend::browser) struct MultiviewExtension {
    object: js_sys::Object,
    attach: Function,
}

impl MultiviewExtension {
    pub(in crate::backend::browser) fn acquire(raw: &Gl) -> Option<Self> {
        let object = raw.get_extension("OVR_multiview2").ok().flatten()?;
        let attach = Reflect::get(
            &object,
            &JsValue::from_str("framebufferTextureMultiviewOVR"),
        )
        .ok()?
        .dyn_into()
        .ok()?;
        Some(Self { object, attach })
    }

    pub(in crate::backend::browser) fn attach(
        &self,
        attachment: u32,
        texture: &web_sys::WebGlTexture,
        mip: u32,
        base: u32,
        count: u32,
    ) -> Result<(), JsValue> {
        let args = Array::new();
        args.push(&JsValue::from_f64(f64::from(Gl::FRAMEBUFFER)));
        args.push(&JsValue::from_f64(f64::from(attachment)));
        args.push(texture.as_ref());
        args.push(&JsValue::from_f64(f64::from(mip)));
        args.push(&JsValue::from_f64(f64::from(base)));
        args.push(&JsValue::from_f64(f64::from(count)));
        self.attach.apply(&self.object, &args).map(|_| ())
    }
}
