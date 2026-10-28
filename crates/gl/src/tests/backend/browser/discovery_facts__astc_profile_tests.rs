use super::*;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn hdr_requires_a_successful_profile_method() {
    let object = Object::new();
    let method = js_sys::Function::new_no_args("return ['ldr', 'hdr'];");
    Reflect::set(&object, &JsValue::from_str("getSupportedProfiles"), &method).unwrap();
    assert!(astc_hdr_profile(&object));
    let method = js_sys::Function::new_no_args("return ['ldr'];");
    Reflect::set(&object, &JsValue::from_str("getSupportedProfiles"), &method).unwrap();
    assert!(!astc_hdr_profile(&object));
    let method = js_sys::Function::new_no_args("throw new Error('lost context');");
    Reflect::set(&object, &JsValue::from_str("getSupportedProfiles"), &method).unwrap();
    assert!(!astc_hdr_profile(&object));
}
