//! Annotation state is preserved when an active raster pass is reopened.
use super::*;
use js_sys::{Object, Reflect};
use std::{cell::RefCell, rc::Rc};
use wasm_bindgen::{JsValue, closure::Closure};

#[wasm_bindgen_test]
fn raster_split_restores_nested_debug_groups_in_order() {
    let calls = Rc::new(RefCell::new(Vec::<String>::new()));
    let observed = calls.clone();
    let push = Closure::<dyn FnMut(JsValue)>::new(move |label: JsValue| {
        observed
            .borrow_mut()
            .push(label.as_string().expect("debug group label"));
    });
    let receiver = Object::new();
    Reflect::set(
        &receiver,
        &JsValue::from_str("pushDebugGroup"),
        push.as_ref(),
    )
    .unwrap();
    let labels = vec![
        "render graph pass".to_owned(),
        "nested operation".to_owned(),
    ];

    crate::backend::command::restore_debug_groups(&labels, receiver.as_ref()).unwrap();

    assert_eq!(&*calls.borrow(), &["render graph pass", "nested operation"]);
}
