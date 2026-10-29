//! Annotation state is preserved when an active raster pass is reopened.
use super::*;
use js_sys::{Array, Object, Reflect};
use std::{cell::RefCell, rc::Rc};
use wasm_bindgen::{JsValue, closure::Closure};

fn set(object: &Object, name: &str, value: &JsValue) {
    Reflect::set(object, &JsValue::from_str(name), value).unwrap();
}

fn descriptor(colors: &[(bool, &str)], depth: Option<(&str, &str, bool, bool)>) -> JsValue {
    let descriptor = Object::new();
    let attachments = Array::new();
    for &(present, store) in colors {
        if !present {
            attachments.push(&JsValue::NULL);
            continue;
        }
        let attachment = Object::new();
        set(&attachment, "storeOp", &JsValue::from_str(store));
        attachments.push(&attachment);
    }
    set(&descriptor, "colorAttachments", attachments.as_ref());
    set(&descriptor, "depthStencilAttachment", &JsValue::NULL);
    if let Some((depth_store, stencil_store, depth_read_only, stencil_read_only)) = depth {
        let attachment = Object::new();
        set(&attachment, "depthStoreOp", &JsValue::from_str(depth_store));
        set(
            &attachment,
            "stencilStoreOp",
            &JsValue::from_str(stencil_store),
        );
        set(
            &attachment,
            "depthReadOnly",
            &JsValue::from_bool(depth_read_only),
        );
        set(
            &attachment,
            "stencilReadOnly",
            &JsValue::from_bool(stencil_read_only),
        );
        set(&descriptor, "depthStencilAttachment", attachment.as_ref());
    }
    descriptor.into()
}

fn state(descriptor: JsValue) -> JsValue {
    let state = Object::new();
    set(&state, "descriptor", &descriptor);
    state.into()
}

#[wasm_bindgen_test]
fn raster_split_rejects_discarded_writable_attachments() {
    for descriptor in [
        descriptor(&[(true, "discard")], None),
        descriptor(&[], Some(("discard", "store", false, false))),
        descriptor(&[], Some(("store", "discard", false, false))),
    ] {
        assert!(matches!(
            crate::backend::pass_split::validate_split_stores(&state(descriptor)),
            Err(Error::Unsupported(
                "WebGPU internal pass split with discarded attachments"
            ))
        ));
    }
}

#[wasm_bindgen_test]
fn raster_split_allows_stored_attachments_and_read_only_aspects() {
    for descriptor in [
        descriptor(&[(true, "store")], None),
        descriptor(&[], Some(("discard", "discard", true, true))),
        descriptor(
            &[(false, "discard")],
            Some(("store", "store", false, false)),
        ),
    ] {
        crate::backend::pass_split::validate_split_stores(&state(descriptor)).unwrap();
    }
}

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
