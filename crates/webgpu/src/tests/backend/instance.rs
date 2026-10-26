use super::indirect_capabilities;

#[wasm_bindgen_test::wasm_bindgen_test]
fn indirect_commands_do_not_depend_on_first_instance_feature() {
    assert_eq!(indirect_capabilities(false), (true, false, true, true));
    assert_eq!(indirect_capabilities(true), (true, true, true, true));
}
