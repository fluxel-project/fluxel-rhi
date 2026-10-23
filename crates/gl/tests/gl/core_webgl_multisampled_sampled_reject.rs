//! WebGL2 exposes multisampled renderbuffers, not sampled multisample textures.
use super::*;

#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_multisampled_sampled_rejected() {
    use wasm_bindgen::JsCast;
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(8);
    canvas.set_height(8);
    let d = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [2, 2, 1],
        array_layers: 2,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 4,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        view_formats: vec![],
    };
    assert!(
        !matches!(d.texture_support(&desc), Support::Supported(_)),
        "WebGL2 must not advertise sampled multisample array textures"
    );
    assert!(
        d.create_texture(&desc).is_err(),
        "WebGL2 must reject sampled multisample array creation"
    );
}
