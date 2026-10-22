//! Native canvas configuration and usage validation on a real WebGPU device.
use super::*;
use crate::backend::js;
use js_sys::{Array, Object, Promise};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::HtmlCanvasElement;

fn obj() -> JsValue {
    Object::new().into()
}
fn set(value: &JsValue, key: &str, item: JsValue) {
    js::set(value, key, &item).unwrap();
}
fn num(value: u32) -> JsValue {
    JsValue::from_f64(f64::from(value))
}

#[wasm_bindgen_test(async)]
async fn gpu_canvas_configurations_and_native_usage() {
    let device = open().await;
    let canvas: HtmlCanvasElement = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into()
        .unwrap();
    canvas.set_width(4);
    canvas.set_height(4);
    let mut surface = WebGpu::surface_from_canvas(&device, canvas.clone()).unwrap();
    // SAFETY: the fixture uses the native device only to validate this typed device's canvas.
    let native = unsafe { device.native_ref() };
    let raw_device = &native.0.device;
    let queue = &native.0.queue;
    let bgra_storage = js::has_feature(raw_device, "bgra8unorm-storage");
    let flags = [
        SurfaceUsage::RENDER_ATTACHMENT,
        SurfaceUsage::COPY_SRC,
        SurfaceUsage::COPY_DST,
        SurfaceUsage::TEXTURE_BINDING,
        SurfaceUsage::STORAGE_BINDING,
        SurfaceUsage::RENDER_ATTACHMENT
            | SurfaceUsage::COPY_SRC
            | SurfaceUsage::COPY_DST
            | SurfaceUsage::TEXTURE_BINDING
            | SurfaceUsage::STORAGE_BINDING,
    ];
    for format in [
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba16Float,
    ] {
        for alpha_mode in [
            CompositeAlphaMode::Opaque,
            CompositeAlphaMode::Premultiplied,
        ] {
            for color_space in [ColorSpace::Srgb, ColorSpace::DisplayP3] {
                for flag in flags {
                    let config = SurfaceConfig {
                        extent: [4, 4],
                        format,
                        present_mode: PresentMode::Fifo,
                        alpha_mode,
                        color_space,
                        usage: flag,
                        maximum_frame_latency: 2,
                    };
                    if format == TextureFormat::Bgra8Unorm
                        && flag.0 & SurfaceUsage::STORAGE_BINDING.0 != 0
                        && !bgra_storage
                    {
                        assert!(matches!(
                            surface.configuration_support(&config),
                            Support::Unsupported(_)
                        ));
                        continue;
                    }
                    js::call1(
                        raw_device,
                        "pushErrorScope",
                        &JsValue::from_str("validation"),
                    )
                    .unwrap();
                    surface.configure(&config).await.unwrap();
                    let lease = surface.acquire().await.unwrap();
                    let context: JsValue = canvas.get_context("webgpu").unwrap().unwrap().into();
                    let texture = js::call0(&context, "getCurrentTexture").unwrap();
                    let view = js::call1(&texture, "createView", &obj()).unwrap();
                    if flag.0 & SurfaceUsage::RENDER_ATTACHMENT.0 != 0 {
                        let attachment = obj();
                        set(&attachment, "view", view.clone());
                        set(&attachment, "loadOp", JsValue::from_str("clear"));
                        set(&attachment, "storeOp", JsValue::from_str("store"));
                        let clear = obj();
                        for channel in ["r", "g", "b", "a"] {
                            set(&clear, channel, num(0));
                        }
                        set(&attachment, "clearValue", clear);
                        let colors = Array::new();
                        colors.push(&attachment);
                        let desc = obj();
                        set(&desc, "colorAttachments", colors.into());
                        let encoder =
                            js::call1(raw_device, "createCommandEncoder", &obj()).unwrap();
                        let pass = js::call1(&encoder, "beginRenderPass", &desc).unwrap();
                        js::call0(&pass, "end").unwrap();
                        let command = js::call0(&encoder, "finish").unwrap();
                        let commands = Array::new();
                        commands.push(&command);
                        js::call1(queue, "submit", &commands.into()).unwrap();
                        JsFuture::from(Promise::from(
                            js::call0(queue, "onSubmittedWorkDone").unwrap(),
                        ))
                        .await
                        .unwrap();
                    }
                    for binding_usage in
                        [SurfaceUsage::TEXTURE_BINDING, SurfaceUsage::STORAGE_BINDING]
                    {
                        if flag.0 & binding_usage.0 == 0 {
                            continue;
                        }
                        let layout_entry = obj();
                        set(&layout_entry, "binding", num(0));
                        set(&layout_entry, "visibility", num(4));
                        let subtype = obj();
                        if binding_usage == SurfaceUsage::STORAGE_BINDING {
                            set(
                                &subtype,
                                "format",
                                JsValue::from_str(
                                    crate::backend::resource::texture_format(format).unwrap(),
                                ),
                            );
                            set(&subtype, "access", JsValue::from_str("write-only"));
                            set(&layout_entry, "storageTexture", subtype);
                        } else {
                            set(&layout_entry, "texture", subtype);
                        }
                        let entries = Array::new();
                        entries.push(&layout_entry);
                        let layout_desc = obj();
                        set(&layout_desc, "entries", entries.into());
                        let layout =
                            js::call1(raw_device, "createBindGroupLayout", &layout_desc).unwrap();
                        let entry = obj();
                        set(&entry, "binding", num(0));
                        set(&entry, "resource", view.clone());
                        let entries = Array::new();
                        entries.push(&entry);
                        let group_desc = obj();
                        set(&group_desc, "layout", layout);
                        set(&group_desc, "entries", entries.into());
                        js::call1(raw_device, "createBindGroup", &group_desc).unwrap();
                    }
                    for copy_usage in [SurfaceUsage::COPY_SRC, SurfaceUsage::COPY_DST] {
                        if flag.0 & copy_usage.0 == 0 {
                            continue;
                        }
                        let buffer_desc = obj();
                        set(&buffer_desc, "size", num(1024));
                        set(&buffer_desc, "usage", num(12));
                        let buffer = js::call1(raw_device, "createBuffer", &buffer_desc).unwrap();
                        let encoder =
                            js::call1(raw_device, "createCommandEncoder", &obj()).unwrap();
                        let tex = obj();
                        set(&tex, "texture", texture.clone());
                        let data = obj();
                        set(&data, "buffer", buffer);
                        set(&data, "bytesPerRow", num(256));
                        set(&data, "rowsPerImage", num(4));
                        let extent = obj();
                        set(&extent, "width", num(4));
                        set(&extent, "height", num(4));
                        set(&extent, "depthOrArrayLayers", num(1));
                        let (method, args) = if copy_usage == SurfaceUsage::COPY_SRC {
                            ("copyTextureToBuffer", vec![tex, data, extent])
                        } else {
                            ("copyBufferToTexture", vec![data, tex, extent])
                        };
                        js::call(&encoder, method, &args).unwrap();
                        let command = js::call0(&encoder, "finish").unwrap();
                        let commands = Array::new();
                        commands.push(&command);
                        js::call1(queue, "submit", &commands.into()).unwrap();
                        JsFuture::from(Promise::from(
                            js::call0(queue, "onSubmittedWorkDone").unwrap(),
                        ))
                        .await
                        .unwrap();
                    }
                    let error = JsFuture::from(Promise::from(
                        js::call0(raw_device, "popErrorScope").unwrap(),
                    ))
                    .await
                    .unwrap();
                    assert!(
                        error.is_null() || error.is_undefined(),
                        "canvas {:?} {:?} {:?} usage {:?}: {}",
                        format,
                        alpha_mode,
                        color_space,
                        flag,
                        js::message(&error)
                    );
                    drop(lease);
                }
            }
        }
    }
}
