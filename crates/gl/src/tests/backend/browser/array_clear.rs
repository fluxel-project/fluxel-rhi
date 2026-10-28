//! Raw WebGL2 array attachment clear/readback diagnostic.
use wasm_bindgen::JsCast;
use web_sys::WebGl2RenderingContext as Gl;

#[wasm_bindgen_test::wasm_bindgen_test]
fn webgl2_raw_initialized_array_mip_clear() {
    let canvas: web_sys::HtmlCanvasElement = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into()
        .unwrap();
    let gl: Gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap();
    let check = |step: &str| assert_eq!(gl.get_error(), Gl::NO_ERROR, "{step}");
    let texture = gl.create_texture().unwrap();
    gl.bind_texture(Gl::TEXTURE_2D_ARRAY, Some(&texture));
    gl.tex_storage_3d(Gl::TEXTURE_2D_ARRAY, 2, Gl::RGBA8, 4, 4, 2);
    check("storage");
    // ANGLE/D3D11 can lose a first attachment clear for an untouched array
    // slice despite immutable-storage initialization rules. Touch every layer
    // of each mip with one 1x1 zero column, matching BrowserOwner's allocation
    // workaround without returning to one native call per layer.
    for mip in 0..2 {
        let layers = 2_i32;
        let zeros = js_sys::Uint8Array::new_with_length((layers * 4) as u32);
        gl.tex_sub_image_3d_with_opt_array_buffer_view(
            Gl::TEXTURE_2D_ARRAY,
            mip,
            0,
            0,
            0,
            1,
            1,
            layers,
            Gl::RGBA,
            Gl::UNSIGNED_BYTE,
            Some(&zeros),
        )
        .unwrap();
        check("batched initialization");
    }
    for mip in 0..2 {
        for layer in 0..2 {
            let fb = gl.create_framebuffer().unwrap();
            gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&fb));
            gl.framebuffer_texture_layer(
                Gl::FRAMEBUFFER,
                Gl::COLOR_ATTACHMENT0,
                Some(&texture),
                mip,
                layer,
            );
            check("attachment");
            assert_eq!(
                gl.check_framebuffer_status(Gl::FRAMEBUFFER),
                Gl::FRAMEBUFFER_COMPLETE
            );
            check("framebuffer status");
            gl.enable(Gl::SCISSOR_TEST);
            gl.scissor(0, 0, 4 >> mip, 4 >> mip);
            gl.color_mask(true, true, true, true);
            gl.clear_bufferfv_with_f32_array(Gl::COLOR, 0, &[0., 0., 1., 1.]);
            check("clear");
            let mut rgba = [0; 4];
            gl.read_pixels_with_opt_u8_array(
                0,
                0,
                1,
                1,
                Gl::RGBA,
                Gl::UNSIGNED_BYTE,
                Some(&mut rgba),
            )
            .unwrap();
            check("read");
            assert_eq!(rgba, [0, 0, 255, 255], "mip {mip}, layer {layer}");
            gl.bind_framebuffer(Gl::FRAMEBUFFER, None);
            gl.delete_framebuffer(Some(&fb));
            check("release");
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_rhi_array_zero_init_restores_unpack_state() {
    use crate::{Gl as BackendGl, GlInstance};
    use fluxel_rhi_core::api::{
        AdapterId, DeviceRequest, Instance, InstanceId, TextureDesc, TextureDimension,
        TextureFormat, TextureUsage,
    };

    let canvas: web_sys::HtmlCanvasElement = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into()
        .unwrap();
    canvas.set_width(8);
    canvas.set_height(8);
    // Keep this handle to the context adopted by `GlInstance`: setting and
    // reading the state through a second raw handle still addresses the same
    // BrowserOwner WebGL context.
    let raw: Gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap();
    let native = GlInstance::from_webgl2_canvas(canvas).unwrap();
    let instance_id = InstanceId::allocate().unwrap();
    // SAFETY: `native` owns the canvas context for the full fixture.
    let instance = unsafe { Instance::<BackendGl>::from_native(instance_id, native) };
    let device = instance
        .open(DeviceRequest {
            adapter: AdapterId {
                instance: instance_id,
                ordinal: 0,
            },
            required_features: vec![],
        })
        .await
        .unwrap();

    let pbo = raw.create_buffer().unwrap();
    raw.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, Some(&pbo));
    raw.buffer_data_with_i32(Gl::PIXEL_UNPACK_BUFFER, 256, Gl::STREAM_DRAW);
    let parameters = [
        (Gl::UNPACK_ALIGNMENT, 8),
        (Gl::UNPACK_ROW_LENGTH, 7),
        (Gl::UNPACK_IMAGE_HEIGHT, 9),
        (Gl::UNPACK_SKIP_PIXELS, 1),
        (Gl::UNPACK_SKIP_ROWS, 2),
        (Gl::UNPACK_SKIP_IMAGES, 1),
    ];
    for (parameter, value) in parameters {
        raw.pixel_storei(parameter, value);
    }

    let assert_unpack = |label: &str| {
        let bound = raw
            .get_parameter(Gl::PIXEL_UNPACK_BUFFER_BINDING)
            .unwrap()
            .dyn_into::<web_sys::WebGlBuffer>()
            .unwrap();
        assert_eq!(bound, pbo, "{label}: PBO binding");
        for (parameter, expected) in parameters {
            let actual = raw.get_parameter(parameter).unwrap().as_f64().unwrap() as i32;
            assert_eq!(actual, expected, "{label}: pixel-store {parameter:#x}");
        }
        assert_eq!(raw.get_error(), Gl::NO_ERROR, "{label}: raw error");
    };
    let texture = |dimension, extent, array_layers| {
        device
            .create_texture(&TextureDesc {
                label: None,
                dimension,
                extent,
                array_layers,
                format: TextureFormat::Rgba8Unorm,
                mip_levels: 2,
                samples: 1,
                usage: TextureUsage::COPY_SRC
                    | TextureUsage::COPY_DST
                    | TextureUsage::TEXTURE_BINDING,
                view_formats: vec![],
            })
            .unwrap()
    };
    let array = texture(TextureDimension::D2, [4, 4, 1], 3);
    assert_unpack("D2 array allocation");
    let volume = texture(TextureDimension::D3, [4, 4, 3], 1);
    assert_unpack("D3 allocation");

    drop((array, volume));
    raw.bind_buffer(Gl::PIXEL_UNPACK_BUFFER, None);
    raw.delete_buffer(Some(&pbo));
}
