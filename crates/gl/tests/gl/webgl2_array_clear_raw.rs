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
    // A one-pixel upload to each slice forces WebGL's subresource
    // initialization before the framebuffer clear is issued.
    for mip in 0..2 {
        for layer in 0..2 {
            let data: js_sys::Object = js_sys::Uint8Array::new_with_length(4).into();
            gl.tex_sub_image_3d_with_opt_array_buffer_view(
                Gl::TEXTURE_2D_ARRAY,
                mip,
                0,
                0,
                layer,
                1,
                1,
                1,
                Gl::RGBA,
                Gl::UNSIGNED_BYTE,
                Some(&data),
            )
            .unwrap();
            check("initialize");
        }
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
