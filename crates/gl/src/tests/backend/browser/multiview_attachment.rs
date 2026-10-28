//! OVR framebuffer attachment and view-index output on a browser GPU.
use super::{BrowserColorAttachment, BrowserOwner, WebGl2BrowserDiscovery};
use crate::backend::api::{
    ContextStamp, GlExtent3d, GlFormat, GlTextureDesc, GlTextureDimension, GlTextureUsage,
};
use fluxel_rhi_core::api::{ClearColor, DeviceId, LaneId, LoadOp, Rect, StoreOp};
use wasm_bindgen::JsCast;
use web_sys::WebGl2RenderingContext as Gl;

#[wasm_bindgen_test::wasm_bindgen_test]
fn webgl2_multiview_browser_attachment() {
    let canvas: web_sys::HtmlCanvasElement = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into()
        .unwrap();
    canvas.set_width(4);
    canvas.set_height(4);
    let gl: Gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap();
    if gl.get_extension("OVR_multiview2").unwrap().is_none() {
        web_sys::console::log_1(&"OVR_multiview2 unavailable".into());
        return;
    }
    let owner = BrowserOwner::new(
        WebGl2BrowserDiscovery::open(ContextStamp::allocate().unwrap(), canvas).unwrap(),
    );
    assert!(
        owner
            .discovery_snapshot()
            .capabilities()
            .supports(crate::backend::api::GlCapability::Multiview)
    );
    assert!(owner.discovery_snapshot().max_multiview_view_count() >= 2);
    let texture = owner
        .create_texture(GlTextureDesc {
            dimension: GlTextureDimension::D2Array,
            extent: GlExtent3d {
                width: 4,
                height: 4,
                depth_or_layers: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            format: GlFormat::Rgba8Unorm,
            usage: GlTextureUsage::RENDER_ATTACHMENT,
        })
        .unwrap();
    let mut encoder = owner
        .begin_encoder(LaneId {
            device: DeviceId::allocate().unwrap(),
            ordinal: 0,
        })
        .unwrap();
    let colors = [Some(BrowserColorAttachment {
        texture: Some(&texture),
        surface_format: None,
        mip_level: 0,
        layer: 1,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
        store: StoreOp::Store,
        resolve: None,
    })];
    let mismatched = [
        Some(BrowserColorAttachment {
            texture: Some(&texture),
            surface_format: None,
            mip_level: 0,
            layer: 0,
            load: LoadOp::Load,
            store: StoreOp::Store,
            resolve: None,
        }),
        Some(BrowserColorAttachment {
            texture: Some(&texture),
            surface_format: None,
            mip_level: 0,
            layer: 1,
            load: LoadOp::Load,
            store: StoreOp::Store,
            resolve: None,
        }),
    ];
    let area = Rect {
        x: 0,
        y: 0,
        width: 4,
        height: 4,
    };
    assert!(matches!(
        encoder.begin_multiview_attachments(&mismatched, None, [4, 4], area, 2),
        Err(fluxel_rhi_core::api::Error::Unsupported(_))
    ));
    assert!(matches!(
        encoder.begin_multiview_attachments(&colors, None, [4, 4], area, 4),
        Err(fluxel_rhi_core::api::Error::InvalidRange)
    ));
    encoder
        .begin_multiview_attachments(
            &colors,
            None,
            [4, 4],
            Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            2,
        )
        .unwrap();
    encoder.end_offscreen_raster().unwrap();
    let clear_fb = gl.create_framebuffer().unwrap();
    gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&clear_fb));
    for layer in 1..=2 {
        gl.framebuffer_texture_layer(
            Gl::FRAMEBUFFER,
            Gl::COLOR_ATTACHMENT0,
            texture.raw.as_ref(),
            0,
            layer,
        );
        let mut pixel = [0; 4];
        gl.read_pixels_with_opt_u8_array(1, 1, 1, 1, Gl::RGBA, Gl::UNSIGNED_BYTE, Some(&mut pixel))
            .unwrap();
        assert_eq!(gl.get_error(), Gl::NO_ERROR);
        assert_eq!(pixel, [0, 0, 0, 255], "clear layer {layer}");
    }
    gl.bind_framebuffer(Gl::FRAMEBUFFER, None);
    gl.delete_framebuffer(Some(&clear_fb));
    encoder
        .begin_multiview_attachments(
            &colors,
            None,
            [4, 4],
            Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            2,
        )
        .unwrap();
    let compile = |kind, source| {
        let s = gl.create_shader(kind).unwrap();
        gl.shader_source(&s, source);
        gl.compile_shader(&s);
        assert!(
            gl.get_shader_parameter(&s, Gl::COMPILE_STATUS)
                .as_bool()
                .unwrap_or(false),
            "{:?}",
            gl.get_shader_info_log(&s)
        );
        s
    };
    let vertex = compile(
        Gl::VERTEX_SHADER,
        "#version 300 es\n#extension GL_OVR_multiview2 : require\nlayout(num_views=2) in; flat out uint id; void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0.,1.);id=gl_ViewID_OVR;}",
    );
    let fragment = compile(
        Gl::FRAGMENT_SHADER,
        "#version 300 es\nprecision highp float; precision highp int; flat in uint id; out vec4 color; void main(){color=id==0u?vec4(1,0,0,1):vec4(0,1,0,1);}",
    );
    let program = gl.create_program().unwrap();
    gl.attach_shader(&program, &vertex);
    gl.attach_shader(&program, &fragment);
    gl.link_program(&program);
    assert!(
        gl.get_program_parameter(&program, Gl::LINK_STATUS)
            .as_bool()
            .unwrap_or(false),
        "{:?}",
        gl.get_program_info_log(&program)
    );
    let vao = gl.create_vertex_array().unwrap();
    gl.bind_vertex_array(Some(&vao));
    gl.use_program(Some(&program));
    gl.draw_arrays(Gl::TRIANGLES, 0, 3);
    assert_eq!(gl.get_error(), Gl::NO_ERROR);
    encoder.end_offscreen_raster().unwrap();
    let fb = gl.create_framebuffer().unwrap();
    gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&fb));
    for (layer, expected) in [
        (0, [0, 0, 0, 0]),
        (1, [255, 0, 0, 255]),
        (2, [0, 255, 0, 255]),
        (3, [0, 0, 0, 0]),
    ] {
        gl.framebuffer_texture_layer(
            Gl::FRAMEBUFFER,
            Gl::COLOR_ATTACHMENT0,
            texture.raw.as_ref(),
            0,
            layer,
        );
        assert_eq!(
            gl.check_framebuffer_status(Gl::FRAMEBUFFER),
            Gl::FRAMEBUFFER_COMPLETE
        );
        let mut pixel = [0; 4];
        gl.read_pixels_with_opt_u8_array(1, 1, 1, 1, Gl::RGBA, Gl::UNSIGNED_BYTE, Some(&mut pixel))
            .unwrap();
        assert_eq!(gl.get_error(), Gl::NO_ERROR);
        assert_eq!(pixel, expected, "layer {layer}");
    }
    gl.bind_framebuffer(Gl::FRAMEBUFFER, None);
    gl.delete_framebuffer(Some(&fb));
    gl.delete_vertex_array(Some(&vao));
    gl.delete_program(Some(&program));
    gl.delete_shader(Some(&vertex));
    gl.delete_shader(Some(&fragment));
}
