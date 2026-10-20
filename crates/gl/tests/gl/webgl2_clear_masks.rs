//! Browser GPU regression for the private WebGL2 framebuffer provider.
use super::WebGl2BrowserDiscovery;
use crate::backend::gl::api::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;
use web_sys::WebGl2RenderingContext as GL;

wasm_bindgen_test_configure!(run_in_browser);

fn pipeline(provider: &mut WebGl2BrowserDiscovery, z: &str, green: bool) -> GlRasterPipeline {
    let source = |stage, hash, text: String| GlShaderSource {
        stage,
        dialect: GlShaderDialect::Embedded { version: 300 },
        entry_point: "main".into(),
        source_hash: GlShaderKey([hash; 32]),
        text,
        debug_name: None,
    };
    let (program, _) = provider.create_program(&GlProgramDescriptor {
        kind: GlProgramKind::Raster {
            vertex: source(GlShaderStage::Vertex, if green { 3 } else { 1 }, format!("#version 300 es\nvoid main(){{vec2 p[3]=vec2[3](vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));gl_Position=vec4(p[gl_VertexID],{z},1.);}}")),
            fragment: source(GlShaderStage::Fragment, if green { 4 } else { 2 }, format!("#version 300 es\nprecision highp float; out vec4 color; void main(){{color=vec4({}, {}, 0., 1.);}}", if green { "0." } else { "1." }, if green { "1." } else { "0." })),
        }, layout: GlPipelineLayout { bindings: vec![] }, debug_name: None,
    }).unwrap();
    let vertex_array = provider
        .create_vertex_array(&GlVertexLayout {
            buffers: vec![],
            attributes: vec![],
        })
        .unwrap();
    let stencil = GlStencilFaceState {
        compare: GlDepthCompareFunction::Always,
        fail_op: GlStencilOperation::Keep,
        depth_fail_op: GlStencilOperation::Keep,
        pass_op: GlStencilOperation::Keep,
        read_mask: 0,
        write_mask: 0,
    };
    GlRasterPipeline {
        program,
        vertex_array,
        state: GlRasterState {
            topology: GlPrimitiveTopology::Triangles,
            cull_mode: GlCullMode::None,
            front_face: GlFrontFace::CounterClockwise,
            depth_stencil: Some(GlDepthStencilState {
                depth_write_enabled: false,
                depth_compare: GlDepthCompareFunction::Less,
                depth_bias: [0; 3],
                stencil_front: stencil,
                stencil_back: stencil,
                stencil_reference: 0,
            }),
            color_targets: vec![GlColorTargetState {
                write_mask: 15,
                blend: None,
            }],
            multisample: GlMultisampleState {
                sample_count: 1,
                alpha_to_coverage_enabled: false,
                sample_mask: u32::MAX,
            },
            viewport: GlViewport {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
                min_depth: 0f32.to_bits(),
                max_depth: 1f32.to_bits(),
            },
            scissor: None,
            blend_constant: [0; 4],
        },
    }
}

#[wasm_bindgen_test]
fn clear_depth_ignores_old_mask_and_restores_pipeline() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(8);
    canvas.set_height(8);
    let mut provider =
        WebGl2BrowserDiscovery::open(ContextStamp::allocate().unwrap(), canvas).unwrap();
    let version = provider
        .raw
        .get_parameter(GL::VERSION)
        .unwrap()
        .as_string()
        .unwrap();
    let renderer = provider
        .raw
        .get_parameter(GL::RENDERER)
        .unwrap()
        .as_string()
        .unwrap();
    let unmasked = super::driver_identity::unmasked_identity(&provider.raw)
        .map(|(vendor, renderer)| format!("{vendor} / {renderer}"))
        .unwrap_or_else(|| "unavailable".into());
    web_sys::console::log_1(
        &format!(
            "WEBGL_VERSION={version}; WEBGL_RENDERER={renderer}; WEBGL_UNMASKED_RENDERER={unmasked}"
        )
        .into(),
    );
    let color = provider
        .create_render_buffer(GlRenderBufferDesc {
            format: GlFormat::Rgba8Unorm,
            width: 8,
            height: 8,
            samples: 1,
        })
        .unwrap();
    let depth = provider
        .create_render_buffer(GlRenderBufferDesc {
            format: GlFormat::Depth32Float,
            width: 8,
            height: 8,
            samples: 1,
        })
        .unwrap();
    let view = |target, format| GlTextureView {
        target,
        format,
        mip_level: 0,
        array_layer: 0,
        layer_count: 1,
        width: 8,
        height: 8,
        sample_count: 1,
    };
    let color_view = view(
        GlAttachmentTarget::Renderbuffer(color),
        GlFormat::Rgba8Unorm,
    );
    let depth_view = view(
        GlAttachmentTarget::Renderbuffer(depth),
        GlFormat::Depth32Float,
    );
    let framebuffer = provider
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![color_view],
            depth_stencil_attachment: Some(depth_view),
            draw_buffers: vec![0],
        })
        .unwrap();
    let red = pipeline(&mut provider, "0.", false);
    let green = pipeline(&mut provider, "0.3", true);
    let pass = |value: f32| GlRenderPassDescriptor {
        target: GlRenderTarget::Offscreen(framebuffer),
        color_attachments: vec![GlColorAttachment {
            view: GlPassAttachmentView::Allocated(color_view),
            resolve_target: None,
            load: GlLoadOp::Clear,
            store: GlStoreOp::Store,
            clear: GlColorClearValue {
                red: 0,
                green: 0,
                blue: 0,
                alpha: 1f32.to_bits(),
            },
        }],
        depth_stencil_attachment: Some(GlDepthStencilAttachment {
            view: depth_view,
            depth_load: GlLoadOp::Clear,
            depth_store: GlStoreOp::Store,
            stencil_load: GlLoadOp::Load,
            stencil_store: GlStoreOp::Store,
            clear: GlDepthStencilClearValue {
                depth: value.to_bits(),
                stencil: 0,
            },
        }),
    };
    provider.begin_render_pass(&pass(0.25)).unwrap();
    provider.set_raster_pipeline(&red).unwrap();
    provider.end_render_pass().unwrap();
    provider.begin_render_pass(&pass(0.75)).unwrap();
    provider.set_raster_pipeline(&red).unwrap();
    let draw = GlDrawCommand::NonIndexed(GlNonIndexedDraw {
        first_vertex: 0,
        vertex_count: 3,
        instance_count: 1,
    });
    provider.draw_raster(draw).unwrap();
    let read = |provider: &WebGl2BrowserDiscovery| {
        let mut pixel = [0; 4];
        provider
            .raw
            .read_pixels_with_opt_u8_array(
                4,
                4,
                1,
                1,
                GL::RGBA,
                GL::UNSIGNED_BYTE,
                Some(&mut pixel),
            )
            .unwrap();
        pixel
    };
    assert_eq!(read(&provider), [255, 0, 0, 255], "depth clear was masked");
    provider.set_raster_pipeline(&green).unwrap();
    provider.draw_raster(draw).unwrap();
    assert_eq!(
        read(&provider),
        [0, 255, 0, 255],
        "pipeline depth mask was not restored"
    );
    assert_eq!(provider.raw.get_error(), GL::NO_ERROR);
    provider.end_render_pass().unwrap();
    for pipeline in [red, green] {
        provider
            .destroy_vertex_array(pipeline.vertex_array)
            .unwrap();
        provider.destroy_program(pipeline.program).unwrap();
    }
    provider.destroy_framebuffer(framebuffer).unwrap();
    provider.destroy_render_buffer(depth).unwrap();
    provider.destroy_render_buffer(color).unwrap();
}
