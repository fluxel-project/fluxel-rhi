//! Browser GPU checks for attachment locations and dynamic raster state.
use super::WebGl2BrowserDiscovery;
use crate::backend::gl::api::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;
use web_sys::WebGl2RenderingContext as GL;

wasm_bindgen_test_configure!(run_in_browser);

fn open() -> WebGl2BrowserDiscovery {
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
    WebGl2BrowserDiscovery::open(ContextStamp::allocate().unwrap(), canvas).unwrap()
}

fn buffer(p: &mut WebGl2BrowserDiscovery, format: GlFormat) -> (RenderbufferId, GlTextureView) {
    let id = p
        .create_render_buffer(GlRenderBufferDesc {
            format,
            width: 8,
            height: 8,
            samples: 1,
        })
        .unwrap();
    (
        id,
        GlTextureView {
            aspect: crate::backend::gl::api::GlTextureAspect::All,
            target: GlAttachmentTarget::Renderbuffer(id),
            format,
            mip_level: 0,
            array_layer: 0,
            layer_count: 1,
            width: 8,
            height: 8,
            sample_count: 1,
        },
    )
}

fn color(view: GlTextureView) -> Option<GlColorAttachment> {
    Some(GlColorAttachment {
        view: GlPassAttachmentView::Allocated(view),
        resolve_target: None,
        load: GlLoadOp::Clear,
        store: GlStoreOp::Store,
        clear: GlColorClearValue::Float([0, 0, 0, 1f32.to_bits()]),
    })
}

fn pipeline(p: &mut WebGl2BrowserDiscovery, fragment: &str, count: usize) -> GlRasterPipeline {
    let source = |stage, hash, text: String| GlShaderSource {
        stage,
        dialect: GlShaderDialect::Embedded { version: 300 },
        entry_point: "main".into(),
        source_hash: GlShaderKey([hash; 32]),
        text,
        spirv: None,
        debug_name: None,
    };
    let (program, _) = p.create_program(&GlProgramDescriptor {
        kind: GlProgramKind::Raster {
            vertex: source(GlShaderStage::Vertex, 11, "#version 300 es\nvoid main(){vec2 p[3]=vec2[3](vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));gl_Position=vec4(p[gl_VertexID],0.,1.);}".into()),
            fragment: source(GlShaderStage::Fragment, 12, format!("#version 300 es\nprecision highp float; {fragment}")),
        }, layout: GlPipelineLayout { bindings: vec![] }, debug_name: None,
    }).unwrap();
    let vertex_array = p
        .create_vertex_array(&GlVertexLayout {
            buffers: vec![],
            attributes: vec![],
        })
        .unwrap();
    GlRasterPipeline {
        program,
        vertex_array,
        state: GlRasterState {
            rasterization: Default::default(),
            topology: GlPrimitiveTopology::Triangles,
            cull_mode: GlCullMode::None,
            front_face: GlFrontFace::CounterClockwise,
            depth_stencil: None,
            color_targets: vec![
                GlColorTargetState {
                    write_mask: 15,
                    blend: None
                };
                count
            ],
            multisample: GlMultisampleState {
                sample_count: 1,
                alpha_to_coverage_enabled: false,
                sample_mask: u64::MAX,
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

fn draw(p: &mut WebGl2BrowserDiscovery) {
    p.draw_raster(GlDrawCommand::NonIndexed(GlNonIndexedDraw {
        first_vertex: 0,
        vertex_count: 3,
        instance_count: 1,
    }))
    .unwrap();
}
fn pixel(p: &WebGl2BrowserDiscovery, attachment: u32) -> [u8; 4] {
    p.raw.read_buffer(GL::COLOR_ATTACHMENT0 + attachment);
    let mut value = [0; 4];
    p.raw
        .read_pixels_with_opt_u8_array(4, 4, 1, 1, GL::RGBA, GL::UNSIGNED_BYTE, Some(&mut value))
        .unwrap();
    assert_eq!(p.raw.get_error(), GL::NO_ERROR);
    value
}
fn clean_pipeline(p: &mut WebGl2BrowserDiscovery, pipeline: GlRasterPipeline) {
    p.destroy_vertex_array(pipeline.vertex_array).unwrap();
    p.destroy_program(pipeline.program).unwrap();
}

#[wasm_bindgen_test]
fn mrt_location_hole_preserves_fragment_output_indices() {
    let mut p = open();
    let (a, av) = buffer(&mut p, GlFormat::Rgba8Unorm);
    let (b, bv) = buffer(&mut p, GlFormat::Rgba8Unorm);
    let framebuffer = p
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![Some(av), None, Some(bv)],
            depth_stencil_attachment: None,
            draw_buffers: vec![0, u32::MAX, 2],
        })
        .unwrap();
    let pipe = pipeline(
        &mut p,
        "layout(location=0) out vec4 a; layout(location=2) out vec4 b; void main(){a=vec4(1.,0.,0.,1.);b=vec4(0.,1.,0.,1.);}",
        3,
    );
    p.begin_render_pass(&GlRenderPassDescriptor {
        render_area: None,
        target: GlRenderTarget::Offscreen(framebuffer),
        color_attachments: vec![color(av), None, color(bv)],
        depth_stencil_attachment: None,
    })
    .unwrap();
    p.set_raster_pipeline(&pipe).unwrap();
    draw(&mut p);
    assert_eq!(pixel(&p, 0), [255, 0, 0, 255]);
    assert_eq!(pixel(&p, 2), [0, 255, 0, 255]);
    p.end_render_pass().unwrap();
    clean_pipeline(&mut p, pipe);
    p.destroy_framebuffer(framebuffer).unwrap();
    p.destroy_render_buffer(a).unwrap();
    p.destroy_render_buffer(b).unwrap();
}

#[wasm_bindgen_test]
fn dynamic_blend_and_stencil_reference_affect_draws() {
    let mut p = open();
    let (c, cv) = buffer(&mut p, GlFormat::Rgba8Unorm);
    let (d, dv) = buffer(&mut p, GlFormat::Depth24PlusStencil8);
    let framebuffer = p
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![Some(cv)],
            depth_stencil_attachment: Some(dv),
            draw_buffers: vec![0],
        })
        .unwrap();
    let mut pipe = pipeline(
        &mut p,
        "out vec4 color; void main(){color=vec4(1.,0.,0.,1.);}",
        1,
    );
    let stencil = GlStencilFaceState {
        compare: GlDepthCompareFunction::Equal,
        fail_op: GlStencilOperation::Keep,
        depth_fail_op: GlStencilOperation::Keep,
        pass_op: GlStencilOperation::Replace,
        read_mask: 255,
        write_mask: 255,
    };
    pipe.state.depth_stencil = Some(GlDepthStencilState {
        depth_write_enabled: false,
        depth_compare: GlDepthCompareFunction::Always,
        depth_bias: [0; 3],
        stencil_front: stencil,
        stencil_back: stencil,
        stencil_reference: 0,
    });
    let blend = GlBlendComponent {
        src_factor: GlBlendFactor::Constant,
        dst_factor: GlBlendFactor::Zero,
        operation: GlBlendOperation::Add,
    };
    pipe.state.color_targets[0].blend = Some(GlBlendState {
        color: blend,
        alpha: blend,
    });
    p.begin_render_pass(&GlRenderPassDescriptor {
        render_area: None,
        target: GlRenderTarget::Offscreen(framebuffer),
        color_attachments: vec![color(cv)],
        depth_stencil_attachment: Some(GlDepthStencilAttachment {
            view: dv,
            depth_load: GlLoadOp::Clear,
            depth_store: GlStoreOp::Store,
            stencil_load: GlLoadOp::Clear,
            stencil_store: GlStoreOp::Store,
            clear: GlDepthStencilClearValue {
                depth: 1f32.to_bits(),
                stencil: 5,
            },
        }),
    })
    .unwrap();
    p.set_raster_pipeline(&pipe).unwrap();
    p.set_blend_constant([0.5, 0.5, 0.5, 1.0]).unwrap();
    p.set_stencil_reference(4).unwrap();
    draw(&mut p);
    assert_eq!(pixel(&p, 0), [0, 0, 0, 255]);
    p.set_stencil_reference(5).unwrap();
    draw(&mut p);
    let actual = pixel(&p, 0);
    assert!((127..=128).contains(&actual[0]), "{actual:?}");
    assert_eq!(&actual[1..], &[0, 0, 255]);
    p.end_render_pass().unwrap();
    clean_pipeline(&mut p, pipe);
    p.destroy_framebuffer(framebuffer).unwrap();
    p.destroy_render_buffer(c).unwrap();
    p.destroy_render_buffer(d).unwrap();
}
