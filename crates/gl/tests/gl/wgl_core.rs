//! WGL hardware fixture for the typed immediate GL contract.
//!
//! This is opt-in because it creates a real Win32 window and executes native
//! GL calls. It verifies typed instance/device/surface construction and an
//! empty direct commands step and framebuffer clear/write-mask behavior.

#[path = "../common/mod.rs"]
mod common;

use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

use fluxel_host::{Window, WindowConfig};
use fluxel_rhi_core::api::{
    AdapterId, CompletionRequest, Device, DeviceRequest, ImmediateFinishOutcome,
    ImmediateRunOutcome, Instance, InstanceId,
};

use crate::{Gl, GlInstance};

/// Wakes the test owner thread when a future makes completion progress.
struct ThreadWake(std::thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

/// Drives Core's cold device-open future without introducing a test runtime.
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

/// Opens an adopted WGL context as the current typed Core device.
fn device(window: &Window) -> Device<Gl> {
    // SAFETY: `window` is declared before the returned device in the fixture,
    // so reverse drop order destroys every GL object before its HWND/handles.
    // The test opens and drops it on this same owner thread.
    let native = unsafe { GlInstance::from_wgl_window(window, [32, 32]) }
        .expect("open WGL context for typed hardware fixture");
    let instance_id = InstanceId::allocate().expect("allocate GL instance id");
    // SAFETY: `native` remains owned by this Core instance through `open`; the
    // returned device retains its adopted context state after the instance is
    // consumed by this fixture's normal drop order.
    let instance = unsafe { Instance::<Gl>::from_native(instance_id, native) };
    block_on(instance.open(DeviceRequest {
        adapter: AdapterId {
            instance: instance_id,
            ordinal: 0,
        },
        required_features: Vec::new(),
    }))
    .expect("open typed GL device from adopted WGL context")
}

#[test]
#[ignore = "requires an explicit WGL hardware-test run"]
fn wgl_typed_device_surface_and_empty_immediate_step() {
    let window = Window::new(WindowConfig::new("Fluxel RHI GL WGL fixture", 32, 32).unwrap())
        .expect("create WGL fixture window");
    let device = device(&window);
    let fixture = common::ImmediateFixture {
        device: &device,
        lane: device.lanes()[0].id,
    };

    let _surface = common::adopted_surface(&device).expect("adopt typed GL surface");
    let plan = common::single_commands_plan(&fixture, 1).expect("compile typed commands plan");
    // SAFETY: this fixture has one same-device graphics lane and retains the
    // device, surface, and window until `finish` returns its issued prefix.
    let mut session = unsafe { common::begin_commands(&fixture, plan, CompletionRequest::None) }
        .expect("begin GL immediate session");
    let run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin direct GL commands"))
        .close();
    assert!(matches!(run, ImmediateRunOutcome::Issued(_)));
    assert!(matches!(
        session.finish(),
        ImmediateFinishOutcome::Issued(_)
    ));
}

#[test]
#[ignore = "requires an explicit WGL hardware-test run"]
fn wgl_clear_ignores_and_restores_pipeline_write_masks() {
    use super::{provider::NativeGlProvider, wgl::WglContextSurface};
    use crate::backend::gl::api::*;
    use glow::HasContext as _;
    use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

    let window = Window::new(WindowConfig::new("Depth clear fixture", 32, 32).unwrap()).unwrap();
    let stamp = ContextStamp::allocate().unwrap();
    let mut context = WglContextSurface::open(
        stamp,
        window.window_handle().unwrap(),
        window.display_handle().unwrap(),
        [32, 32],
    )
    .unwrap();
    let discovery = context.discover().unwrap().clone();
    let unmap = context.load_unmap_buffer().unwrap();
    let gl = context.load_glow().unwrap();
    // SAFETY: this current WGL context owns the temporary objects, which are deleted before returning.
    unsafe {
        let texture = gl.create_texture().unwrap();
        let fbo = gl.create_framebuffer().unwrap();
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_storage_2d(glow::TEXTURE_2D, 1, glow::DEPTH24_STENCIL8, 32, 32);
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::DEPTH_STENCIL_ATTACHMENT,
            glow::TEXTURE_2D,
            Some(texture),
            0,
        );
        gl.draw_buffers(&[glow::NONE]);
        gl.read_buffer(glow::NONE);
        assert_eq!(
            gl.check_framebuffer_status(glow::FRAMEBUFFER),
            glow::FRAMEBUFFER_COMPLETE
        );
        assert_eq!(gl.get_error(), glow::NO_ERROR);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.delete_framebuffer(fbo);
        gl.delete_texture(texture);
    }
    // Production discovery now supplies this core format. Do not replace its
    // exact facts with a fixture-local row: that would bypass the public
    // format admission used by the raster route being exercised.
    assert!(
        discovery
            .formats()
            .get_for(
                GlFormatResourceKind::Texture,
                GlFormat::Depth24PlusStencil8,
                1,
            )
            .is_some_and(|facts| facts.renderable)
    );
    // SAFETY: the window and context outlive the provider, and the context stays current on this thread.
    let mut provider = unsafe { NativeGlProvider::from_discovered(gl, discovery, unmap) };
    let depth_desc = GlTextureDesc {
        dimension: GlTextureDimension::D2,
        extent: GlExtent3d {
            width: 32,
            height: 32,
            depth_or_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        format: GlFormat::Depth24PlusStencil8,
        usage: GlTextureUsage::RENDER_ATTACHMENT,
    };
    let depth = provider.create_texture_resource(depth_desc).unwrap();
    let color = provider
        .create_render_buffer(GlRenderBufferDesc {
            format: GlFormat::Rgba8Unorm,
            width: 32,
            height: 32,
            samples: 1,
        })
        .unwrap();
    let view = GlTextureView {
        aspect: crate::backend::gl::api::GlTextureAspect::All,
        target: GlAttachmentTarget::Texture(depth),
        format: GlFormat::Depth24PlusStencil8,
        mip_level: 0,
        array_layer: 0,
        layer_count: 1,
        width: 32,
        height: 32,
        sample_count: 1,
    };
    let color_view = GlTextureView {
        target: GlAttachmentTarget::Renderbuffer(color),
        format: GlFormat::Rgba8Unorm,
        ..view
    };
    let framebuffer = provider
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![Some(color_view)],
            depth_stencil_attachment: Some(view),
            draw_buffers: vec![0],
        })
        .unwrap();
    let shader = |stage, byte, source: &str| GlShaderSource {
        stage,
        dialect: GlShaderDialect::Desktop { version: 400 },
        entry_point: "main".into(),
        source_hash: GlShaderKey([byte; 32]),
        text: source.into(),
        debug_name: None,
    };
    let (program, _) = provider.create_program(&GlProgramDescriptor {
        kind: GlProgramKind::Raster {
            vertex: shader(GlShaderStage::Vertex, 1, "#version 400 core\nvoid main(){vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexID],0,1);}"),
            fragment: shader(GlShaderStage::Fragment, 2, "#version 400 core\nout vec4 color; void main(){color=vec4(1,0,0,1);}"),
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
        pass_op: GlStencilOperation::Replace,
        read_mask: u32::MAX,
        write_mask: 0,
    };
    let pipeline = GlRasterPipeline {
        program,
        vertex_array,
        state: GlRasterState {
            rasterization: Default::default(),
            topology: GlPrimitiveTopology::Triangles,
            cull_mode: GlCullMode::None,
            front_face: GlFrontFace::CounterClockwise,
            depth_stencil: Some(GlDepthStencilState {
                depth_write_enabled: false,
                depth_compare: GlDepthCompareFunction::Always,
                depth_bias: [0; 3],
                stencil_front: stencil,
                stencil_back: stencil,
                stencil_reference: 0x72,
            }),
            color_targets: vec![GlColorTargetState {
                write_mask: 0,
                blend: None,
            }],
            multisample: GlMultisampleState {
                sample_count: 1,
                alpha_to_coverage_enabled: false,
                sample_mask: u64::MAX,
            },
            viewport: GlViewport {
                x: 0,
                y: 0,
                width: 32,
                height: 32,
                min_depth: 0.0f32.to_bits(),
                max_depth: 1.0f32.to_bits(),
            },
            scissor: None,
            blend_constant: [0; 4],
        },
    };
    let pass = |value: f32| GlRenderPassDescriptor {
        render_area: None,
        target: GlRenderTarget::Offscreen(framebuffer),
        color_attachments: vec![Some(GlColorAttachment {
            view: GlPassAttachmentView::Allocated(color_view),
            resolve_target: None,
            load: GlLoadOp::Clear,
            store: GlStoreOp::Store,
            clear: GlColorClearValue::Float([
                0,
                0.5f32.to_bits(),
                1.0f32.to_bits(),
                1.0f32.to_bits(),
            ]),
        })],
        depth_stencil_attachment: Some(GlDepthStencilAttachment {
            view,
            depth_load: GlLoadOp::Clear,
            depth_store: GlStoreOp::Store,
            stencil_load: GlLoadOp::Clear,
            stencil_store: GlStoreOp::Store,
            clear: GlDepthStencilClearValue {
                depth: value.to_bits(),
                stencil: 0x39,
            },
        }),
    };
    let mut first = pass(0.25);
    first.color_attachments[0].as_mut().unwrap().clear =
        GlColorClearValue::Float([0, 0, 0, 1.0f32.to_bits()]);
    first
        .depth_stencil_attachment
        .as_mut()
        .unwrap()
        .clear
        .stencil = 0x12;
    provider.begin_render_pass(&first).unwrap();
    provider.set_raster_pipeline(&pipeline).unwrap();
    provider.end_render_pass().unwrap();
    provider.begin_render_pass(&pass(0.75)).unwrap();
    let mut depth_bytes = [0_u8; 4];
    // SAFETY: the pass framebuffer remains bound and the output has space for one depth float.
    unsafe {
        provider.gl.read_pixels(
            0,
            0,
            1,
            1,
            glow::DEPTH_COMPONENT,
            glow::FLOAT,
            glow::PixelPackData::Slice(Some(&mut depth_bytes)),
        )
    };
    assert!((f32::from_ne_bytes(depth_bytes) - 0.75).abs() < 0.0001);
    let read_color_stencil = |provider: &NativeGlProvider| {
        let mut color_bytes = [0_u8; 4];
        let mut stencil_bytes = [0_u8; 4];
        // SAFETY: the active pass framebuffer is bound and both buffers hold one pixel including pack alignment.
        unsafe {
            provider.gl.read_pixels(
                0,
                0,
                1,
                1,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut color_bytes)),
            );
            provider.gl.read_pixels(
                0,
                0,
                1,
                1,
                glow::STENCIL_INDEX,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut stencil_bytes)),
            );
        }
        (color_bytes, stencil_bytes[0])
    };
    let expected = ([0, 128, 255, 255], 0x39);
    assert_eq!(read_color_stencil(&provider), expected);
    provider.set_raster_pipeline(&pipeline).unwrap();
    provider
        .draw_raster(GlDrawCommand::NonIndexed(GlNonIndexedDraw {
            first_vertex: 0,
            vertex_count: 3,
            instance_count: 1,
        }))
        .unwrap();
    // SAFETY: the same framebuffer remains bound; a draw at depth 0.5 must not replace the clear when the pipeline mask is restored.
    unsafe {
        provider.gl.read_pixels(
            0,
            0,
            1,
            1,
            glow::DEPTH_COMPONENT,
            glow::FLOAT,
            glow::PixelPackData::Slice(Some(&mut depth_bytes)),
        )
    };
    assert!((f32::from_ne_bytes(depth_bytes) - 0.75).abs() < 0.0001);
    assert_eq!(read_color_stencil(&provider), expected);
    // SAFETY: the fixture owns the current context and queries its error flag.
    assert_eq!(unsafe { provider.gl.get_error() }, glow::NO_ERROR);
    provider.end_render_pass().unwrap();
    let mut stencil_only = pass(0.0);
    stencil_only.color_attachments[0].as_mut().unwrap().load = GlLoadOp::Load;
    let attachment = stencil_only.depth_stencil_attachment.as_mut().unwrap();
    attachment.depth_load = GlLoadOp::Load;
    attachment.clear.stencil = 0x56;
    provider.begin_render_pass(&stencil_only).unwrap();
    assert_eq!(read_color_stencil(&provider), (expected.0, 0x56));
    // SAFETY: the fixture owns the current context and queries the stencil-only clear's error flag.
    assert_eq!(unsafe { provider.gl.get_error() }, glow::NO_ERROR);
    provider.end_render_pass().unwrap();
    provider.destroy_vertex_array(vertex_array).unwrap();
    provider.destroy_program(program).unwrap();
    provider.destroy_framebuffer(framebuffer).unwrap();
    provider.destroy_texture_resource(depth).unwrap();
    provider.destroy_render_buffer(color).unwrap();
}
