//! WGL entry-point evidence for deferred native GL publication.
//!
//! The test wraps the actual `glFlush` and `glGetError` loader entries before
//! constructing `NativeGlProvider`.  It therefore counts driver calls, rather
//! than mirroring provider bookkeeping.  Bootstrap/discovery calls are reset
//! before the checkpoint assertions.

use super::*;

use crate::backend::{
    api::{ContextStamp, GlFenceStatus, GlSyncApi as _},
    native::{context::NativeContextOwner as _, wgl::WglContextSurface},
};
use fluxel_host::{Window, WindowConfig};
use raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};
use std::{cell::Cell, ffi::c_void, mem::transmute};

thread_local! {
    static REAL_FLUSH: Cell<*const c_void> = const { Cell::new(core::ptr::null()) };
    static REAL_GET_ERROR: Cell<*const c_void> = const { Cell::new(core::ptr::null()) };
    static FLUSH_CALLS: Cell<u32> = const { Cell::new(0) };
    static GET_ERROR_CALLS: Cell<u32> = const { Cell::new(0) };
    static REAL_GET_INTEGERV: Cell<*const c_void> = const { Cell::new(core::ptr::null()) };
    static REAL_GET_PROGRAM_IV: Cell<*const c_void> = const { Cell::new(core::ptr::null()) };
    static REAL_GET_UNIFORM_LOCATION: Cell<*const c_void> = const { Cell::new(core::ptr::null()) };
    static REAL_CHECK_FRAMEBUFFER_STATUS: Cell<*const c_void> = const { Cell::new(core::ptr::null()) };
    static REAL_FINISH: Cell<*const c_void> = const { Cell::new(core::ptr::null()) };
    static GET_INTEGERV_CALLS: Cell<u32> = const { Cell::new(0) };
    static GET_PROGRAM_IV_CALLS: Cell<u32> = const { Cell::new(0) };
    static GET_UNIFORM_LOCATION_CALLS: Cell<u32> = const { Cell::new(0) };
    static CHECK_FRAMEBUFFER_STATUS_CALLS: Cell<u32> = const { Cell::new(0) };
    static FINISH_CALLS: Cell<u32> = const { Cell::new(0) };
    static FORCED_GET_ERROR: Cell<u32> = const { Cell::new(glow::NO_ERROR) };
}

/// Preserves the GL `void glFlush(void)` ABI while observing the real dispatch.
unsafe extern "system" fn counted_flush() {
    FLUSH_CALLS.with(|calls| calls.set(calls.get() + 1));
    let raw = REAL_FLUSH.with(Cell::get);
    assert!(
        !raw.is_null(),
        "glFlush wrapper was called before loader setup"
    );
    // SAFETY: the loader saved the exact current WGL context's core entry
    // point and the GL ABI for `glFlush` is `void(void)`.
    unsafe { transmute::<*const c_void, unsafe extern "system" fn()>(raw)() };
}

/// Preserves the GL `GLenum glGetError(void)` ABI while observing dispatch.
unsafe extern "system" fn counted_get_error() -> u32 {
    GET_ERROR_CALLS.with(|calls| calls.set(calls.get() + 1));
    let raw = REAL_GET_ERROR.with(Cell::get);
    assert!(
        !raw.is_null(),
        "glGetError wrapper was called before loader setup"
    );
    // SAFETY: the loader saved the exact current WGL context's core entry
    // point and the GL ABI for `glGetError` is `GLenum(void)`.
    let real = unsafe { transmute::<*const c_void, unsafe extern "system" fn() -> u32>(raw)() };
    let forced = FORCED_GET_ERROR.with(|error| error.replace(glow::NO_ERROR));
    if forced == glow::NO_ERROR {
        real
    } else {
        forced
    }
}

fn inject_next_get_error(error: u32) {
    FORCED_GET_ERROR.with(|next| next.set(error));
}

/// Counts the general state getter used by glow while preserving its GL ABI.
unsafe extern "system" fn counted_get_integerv(pname: u32, value: *mut i32) {
    GET_INTEGERV_CALLS.with(|calls| calls.set(calls.get() + 1));
    let raw = REAL_GET_INTEGERV.with(Cell::get);
    assert!(
        !raw.is_null(),
        "glGetIntegerv wrapper was called before loader setup"
    );
    // SAFETY: the WGL loader supplied the exact GL ABI entry for this context.
    unsafe {
        transmute::<*const c_void, unsafe extern "system" fn(u32, *mut i32)>(raw)(pname, value)
    }
}

/// Counts linked-program metadata observations without changing their result.
unsafe extern "system" fn counted_get_program_iv(program: u32, pname: u32, value: *mut i32) {
    GET_PROGRAM_IV_CALLS.with(|calls| calls.set(calls.get() + 1));
    let raw = REAL_GET_PROGRAM_IV.with(Cell::get);
    assert!(
        !raw.is_null(),
        "glGetProgramiv wrapper was called before loader setup"
    );
    // SAFETY: the WGL loader supplied the exact GL ABI entry for this context.
    unsafe {
        transmute::<*const c_void, unsafe extern "system" fn(u32, u32, *mut i32)>(raw)(
            program, pname, value,
        )
    }
}

/// Counts uniform-location reflection calls without changing their result.
unsafe extern "system" fn counted_get_uniform_location(program: u32, name: *const i8) -> i32 {
    GET_UNIFORM_LOCATION_CALLS.with(|calls| calls.set(calls.get() + 1));
    let raw = REAL_GET_UNIFORM_LOCATION.with(Cell::get);
    assert!(
        !raw.is_null(),
        "glGetUniformLocation wrapper was called before loader setup"
    );
    // SAFETY: the WGL loader supplied the exact GL ABI entry for this context.
    unsafe {
        transmute::<*const c_void, unsafe extern "system" fn(u32, *const i8) -> i32>(raw)(
            program, name,
        )
    }
}

/// Counts FBO completeness checks without changing their result.
unsafe extern "system" fn counted_check_framebuffer_status(target: u32) -> u32 {
    CHECK_FRAMEBUFFER_STATUS_CALLS.with(|calls| calls.set(calls.get() + 1));
    let raw = REAL_CHECK_FRAMEBUFFER_STATUS.with(Cell::get);
    assert!(
        !raw.is_null(),
        "glCheckFramebufferStatus wrapper was called before loader setup"
    );
    // SAFETY: the WGL loader supplied the exact GL ABI entry for this context.
    unsafe { transmute::<*const c_void, unsafe extern "system" fn(u32) -> u32>(raw)(target) }
}

/// Counts forbidden full-GPU waits while preserving the no-argument GL ABI.
unsafe extern "system" fn counted_finish() {
    FINISH_CALLS.with(|calls| calls.set(calls.get() + 1));
    let raw = REAL_FINISH.with(Cell::get);
    assert!(
        !raw.is_null(),
        "glFinish wrapper was called before loader setup"
    );
    // SAFETY: the WGL loader supplied the exact GL ABI entry for this context.
    unsafe { transmute::<*const c_void, unsafe extern "system" fn()>(raw)() }
}

fn reset_counts() {
    FLUSH_CALLS.with(|calls| calls.set(0));
    GET_ERROR_CALLS.with(|calls| calls.set(0));
    GET_INTEGERV_CALLS.with(|calls| calls.set(0));
    GET_PROGRAM_IV_CALLS.with(|calls| calls.set(0));
    GET_UNIFORM_LOCATION_CALLS.with(|calls| calls.set(0));
    CHECK_FRAMEBUFFER_STATUS_CALLS.with(|calls| calls.set(0));
    FINISH_CALLS.with(|calls| calls.set(0));
}

fn counts() -> (u32, u32) {
    (FLUSH_CALLS.with(Cell::get), GET_ERROR_CALLS.with(Cell::get))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CallCounts {
    get_integerv: u32,
    get_program_iv: u32,
    get_uniform_location: u32,
    check_framebuffer_status: u32,
    get_error: u32,
    flush: u32,
    finish: u32,
}

fn detailed_counts() -> CallCounts {
    CallCounts {
        get_integerv: GET_INTEGERV_CALLS.with(Cell::get),
        get_program_iv: GET_PROGRAM_IV_CALLS.with(Cell::get),
        get_uniform_location: GET_UNIFORM_LOCATION_CALLS.with(Cell::get),
        check_framebuffer_status: CHECK_FRAMEBUFFER_STATUS_CALLS.with(Cell::get),
        get_error: GET_ERROR_CALLS.with(Cell::get),
        flush: FLUSH_CALLS.with(Cell::get),
        finish: FINISH_CALLS.with(Cell::get),
    }
}

fn counted_glow(context: &WglContextSurface) -> glow::Context {
    // SAFETY: the caller holds the current WGL context until the returned
    // dispatch and the provider using it are dropped.
    unsafe {
        glow::Context::from_loader_function(|name| {
            let original = context.gl_proc(name);
            match name {
                "glFlush" => {
                    REAL_FLUSH.with(|slot| slot.set(original));
                    counted_flush as *const c_void
                }
                "glGetError" => {
                    REAL_GET_ERROR.with(|slot| slot.set(original));
                    counted_get_error as *const c_void
                }
                "glGetIntegerv" => {
                    REAL_GET_INTEGERV.with(|slot| slot.set(original));
                    counted_get_integerv as *const c_void
                }
                "glGetProgramiv" => {
                    REAL_GET_PROGRAM_IV.with(|slot| slot.set(original));
                    counted_get_program_iv as *const c_void
                }
                "glGetUniformLocation" => {
                    REAL_GET_UNIFORM_LOCATION.with(|slot| slot.set(original));
                    counted_get_uniform_location as *const c_void
                }
                "glCheckFramebufferStatus" => {
                    REAL_CHECK_FRAMEBUFFER_STATUS.with(|slot| slot.set(original));
                    counted_check_framebuffer_status as *const c_void
                }
                "glFinish" => {
                    REAL_FINISH.with(|slot| slot.set(original));
                    counted_finish as *const c_void
                }
                _ => original,
            }
        })
    }
}

#[test]
#[ignore = "requires an explicit WGL hardware-test run"]
fn wgl_checkpoints_publish_once_at_first_observation() {
    let window =
        Window::new(WindowConfig::new("GL publication entry counts", 32, 32).unwrap()).unwrap();
    let stamp = ContextStamp::allocate().unwrap();
    let mut context = WglContextSurface::open(
        stamp,
        window.window_handle().unwrap(),
        window.display_handle().unwrap(),
        [32, 32],
    )
    .unwrap();
    context
        .make_current("construct counted GL dispatch")
        .unwrap();
    let unmap = context.load_unmap_buffer().unwrap();

    // SAFETY: this fixture owns the exact current WGL context until provider
    // drop. The loader delegates every entry except the two ABI-preserving
    // wrappers above to that context's real WGL resolver.
    let glow = counted_glow(&context);
    // SAFETY: `context.make_current` above established the same current WGL
    // context used by this loader; it remains current and exclusively owned.
    let mut provider =
        unsafe { NativeGlProvider::from_current(glow, stamp, unmap, |name| context.gl_proc(name)) }
            .unwrap();

    reset_counts();
    let first = provider.create_fence().unwrap();
    let second = provider.create_fence().unwrap();
    assert_eq!(
        counts(),
        (0, 0),
        "checkpoint creation must only insert fences"
    );

    let first_status = provider.poll_fence(first).unwrap();
    assert!(
        matches!(
            first_status,
            GlFenceStatus::Pending | GlFenceStatus::Complete
        ),
        "first observation is bounded and must produce a valid fence status"
    );
    assert_eq!(
        counts(),
        (1, 1),
        "first headless observation publishes and samples the deferred diagnostic once"
    );

    let _ = provider.poll_fence(second).unwrap();
    let _ = provider.poll_fence(first).unwrap();
    assert_eq!(
        counts(),
        (1, 1),
        "later zero-time polls without new work must not reflush or read errors"
    );
}

/// Uses the native provider's ordinary pass/pipeline/draw SPI, not raw glow
/// drawing. The loader counts actual WGL dispatch after all allocations and
/// program reflection have warmed. This isolates the steady reusable frame
/// budget from cold construction and the final explicit pixel readback.
#[test]
#[ignore = "requires an explicit WGL hardware-test run"]
fn wgl_warm_raster_frames_avoid_state_and_metadata_queries() {
    use crate::backend::api::*;
    use glow::HasContext as _;

    let window =
        Window::new(WindowConfig::new("GL warm frame entry counts", 4, 4).unwrap()).unwrap();
    let stamp = ContextStamp::allocate().unwrap();
    let mut context = WglContextSurface::open(
        stamp,
        window.window_handle().unwrap(),
        window.display_handle().unwrap(),
        [4, 4],
    )
    .unwrap();
    context
        .make_current("construct counted warm-frame dispatch")
        .unwrap();
    let unmap = context.load_unmap_buffer().unwrap();
    let glow = counted_glow(&context);
    // SAFETY: this fixture owns the current context and all provider objects
    // until the provider is dropped before its context and window.
    let mut provider =
        unsafe { NativeGlProvider::from_current(glow, stamp, unmap, |name| context.gl_proc(name)) }
            .unwrap();

    let color = provider
        .create_render_buffer(GlRenderBufferDesc {
            format: GlFormat::Rgba8Unorm,
            width: 4,
            height: 4,
            samples: 1,
        })
        .unwrap();
    let view = GlTextureView {
        aspect: GlTextureAspect::All,
        target: GlAttachmentTarget::Renderbuffer(color),
        format: GlFormat::Rgba8Unorm,
        mip_level: 0,
        array_layer: 0,
        layer_count: 1,
        width: 4,
        height: 4,
        sample_count: 1,
    };
    let framebuffer = provider
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![Some(view)],
            depth_stencil_attachment: None,
            draw_buffers: vec![0],
        })
        .unwrap();
    let shader = |stage, byte, text: &str| GlShaderSource {
        stage,
        dialect: GlShaderDialect::Desktop { version: 400 },
        entry_point: "main".into(),
        source_hash: GlShaderKey([byte; 32]),
        text: text.into(),
        spirv: None,
        debug_name: None,
    };
    let (program, _) = provider
        .create_program(&GlProgramDescriptor {
            kind: GlProgramKind::Raster {
                vertex: shader(
                    GlShaderStage::Vertex,
                    1,
                    "#version 400 core\nvoid main(){vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexID],0,1);}",
                ),
                fragment: shader(
                    GlShaderStage::Fragment,
                    2,
                    "#version 400 core\nout vec4 color; void main(){color=vec4(1,0,0,1);}",
                ),
            },
            layout: GlPipelineLayout { bindings: vec![] },
            debug_name: None,
        })
        .unwrap();
    let vertex_array = provider
        .create_vertex_array(&GlVertexLayout {
            buffers: vec![],
            attributes: vec![],
        })
        .unwrap();
    let pipeline = GlRasterPipeline {
        program,
        vertex_array,
        state: GlRasterState {
            rasterization: Default::default(),
            topology: GlPrimitiveTopology::Triangles,
            cull_mode: GlCullMode::None,
            front_face: GlFrontFace::CounterClockwise,
            depth_stencil: None,
            color_targets: vec![GlColorTargetState {
                write_mask: 15,
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
                width: 4,
                height: 4,
                min_depth: 0.0f32.to_bits(),
                max_depth: 1.0f32.to_bits(),
            },
            scissor: None,
            blend_constant: [0; 4],
        },
    };
    let pass = GlRenderPassDescriptor {
        render_area: None,
        target: GlRenderTarget::Offscreen(framebuffer),
        color_attachments: vec![Some(GlColorAttachment {
            view: GlPassAttachmentView::Allocated(view),
            resolve_target: None,
            load: GlLoadOp::Clear,
            store: GlStoreOp::Store,
            clear: GlColorClearValue::Float([0, 0, 0, 1.0f32.to_bits()]),
        })],
        depth_stencil_attachment: None,
    };
    let draw = GlDrawCommand::NonIndexed(GlNonIndexedDraw {
        first_vertex: 0,
        vertex_count: 3,
        instance_count: 1,
    });
    let issue_frame = |provider: &mut NativeGlProvider| {
        provider.begin_render_pass(&pass).unwrap();
        provider.set_raster_pipeline(&pipeline).unwrap();
        provider.draw_raster(draw).unwrap();
        provider.end_render_pass().unwrap();
        provider.flush().unwrap();
    };

    // Program linking/reflection, FBO construction, and the first state packet
    // are cold work. Do not count them as part of a reusable normal frame.
    issue_frame(&mut provider);
    reset_counts();
    for frame in 0..3 {
        issue_frame(&mut provider);
        let observed = detailed_counts();
        assert_eq!(
            observed,
            CallCounts {
                get_integerv: 0,
                get_program_iv: 0,
                get_uniform_location: 0,
                check_framebuffer_status: 0,
                get_error: 1,
                flush: 1,
                finish: 0,
            },
            "warm normal frame {frame} exceeded the GL query/publication budget",
        );
        reset_counts();
    }

    // The count window is closed before this explicit CPU readback. It proves
    // the same backend pass actually produced the expected color instead of
    // merely exercising no-op command bookkeeping.
    let mut pixel = [0_u8; 4];
    // SAFETY: the fixture owns the context; the framebuffer has a 4x4 RGBA8
    // renderbuffer attachment and `pixel` has room for one packed texel.
    unsafe {
        let raw = provider.framebuffers.get(&framebuffer).unwrap().raw;
        provider
            .gl
            .bind_framebuffer(glow::READ_FRAMEBUFFER, Some(raw));
        provider.gl.read_pixels(
            0,
            0,
            1,
            1,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelPackData::Slice(Some(&mut pixel)),
        );
    }
    assert_eq!(pixel, [255, 0, 0, 255]);
}

/// Verifies the desktop direct GPU texture-to-buffer route independently from
/// CPU readback. `glGetTextureSubImage` is the required transfer operation;
/// its stream diagnostic joins one later publication instead of executing
/// after each reusable transfer.
#[test]
#[ignore = "requires an explicit WGL hardware-test run"]
fn wgl_warm_texture_to_buffer_defers_error_observation_to_publication() {
    use crate::backend::api::*;

    let window =
        Window::new(WindowConfig::new("GL texture-to-buffer counts", 4, 4).unwrap()).unwrap();
    let stamp = ContextStamp::allocate().unwrap();
    let mut context = WglContextSurface::open(
        stamp,
        window.window_handle().unwrap(),
        window.display_handle().unwrap(),
        [4, 4],
    )
    .unwrap();
    context
        .make_current("construct counted texture-to-buffer dispatch")
        .unwrap();
    let unmap = context.load_unmap_buffer().unwrap();
    let glow = counted_glow(&context);
    // SAFETY: this fixture owns the current WGL context for the provider's
    // entire lifetime and retains every object until after the observation.
    let mut provider =
        unsafe { NativeGlProvider::from_current(glow, stamp, unmap, |name| context.gl_proc(name)) }
            .unwrap();
    let texture = provider
        .create_texture_resource(GlTextureDesc {
            dimension: GlTextureDimension::D2,
            extent: GlExtent3d {
                width: 1,
                height: 1,
                depth_or_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            format: GlFormat::Rgba8Unorm,
            usage: GlTextureUsage::COPY_SOURCE | GlTextureUsage::COPY_DESTINATION,
        })
        .unwrap();
    let buffer = provider
        .create_buffer_resource(GlBufferDesc {
            size: 4,
            usage: GlBufferUsage::COPY_DESTINATION | GlBufferUsage::COPY_SOURCE,
        })
        .unwrap();
    let source = GlTextureRegion {
        subresource: GlTextureSubresource {
            texture,
            aspect: GlTextureAspect::Color,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        },
        origin: [0, 0, 0],
        extent: GlExtent3d {
            width: 1,
            height: 1,
            depth_or_layers: 1,
        },
    };
    let destination = GlBufferRange {
        buffer,
        offset: 0,
        size: 4,
    };
    let layout = GlPixelLayout {
        format: GlPixelFormat::Rgba8,
        bytes_per_row: 4,
        rows_per_image: 1,
        offset: 0,
        alignment: 1,
        repack: GlRepackPolicy::Disallow,
    };
    provider
        .upload_texture(source, layout, &[11, 22, 33, 44])
        .unwrap();
    // Upload is a preceding command interval. Publish and consume its
    // diagnostic before measuring the reusable texture-to-buffer operation.
    provider.flush().unwrap();
    reset_counts();

    for transfer in 0..3 {
        provider
            .copy_texture_to_buffer(source, destination, layout)
            .unwrap();
        let observed = detailed_counts();
        assert_eq!(
            observed.get_error, 0,
            "warm texture-to-buffer transfer {transfer} must not observe its driver error locally",
        );
        assert_eq!(
            observed.flush, 0,
            "transfer {transfer} must not publish itself"
        );
        assert_eq!(
            observed.finish, 0,
            "transfer {transfer} must not wait for all GPU work"
        );
        reset_counts();
    }

    provider.flush().unwrap();
    assert_eq!(
        detailed_counts(),
        CallCounts {
            get_integerv: 0,
            get_program_iv: 0,
            get_uniform_location: 0,
            check_framebuffer_status: 0,
            get_error: 1,
            flush: 1,
            finish: 0,
        },
        "one explicit publication must consume all deferred transfer diagnostics once",
    );

    // Explicit CPU consumption is outside the count window. It proves the
    // transfer wrote the GPU destination buffer, rather than only returning
    // success from its validation path.
    assert_eq!(
        provider.read_buffer(destination).unwrap(),
        vec![11, 22, 33, 44]
    );

    // Moving observation out of the transfer must not hide a driver failure.
    provider
        .copy_texture_to_buffer(source, destination, layout)
        .unwrap();
    inject_next_get_error(glow::INVALID_OPERATION);
    let publication = provider.flush();
    assert!(
        matches!(
            publication,
            Err(GlError::Driver {
                operation: "texture-sub-image-to-buffer",
                ..
            }),
        ),
        "deferred transfer error was not reported at publication: {publication:?}",
    );
}
