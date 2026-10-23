//! Public Core immediate raster workload shared by WGL and WebGL2 fixtures.
//! Actual color readback checks both fragment output locations around a hole.
use crate::{Gl, GlInstance};
use fluxel_rhi_core::api::*;
use std::sync::Arc;
#[path = "core_attachment_subresource.rs"]
mod attachment_subresource;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_binding_1d.rs"]
mod binding_1d;
#[path = "core_binding_3d.rs"]
mod binding_3d;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_barrier.rs"]
mod binding_barrier;
#[path = "core_binding_cube.rs"]
mod binding_cube;
#[path = "core_binding_cube_array.rs"]
mod binding_cube_array;
#[path = "core_binding_cube_storage.rs"]
mod binding_cube_storage;
#[path = "core_binding_depth_load.rs"]
mod binding_depth_load;
#[path = "core_binding_direct_buffer_array.rs"]
mod binding_direct_buffer_array;
#[path = "core_binding_direct_sampler_array.rs"]
mod binding_direct_sampler_array;
#[path = "core_binding_dynamic_cube_shadow.rs"]
mod binding_dynamic_cube_shadow;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_image.rs"]
mod binding_dynamic_image;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_image_load.rs"]
mod binding_dynamic_image_load;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_image_mixed.rs"]
mod binding_dynamic_image_mixed;
#[path = "core_binding_dynamic_pairs.rs"]
mod binding_dynamic_pairs;
#[path = "core_binding_dynamic_sampled.rs"]
mod binding_dynamic_sampled;
#[path = "core_binding_dynamic_sampled_array.rs"]
mod binding_dynamic_sampled_array;
#[path = "core_binding_dynamic_sampled_depth.rs"]
mod binding_dynamic_sampled_depth;
#[path = "core_binding_dynamic_sampled_float.rs"]
mod binding_dynamic_sampled_float;
#[path = "core_binding_dynamic_sampled_volume.rs"]
mod binding_dynamic_sampled_volume;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_storage.rs"]
mod binding_dynamic_storage;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_storage_compare.rs"]
mod binding_dynamic_storage_compare;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_storage_signed.rs"]
mod binding_dynamic_storage_signed;
#[path = "core_binding_helper_pairs.rs"]
mod binding_helper_pairs;
#[path = "core_binding_integer_compare.rs"]
mod binding_integer_compare;
#[path = "core_binding_pairs.rs"]
mod binding_pairs;
#[path = "core_binding_partial_array.rs"]
mod binding_partial_array;
#[path = "core_binding_partial_cube.rs"]
mod binding_partial_cube;
#[path = "core_binding_partial_depth.rs"]
mod binding_partial_depth;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_partial_image.rs"]
mod binding_partial_image;
#[path = "core_binding_partial_integer.rs"]
mod binding_partial_integer;
#[path = "core_binding_partial_pairs.rs"]
mod binding_partial_pairs;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_partial_ssbo.rs"]
mod binding_partial_ssbo;
#[path = "core_binding_partial_ubo.rs"]
mod binding_partial_ubo;
#[path = "core_binding_sampling_ops.rs"]
mod binding_sampling_ops;
#[path = "core_binding_shader_validation.rs"]
mod binding_shader_validation;
#[path = "core_binding_shared_sampler.rs"]
mod binding_shared_sampler;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_ssbo.rs"]
mod binding_ssbo;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_stage_images.rs"]
mod binding_stage_images;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_static_image.rs"]
mod binding_static_image;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_static_storage.rs"]
mod binding_static_storage;
#[path = "core_binding_stencil_array.rs"]
mod binding_stencil_array;
#[path = "core_binding_stencil_load.rs"]
mod binding_stencil_load;
#[path = "core_binding_validation.rs"]
mod binding_validation;
#[path = "../common/mod.rs"]
mod common;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_compressed_3d.rs"]
mod compressed_3d;
#[path = "core_compressed_bc1_pbo.rs"]
mod compressed_bc1_pbo;
#[path = "core_compressed_etc2.rs"]
mod compressed_etc2;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_compressed_gather.rs"]
mod compressed_gather;
#[path = "core_compressed_mips.rs"]
mod compressed_mips;
#[path = "core_compressed_pbo.rs"]
mod compressed_pbo;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_compressed_readback.rs"]
mod compressed_readback;
#[path = "core_compressed_readback_matrix.rs"]
mod compressed_readback_matrix;
#[path = "core_compressed_rebind.rs"]
mod compressed_rebind;
#[path = "core_compressed_astc.rs"]
mod core_compressed_astc;
#[path = "core_compressed_bptc.rs"]
mod core_compressed_bptc;
#[path = "core_compressed_etc_matrix.rs"]
mod core_compressed_etc_matrix;
#[path = "core_compressed_rgtc.rs"]
mod core_compressed_rgtc;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_cube_edges.rs"]
mod cube_edges;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_cube_faces.rs"]
mod cube_faces;
#[path = "core_cube_mirror_mips.rs"]
mod cube_mirror_mips;
#[path = "core_depth16.rs"]
mod depth16;
#[path = "core_depth_blit.rs"]
mod depth_blit;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_diagnostic_msaa_array.rs"]
mod diagnostic_msaa_array;
#[path = "core_direct_offsets.rs"]
mod direct_offsets;
#[path = "core_dynamic_features.rs"]
mod dynamic;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_dynamic_msaa_array.rs"]
mod dynamic_msaa_array;
#[path = "core_fragment_origin.rs"]
mod fragment_origin;
#[cfg(feature = "native-gles-egl")]
#[path = "core_gles_timestamp.rs"]
mod gles_timestamp;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_indirect.rs"]
mod indirect;
#[path = "core_layer_blit.rs"]
mod layer_blit;
#[path = "core_mapping.rs"]
mod mapping;
#[path = "core_mrt_indexed.rs"]
mod mrt_indexed;
#[path = "core_mrt_resolve.rs"]
mod mrt_resolve;
#[path = "core_mrt_surface_resolve.rs"]
mod mrt_surface_resolve;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_msaa_array_sampling.rs"]
mod msaa_array_sampling;
#[path = "core_multiview.rs"]
mod multiview;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_native_interop.rs"]
mod native_interop;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_native_mixed_srgb.rs"]
mod native_mixed_srgb;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_native_srgb.rs"]
mod native_srgb;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_native_surface_features.rs"]
mod native_surface_features;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_native_view.rs"]
mod native_view;
#[path = "core_ordinary_clear.rs"]
mod ordinary_clear;
#[path = "core_ordinary_formats.rs"]
mod ordinary_formats;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_pbo_readback.rs"]
mod pbo_readback;
#[path = "core_pbo_upload.rs"]
mod pbo_upload;
#[path = "core_pipeline_override.rs"]
mod pipeline_override;
#[path = "core_pipeline_statistics.rs"]
mod pipeline_statistics;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_program_cache.rs"]
mod program_cache;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_push_compute.rs"]
mod push_compute;
#[path = "core_push_constants.rs"]
mod push_constants;
#[path = "core_push_matrix.rs"]
mod push_matrix;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_sample_mask.rs"]
mod sample_mask;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_sampler_border.rs"]
mod sampler_border;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_sampler_border_integer.rs"]
mod sampler_border_integer;
#[path = "core_stencil_blit.rs"]
mod stencil_blit;
#[cfg(feature = "native-gles-egl")]
#[path = "core_stencil_extract_mask.rs"]
mod stencil_extract_mask;
#[cfg(feature = "native-gles-egl")]
#[path = "core_stencil_extract_mrt.rs"]
mod stencil_extract_mrt;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_storage_3d.rs"]
mod storage_3d;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_storage_array.rs"]
mod storage_array;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_storage_formats.rs"]
mod storage_formats;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_surface_features.rs"]
mod surface_features;
#[path = "core_timestamp.rs"]
mod timestamp;
#[path = "core_transition_validation.rs"]
mod transition_validation;
#[path = "core_vertex_formats.rs"]
mod vertex_formats;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_pbo_readback.rs"]
mod webgl_pbo_readback;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_query.rs"]
mod webgl_query;

async fn open(native: GlInstance) -> Device<Gl> {
    let id = InstanceId::allocate().unwrap();
    // SAFETY: the adopted context and its canvas/window remain alive until the device is dropped.
    let instance = unsafe { Instance::<Gl>::from_native(id, native) };
    let device = instance
        .open(DeviceRequest {
            adapter: AdapterId {
                instance: id,
                ordinal: 0,
            },
            required_features: vec![],
        })
        .await
        .unwrap();
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;
        // SAFETY: the fixture owns this device on its context thread.
        let native = unsafe { device.native_ref() };
        native
            .0
            .with_native("fixture renderer identity", |provider| {
                let info = provider.discovery().context();
                eprintln!(
                    "GL_VERSION={}; GL_VENDOR={}; GL_RENDERER={}",
                    info.version(),
                    info.vendor(),
                    info.renderer()
                );
                Ok(())
            })
            .unwrap();
    }
    device
}

async fn mrt_hole(d: &Device<Gl>) {
    let source = b"struct O { @builtin(position) p: vec4f }; struct F { @location(0) a: vec4f, @location(2) b: vec4f }; @vertex fn vs(@builtin(vertex_index) i:u32)->O { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); var o:O; o.p=vec4f(p[i],0.5,1.); return o; } @fragment fn fs()->F { var o:F; o.a=vec4f(1.,0.,0.,1.); o.b=vec4f(0.,1.,0.,1.); return o; }";
    let shader = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source.as_slice()),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = d.shader(shader("vs", ShaderStage::Vertex)).await.unwrap();
    let fragment = d.shader(shader("fs", ShaderStage::Fragment)).await.unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let state = ColorTargetState {
        format: TextureFormat::Rgba8Unorm,
        blend: None,
        write_mask: 15,
    };
    let pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: vertex,
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment,
                entry: "fs".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![],
            color_targets: vec![Some(state.clone()), None, Some(state)],
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: u32::MAX as u64,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
    let texture = || {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap()
    };
    let a = texture();
    let b = texture();
    let view = |t: &Texture<Gl>| {
        d.create_view(
            t,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap()
    };
    let av = view(&a);
    let bv = view(&b);
    let attachment = |v| {
        Some(ColorAttachment {
            view: AttachmentView::Texture(v),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
            store: StoreOp::Store,
        })
    };
    let colors = [attachment(&av), None, attachment(&bv)];
    let output = || {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let ab = output();
    let bb = output();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 91).unwrap();
    // SAFETY: both textures, buffers, pipeline and adopted context outlive completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin immediate commands"));
    let e = run.commands();
    let mut pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    for (texture, out) in [(&a, &ab), (&b, &bb)] {
        e.texture_to_buffer(
            TextureRegion {
                texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            out,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
    }
    let at = e.readback_buffer(&ab, 0..256).unwrap();
    let bt = e.readback_buffer(&bb, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish immediate commands"),
    };
    d.wait_completion(&point).await.unwrap();
    for (ticket, expected) in [(at, [255, 0, 0, 255]), (bt, [0, 255, 0, 255])] {
        // SAFETY: this completion point covers the draw and both readbacks above.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        assert_eq!(&bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4], &expected);
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_mrt_hole() {
    use fluxel_host::{Window, WindowConfig};
    use std::{
        future::Future,
        task::{Context, Poll, Wake, Waker},
    };
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    fn block_on<F: Future>(f: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(mrt_hole(&d));
    block_on(dynamic::depth_stencil_and_dynamic(&d));
    block_on(native_surface_features::mixed_surface(&d));
    block_on(native_surface_features::integer_clear(&d));
    block_on(native_surface_features::msaa_surface(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_mrt_hole() {
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
    let native = GlInstance::from_webgl2_canvas(canvas.clone()).unwrap();
    let gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::WebGl2RenderingContext>()
        .unwrap();
    let version = gl
        .get_parameter(web_sys::WebGl2RenderingContext::VERSION)
        .unwrap();
    let renderer = gl
        .get_extension("WEBGL_debug_renderer_info")
        .unwrap()
        .map(|_| gl.get_parameter(0x9246).unwrap())
        .unwrap_or_else(|| {
            gl.get_parameter(web_sys::WebGl2RenderingContext::RENDERER)
                .unwrap()
        });
    web_sys::console::log_1(
        &format!("WEBGL_VERSION={version:?}; WEBGL_RENDERER={renderer:?}").into(),
    );
    let d = open(native).await;
    mrt_hole(&d).await;
    dynamic::depth_stencil_and_dynamic(&d).await;
    let gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::WebGl2RenderingContext>()
        .unwrap();
    assert_eq!(gl.get_error(), web_sys::WebGl2RenderingContext::NO_ERROR);
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_raster_features() {
    use std::{
        future::Future,
        task::{Context, Poll, Wake, Waker},
    };
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    fn block_on<F: Future>(f: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(mrt_hole(&d));
    block_on(dynamic::depth_stencil_and_dynamic(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_srgb_surface.rs"]
mod srgb_surface;

#[path = "core_zero_stride.rs"]
mod core_zero_stride;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_integer_copy.rs"]
mod webgl_integer_copy;

#[path = "core_polygon_offset_clamp.rs"]
mod core_polygon_offset_clamp;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_native_color_readback.rs"]
mod native_color_readback;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_native_integer_copy.rs"]
mod native_integer_copy;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_rgb9e5_readback.rs"]
mod rgb9e5_readback;

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_raster_modes.rs"]
mod raster_modes;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_raster_modes.rs"]
mod webgl_raster_modes;

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_format_msaa.rs"]
mod format_msaa;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_zero_stride.rs"]
mod webgl_zero_stride;

#[path = "core_dual_source_blend.rs"]
mod dual_source_blend;

#[path = "core_direct_dual_source_blend.rs"]
mod direct_dual_source_blend;

#[path = "core_clear_nonrenderable.rs"]
mod clear_nonrenderable;

#[path = "core_binding_static_pairs.rs"]
mod binding_static_pairs;

#[path = "core_binding_dynamic_diagonal.rs"]
mod binding_dynamic_diagonal;
#[path = "core_binding_dynamic_helper.rs"]
mod binding_dynamic_helper;
#[path = "core_binding_dynamic_implicit.rs"]
mod binding_dynamic_implicit;

#[path = "core_binding_partial_compare.rs"]
mod binding_partial_compare;

#[path = "core_binding_dynamic_query.rs"]
mod binding_dynamic_query;

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_image_query.rs"]
mod binding_dynamic_image_query;

#[path = "core_binding_dynamic_bias.rs"]
mod binding_dynamic_bias;

#[path = "core_binding_dynamic_gather.rs"]
mod binding_dynamic_gather;

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "core_msaa_array_formats.rs"]
mod msaa_array_formats;

#[path = "core_binding_dynamic_query_layers.rs"]
mod binding_dynamic_query_layers;

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_query_samples.rs"]
mod binding_dynamic_query_samples;

#[path = "core_binding_unit_switch.rs"]
mod binding_unit_switch;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_multisampled_sampled_reject.rs"]
mod webgl_multisampled_sampled_reject;

#[cfg(any(
    all(target_arch = "wasm32", feature = "webgl2"),
    all(feature = "native-gles-egl", not(target_arch = "wasm32"))
))]
#[path = "core_binding_cube_mirror.rs"]
mod binding_cube_mirror;

#[path = "core_binding_dynamic_ubo.rs"]
mod binding_dynamic_ubo;

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "core_binding_direct_image_array.rs"]
mod binding_direct_image_array;

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "core_binding_direct_image_units.rs"]
mod binding_direct_image_units;

#[cfg(feature = "native-gles-egl")]
#[path = "core_query_stencil_extract.rs"]
mod query_stencil_extract;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_stencil_transfer.rs"]
mod stencil_transfer;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_binding_runtime_prefix.rs"]
mod binding_runtime_prefix;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_binding_runtime_divergent_storage.rs"]
mod binding_runtime_divergent_storage;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_binding_runtime_image.rs"]
mod binding_runtime_image;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_binding_runtime_hole.rs"]
mod binding_runtime_hole;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_binding_runtime_uniform.rs"]
mod binding_runtime_uniform;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_packed_plane_copy.rs"]
mod packed_plane_copy;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_desktop_multiview.rs"]
mod desktop_multiview;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_direct_color_readback.rs"]
mod direct_color_readback;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_snorm_endpoint_readback.rs"]
mod snorm_endpoint_readback;
