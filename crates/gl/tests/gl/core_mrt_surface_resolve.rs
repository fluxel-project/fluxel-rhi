//! MRT resolves with a hole and a nonzero surface output slot.
use super::*;

async fn mrt_surface_resolve(d: &Device<Gl>) -> Option<Vec<u8>> {
    let mut surface = common::adopted_surface(d).unwrap();
    let format = TextureFormat::Rgba8Unorm;
    surface
        .configure(&SurfaceConfig {
            extent: [120, 64],
            format,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 1,
        })
        .await
        .unwrap();
    let image = surface.acquire().await.unwrap();
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [120, 64, 1],
        array_layers: 1,
        format,
        mip_levels: 1,
        samples: 4,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    };
    if !matches!(d.texture_support(&desc), Support::Supported(_)) {
        eprintln!("SKIP: no multisample RGBA8");
        return None;
    }
    let source0 = d.create_texture(&desc).unwrap();
    let source2 = d.create_texture(&desc).unwrap();
    let target = d
        .create_texture(&TextureDesc {
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            ..desc
        })
        .unwrap();
    let view = |texture| {
        d.create_view(
            texture,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap()
    };
    let view0 = view(&source0);
    let view2 = view(&source2);
    let target_view = view(&target);
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 456).unwrap();
    // SAFETY: surface lease and all textures live through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin MRT surface resolve"));
    let e = run.commands();
    let surface_base = [Some(ColorAttachment {
        view: AttachmentView::Surface(&image),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 1., 1.])),
        store: StoreOp::Store,
    })];
    e.begin_raster(&RasterPassDesc {
        label: None,
        colors: &surface_base,
        depth_stencil: None,
        render_area: Rect {
            x: 0,
            y: 0,
            width: 120,
            height: 64,
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    })
    .unwrap()
    .finish()
    .unwrap();
    let colors = [
        Some(ColorAttachment {
            view: AttachmentView::Texture(&view0),
            resolve_target: Some(AttachmentView::Texture(&target_view)),
            load: LoadOp::Clear(ClearColor::Float([0., 1., 0., 1.])),
            store: StoreOp::Store,
        }),
        None,
        Some(ColorAttachment {
            view: AttachmentView::Texture(&view2),
            resolve_target: Some(AttachmentView::Surface(&image)),
            load: LoadOp::Clear(ClearColor::Float([1., 0., 0., 1.])),
            store: StoreOp::Store,
        }),
    ];
    e.begin_raster(&RasterPassDesc {
        label: None,
        colors: &colors,
        depth_stencil: None,
        render_area: Rect {
            x: 20,
            y: 10,
            width: 30,
            height: 20,
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    })
    .unwrap()
    .finish()
    .unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &target,
            mip: 0,
            origin: [20, 10, 0],
            extent: [8, 8, 1],
            aspect: TextureAspects::COLOR,
        },
        &buffer,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 32,
            rows_per_image: 8,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&buffer, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish MRT surface resolve"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers the texture resolve and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    assert_eq!(&bytes[0..4], &[0, 255, 0, 255]);
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        // SAFETY: the lease and current context are valid for this read.
        let target = unsafe { image.native_ref() }.default_target().unwrap();
        // SAFETY: this device owns the active context on the current thread.
        let native = unsafe { d.native_ref() };
        let pixels = native
            .0
            .with_native("inspect MRT surface", |p| {
                p.read_default_framebuffer(target)
            })
            .unwrap()
            .bytes;
        let outside = (4 * 120 + 4) * 4;
        let inside = (15 * 120 + 25) * 4;
        assert_eq!(&pixels[outside..outside + 4], &[0, 0, 255, 255]);
        assert_eq!(&pixels[inside..inside + 4], &[255, 0, 0, 255]);
    }
    Some(bytes.to_vec())
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, Wake, Waker};
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_mrt_surface_resolve() {
    let window = fluxel_host::Window::new(
        fluxel_host::WindowConfig::new("MRT surface resolve", 120, 64).unwrap(),
    )
    .unwrap();
    // SAFETY: the window outlives its adopted context.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    block_on(mrt_surface_resolve(&block_on(open(native))));
}
#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES context"]
fn gles_core_mrt_surface_resolve() {
    use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
    let window = match fluxel_host::Window::new(
        fluxel_host::WindowConfig::new("GLES MRT surface resolve", 120, 64).unwrap(),
    ) {
        Ok(window) => window,
        Err(fluxel_host::WindowError::UnsupportedPlatform) => {
            eprintln!("SKIP: fluxel-host has no window implementation for this platform");
            return;
        }
        Err(error) => panic!("create EGL fixture window: {error:?}"),
    };
    let display = window.display_handle().unwrap().as_raw();
    let handle = window.window_handle().unwrap().as_raw();
    // SAFETY: the window and display outlive the adopted EGL context.
    let native = match unsafe { GlInstance::from_egl_window(display, handle) } {
        Ok(native) => native,
        #[cfg(windows)]
        Err(Error::Backend(message)) if message.contains("UnsupportedHandlePair") => {
            eprintln!("SKIP: Windows EGL window handles are unavailable: {message}");
            return;
        }
        Err(error) => panic!("create EGL window context: {error:?}"),
    };
    block_on(mrt_surface_resolve(&block_on(open(native))));
}
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_mrt_surface_resolve() {
    use wasm_bindgen::JsCast;
    use web_sys::WebGl2RenderingContext as GlRaw;
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(120);
    canvas.set_height(64);
    let device = open(GlInstance::from_webgl2_canvas(canvas.clone()).unwrap()).await;
    {
        assert!(
            mrt_surface_resolve(&device).await.is_some(),
            "WebGL2 RGBA8 multisample renderbuffer required"
        );
        let gl = canvas
            .get_context("webgl2")
            .unwrap()
            .unwrap()
            .dyn_into::<GlRaw>()
            .unwrap();
        gl.bind_framebuffer(GlRaw::FRAMEBUFFER, None);
        let mut pixel = [0; 4];
        gl.read_pixels_with_opt_u8_array(
            4,
            4,
            1,
            1,
            GlRaw::RGBA,
            GlRaw::UNSIGNED_BYTE,
            Some(&mut pixel),
        )
        .unwrap();
        assert_eq!(pixel, [0, 0, 255, 255]);
        let mut resolved = [0; 4];
        gl.read_pixels_with_opt_u8_array(
            25,
            64 - 1 - 15,
            1,
            1,
            GlRaw::RGBA,
            GlRaw::UNSIGNED_BYTE,
            Some(&mut resolved),
        )
        .unwrap();
        assert_eq!(resolved, [255, 0, 0, 255]);
        assert_eq!(gl.get_error(), GlRaw::NO_ERROR);
    }
}

/// A pbuffer has no Core Surface lease, but its default framebuffer lets this
/// test exercise the same native surface resolve helper after a Core MRT pass.
#[cfg(all(target_os = "linux", feature = "native-gles-egl"))]
#[test]
#[ignore = "requires an EGL GLES pbuffer"]
fn gles_core_mrt_resolve_pbuffer() {
    use crate::backend::api::{GlDefaultFramebufferTarget, GlFamilyApi as _, GlFormat};
    let d = block_on(open(GlInstance::gles_pbuffer([8, 8]).unwrap()));
    block_on(async {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 4,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        };
        if !matches!(d.texture_support(&desc), Support::Supported(_)) {
            eprintln!("SKIP: no multisample RGBA8");
            return;
        }
        let source0 = d.create_texture(&desc).unwrap();
        let source2 = d.create_texture(&desc).unwrap();
        let target = d
            .create_texture(&TextureDesc {
                samples: 1,
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
                ..desc
            })
            .unwrap();
        let view = |texture| {
            d.create_view(
                texture,
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
        let view0 = view(&source0);
        let view2 = view(&source2);
        let target_view = view(&target);
        let buffer = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 256,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let fixture = common::ImmediateFixture {
            device: &d,
            lane: d.lanes()[0].id,
        };
        let plan = common::single_commands_plan(&fixture, 457).unwrap();
        // SAFETY: all resources remain live through completion.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin pbuffer MRT"));
        let e = run.commands();
        let colors = [
            Some(ColorAttachment {
                view: AttachmentView::Texture(&view0),
                resolve_target: Some(AttachmentView::Texture(&target_view)),
                load: LoadOp::Clear(ClearColor::Float([0., 1., 0., 1.])),
                store: StoreOp::Store,
            }),
            None,
            Some(ColorAttachment {
                view: AttachmentView::Texture(&view2),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([1., 0., 0., 1.])),
                store: StoreOp::Store,
            }),
        ];
        e.begin_raster(&RasterPassDesc {
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
        .unwrap()
        .finish()
        .unwrap();
        e.texture_to_buffer(
            TextureRegion {
                texture: &target,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            &buffer,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
        let ticket = e.readback_buffer(&buffer, 0..256).unwrap();
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish pbuffer MRT"),
        };
        d.wait_completion(&point).await.unwrap();
        // SAFETY: completion covers texture readback and the view is retained.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        assert_eq!(&bytes[0..4], &[0, 255, 0, 255]);
        // SAFETY: the Core view and the same device remain alive on this thread.
        let source = unsafe { view2.native_ref() }.native;
        let native = unsafe { d.native_ref() };
        let pixels = native
            .0
            .with_native("resolve MRT to pbuffer", |provider| {
                let default = GlDefaultFramebufferTarget {
                    frame_serial: 1,
                    context: provider.context_stamp(),
                    width: 8,
                    height: 8,
                    sample_count: 1,
                    color_format: GlFormat::Rgba8Unorm,
                };
                crate::backend::rhi::raster::resolve::surface(
                    provider, source, default, None,
                )?;
                provider.read_default_framebuffer(default)
            })
            .unwrap()
            .bytes;
        assert_eq!(&pixels[0..4], &[255, 0, 0, 255]);
        eprintln!("Core MRT texture resolve and native pbuffer surface resolve passed");
    });
}
