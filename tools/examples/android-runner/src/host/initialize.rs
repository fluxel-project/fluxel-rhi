//! Android window adoption and backend/scene initialization on the event-loop owner thread.
//! The retained window outlives every native surface and context borrow.

use super::*;

impl App {
    pub(super) fn init(&mut self, event: &ActiveEventLoop) -> Result<()> {
        let window = Arc::new(
            event
                .create_window(WindowAttributes::default().with_title("Fluxel RHI Android capture"))
                .map_err(|e| Error::Backend(e.to_string()))?,
        );
        let size = window.inner_size();
        let extent = [size.width.max(1), size.height.max(1)];
        #[cfg(feature = "vulkan")]
        let gpu = {
            let native = fluxel_rhi::vulkan::VulkanInstance::new()?;
            let core = native.instance()?;
            let adapter = core
                .enumerate()?
                .into_iter()
                .next()
                .ok_or(Error::Unsupported("Vulkan adapter"))?;
            eprintln!("VULKAN_ADAPTER={}", adapter.name);
            let device = utils::block_on(core.open(DeviceRequest {
                adapter: adapter.id,
                required_features: vec![],
            }))?;
            let wh = window
                .window_handle()
                .map_err(|e| Error::Backend(e.to_string()))?
                .as_raw();
            let dh = window
                .display_handle()
                .map_err(|e| Error::Backend(e.to_string()))?
                .as_raw();
            // SAFETY: the Arc<Window> in Gpu owns the Android ANativeWindow for the entire surface lifetime.
            let mut surface = unsafe {
                fluxel_rhi::vulkan::Vulkan::surface_from_raw_window_handle(&device, wh, dh)?
            };
            let caps = surface.capabilities()?;
            eprintln!(
                "VULKAN_SURFACE_CAPS formats={:?} alpha={:?} usages={:?} extent={:?}..{:?}",
                caps.formats, caps.alpha_modes, caps.usages, caps.min_extent, caps.max_extent
            );
            let format = *caps
                .formats
                .first()
                .ok_or(Error::Unsupported("surface format"))?;
            let alpha = *caps
                .alpha_modes
                .first()
                .ok_or(Error::Unsupported("surface alpha mode"))?;
            let config = SurfaceConfig {
                extent,
                format,
                present_mode: PresentMode::Fifo,
                alpha_mode: alpha,
                color_space: ColorSpace::Srgb,
                usage: SurfaceUsage::RENDER_ATTACHMENT,
                maximum_frame_latency: 2,
            };
            utils::block_on(surface.configure(&config))?;
            let lane = device
                .lanes()
                .iter()
                .find(|x| x.class == LaneClass::Graphics)
                .map(|x| x.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            #[cfg(all(
                not(feature = "demo02"),
                not(feature = "demo03"),
                not(feature = "demo04"),
                not(feature = "demo05"),
                not(feature = "demo06"),
                not(feature = "demo07"),
                not(feature = "demo08"),
                not(feature = "demo09"),
                not(feature = "demo10"),
                not(feature = "demo11"),
                not(feature = "demo12"),
                not(feature = "demo13"),
                not(feature = "demo14"),
                not(feature = "demo15")
            ))]
            let scene = utils::block_on(triangle::Triangle::new(&device, lane, format))?;
            #[cfg(feature = "demo02")]
            let (pixels, width, height) = texture_pixels()?;
            #[cfg(feature = "demo02")]
            let pixel_extent = [width, height];
            #[cfg(feature = "demo02")]
            let scene = utils::block_on(texture::TextureExample::new(
                &device,
                lane,
                format,
                texture::TexturePixels {
                    rgba: &pixels,
                    width: pixel_extent[0],
                    height: pixel_extent[1],
                },
                Some((TEXTURE_WGSL, TEXTURE_WGSL)),
            ))?;
            #[cfg(feature = "demo03")]
            let scene =
                utils::block_on(instancing::Instancing::new(&device, lane, format, extent))?;
            #[cfg(feature = "demo04")]
            let scene = utils::block_on(dynamic_uniform::DynamicUniforms::new(
                &device, lane, format, extent,
            ))?;
            #[cfg(feature = "demo05")]
            let scene = utils::block_on(push_constants::PushWorkload::new(
                &device, lane, format, extent,
            ))?;
            #[cfg(feature = "demo06")]
            let scene = utils::block_on(compute::Compute::new(&device, lane, format))?;
            #[cfg(feature = "demo07")]
            let scene = utils::block_on(offscreen::Offscreen::new(&device, lane, format, extent))?;
            #[cfg(feature = "demo08")]
            let scene = utils::block_on(msaa::Msaa::new(&device, lane, format, extent))?;
            #[cfg(feature = "demo09")]
            let scene =
                utils::block_on(texture_mipmap::Tunnel::new(&device, lane, format, extent))?;
            #[cfg(feature = "demo10")]
            let scene =
                utils::block_on(indirect_draw::Workload::new(&device, lane, format, extent))?;
            #[cfg(feature = "demo11")]
            let scene = if device.queries(QueryKind::Occlusion).is_some() {
                utils::block_on(occlusion_query::Workload::new(&device, lane, format))?
            } else {
                utils::block_on(occlusion_query::Workload::new_visual_fallback(
                    &device, lane, format,
                ))?
            };
            Gpu {
                #[cfg(feature = "demo15")]
                scene: {
                    eprintln!("RHI_MULTIVIEW_MODE=native");
                    utils::block_on(multiview::Multiview::new(&device, lane, format, extent))?
                },
                #[cfg(feature = "demo14")]
                scene: {
                    let fallback =
                        descriptor_indexing::require_descriptor_indexing(&device).is_err();
                    eprintln!(
                        "RHI_DESCRIPTOR_MODE={}",
                        if fallback {
                            "fixed-texture-array"
                        } else {
                            "native-indexing"
                        }
                    );
                    utils::block_on(descriptor_indexing::DescriptorIndexing::new(
                        &device, lane, format, extent, fallback,
                    ))?
                },
                #[cfg(feature = "demo13")]
                scene: utils::block_on(multithreaded_recording::Multithreaded::new(
                    &device, lane, format,
                ))?,
                #[cfg(feature = "demo12")]
                scene: utils::block_on(screenshot::Screenshot::new_with_surface_format(
                    &device, lane, extent, format,
                ))?,
                window: window.clone(),
                device,
                surface,
                #[cfg(not(any(
                    feature = "demo12",
                    feature = "demo13",
                    feature = "demo14",
                    feature = "demo15"
                )))]
                scene,
                #[cfg(feature = "demo03")]
                time: std::time::Instant::now(),
                #[cfg(feature = "demo04")]
                time: std::time::Instant::now(),
                #[cfg(feature = "demo07")]
                time: std::time::Instant::now(),
                #[cfg(feature = "demo08")]
                time: std::time::Instant::now(),
                #[cfg(feature = "demo10")]
                time: std::time::Instant::now(),
                #[cfg(feature = "demo02")]
                pixels,
                #[cfg(feature = "demo02")]
                pixel_extent,
                lane,
                extent,
                format,
                alpha,
            }
        };
        #[cfg(feature = "gles")]
        let gpu = {
            let wh = window
                .window_handle()
                .map_err(|e| Error::Backend(e.to_string()))?
                .as_raw();
            let dh = window
                .display_handle()
                .map_err(|e| Error::Backend(e.to_string()))?
                .as_raw();
            // SAFETY: Gpu retains the Winit window and all EGL/GL objects are created and used on this event-loop thread.
            let native = unsafe { fluxel_rhi::gl::GlInstance::from_egl_window(dh, wh)? };
            // SAFETY: the instance owns the adopted EGL context; the window is
            // retained by Gpu, and all uses stay on this Android event-loop thread.
            let core = unsafe {
                Instance::<fluxel_rhi::gl::Gl>::from_native(InstanceId::allocate()?, native)
            };
            let adapter = core
                .enumerate()?
                .into_iter()
                .next()
                .ok_or(Error::Unsupported("GLES adapter"))?;
            eprintln!("GLES_ADAPTER={}", adapter.name);
            let device = utils::block_on(core.open(DeviceRequest {
                adapter: adapter.id,
                required_features: vec![],
            }))?;
            let mut surface = fluxel_rhi::gl::Gl::surface(&device)?;
            let format = *surface
                .capabilities()?
                .formats
                .first()
                .ok_or(Error::Unsupported("surface format"))?;
            let config = SurfaceConfig {
                extent,
                format,
                present_mode: PresentMode::Fifo,
                alpha_mode: CompositeAlphaMode::Opaque,
                color_space: ColorSpace::Srgb,
                usage: SurfaceUsage::RENDER_ATTACHMENT,
                maximum_frame_latency: 1,
            };
            utils::block_on(surface.configure(&config))?;
            let lane = device
                .lanes()
                .iter()
                .find(|x| x.class == LaneClass::Graphics)
                .map(|x| x.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            #[cfg(all(
                not(feature = "demo02"),
                not(feature = "demo03"),
                not(feature = "demo04"),
                not(feature = "demo05"),
                not(feature = "demo06"),
                not(feature = "demo07"),
                not(feature = "demo08"),
                not(feature = "demo09"),
                not(feature = "demo10"),
                not(feature = "demo11"),
                not(feature = "demo12"),
                not(feature = "demo13"),
                not(feature = "demo14"),
                not(feature = "demo15")
            ))]
            let scene = utils::block_on(triangle::Triangle::new(&device, lane, format))?;
            #[cfg(feature = "demo02")]
            let (pixels, width, height) = texture_pixels()?;
            #[cfg(feature = "demo02")]
            let pixel_extent = [width, height];
            #[cfg(feature = "demo02")]
            let scene = utils::block_on(texture::TextureExample::new(
                &device,
                lane,
                format,
                texture::TexturePixels {
                    rgba: &pixels,
                    width: pixel_extent[0],
                    height: pixel_extent[1],
                },
                Some((TEXTURE_WGSL, TEXTURE_WGSL)),
            ))?;
            #[cfg(feature = "demo03")]
            let scene =
                utils::block_on(instancing::Instancing::new(&device, lane, format, extent))?;
            #[cfg(feature = "demo04")]
            let scene = utils::block_on(dynamic_uniform::DynamicUniforms::new(
                &device, lane, format, extent,
            ))?;
            #[cfg(feature = "demo05")]
            let scene = utils::block_on(push_constants::PushWorkload::new(
                &device, lane, format, extent,
            ))?;
            #[cfg(feature = "demo06")]
            let scene = utils::block_on(compute::Compute::new(&device, lane, format))?;
            #[cfg(feature = "demo07")]
            let scene = utils::block_on(offscreen::Offscreen::new(&device, lane, format, extent))?;
            #[cfg(feature = "demo08")]
            let scene =
                utils::block_on(msaa::Msaa::new_single_sample(&device, lane, format, extent))?;
            #[cfg(feature = "demo09")]
            let scene =
                utils::block_on(texture_mipmap::Tunnel::new(&device, lane, format, extent))?;
            #[cfg(feature = "demo10")]
            let scene =
                utils::block_on(indirect_draw::Workload::new(&device, lane, format, extent))?;
            #[cfg(feature = "demo11")]
            let scene = if device.queries(QueryKind::Occlusion).is_some() {
                utils::block_on(occlusion_query::Workload::new(&device, lane, format))?
            } else {
                utils::block_on(occlusion_query::Workload::new_visual_fallback(
                    &device, lane, format,
                ))?
            };
            Gpu {
                #[cfg(feature = "demo15")]
                scene: {
                    eprintln!("RHI_MULTIVIEW_MODE=sequential-stereo");
                    utils::block_on(multiview::per_eye::StereoFallback::new(
                        &device, format, extent,
                    ))?
                },
                #[cfg(feature = "demo14")]
                scene: {
                    eprintln!("RHI_DESCRIPTOR_MODE=fixed-texture-array");
                    utils::block_on(descriptor_indexing::DescriptorIndexing::new(
                        &device, lane, format, extent, true,
                    ))?
                },
                #[cfg(feature = "demo13")]
                scene: utils::block_on(multithreaded_recording::Sequential::new(&device, format))?,
                #[cfg(feature = "demo12")]
                scene: utils::block_on(screenshot::Screenshot::new_with_surface_format(
                    &device, lane, extent, format,
                ))?,
                window: window.clone(),
                device,
                surface,
                #[cfg(not(any(
                    feature = "demo12",
                    feature = "demo13",
                    feature = "demo14",
                    feature = "demo15"
                )))]
                scene,
                #[cfg(feature = "demo03")]
                time: std::time::Instant::now(),
                #[cfg(feature = "demo04")]
                time: std::time::Instant::now(),
                #[cfg(feature = "demo07")]
                time: std::time::Instant::now(),
                #[cfg(feature = "demo08")]
                time: std::time::Instant::now(),
                #[cfg(feature = "demo10")]
                time: std::time::Instant::now(),
                #[cfg(feature = "demo02")]
                pixels,
                #[cfg(feature = "demo02")]
                pixel_extent,
                lane,
                extent,
                format,
                alpha: CompositeAlphaMode::Opaque,
            }
        };
        self.gpu = Some(gpu);
        window.request_redraw();
        Ok(())
    }
}
