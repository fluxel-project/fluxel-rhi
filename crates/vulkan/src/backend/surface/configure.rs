//! Swapchain replacement and retirement.
//!
//! Configuration owns the generation transition under `SurfaceState::inner`.
//! It never destroys a chain that may still be presented: an old presented
//! chain is retained until acquire supplies the required retirement evidence.

use super::*;

pub(super) fn configure(state: &SurfaceState, config: &api::SurfaceConfig) -> Result<()> {
    validate_config(state, config).map_err(Error::Unsupported)?;
    // SAFETY: configuration holds no queue resources yet and queries the live
    // physical device/surface pair retained by SurfaceState.
    let caps = unsafe {
        state
            .surface_loader
            .get_physical_device_surface_capabilities(state.device.0.physical, state.surface)
    }
    .map_err(vk_error)?;
    // SAFETY: configuration queries the same retained WSI pair as above.
    let formats = unsafe {
        state
            .surface_loader
            .get_physical_device_surface_formats(state.device.0.physical, state.surface)
    }
    .map_err(vk_error)?;
    let format = super::super::resource::format(config.format)
        .ok_or(Error::Unsupported("Vulkan surface format"))?;
    let surface_format = formats
        .into_iter()
        .find(|f| f.format == format && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .ok_or(Error::Unsupported("Vulkan surface format"))?;
    let extent = vk::Extent2D {
        width: config.extent[0],
        height: config.extent[1],
    };
    let mut inner = state.inner.lock().unwrap_or_else(|e| e.into_inner());
    if inner.acquired {
        return Err(Error::InvalidState);
    }
    let next_generation = inner.generation.checked_add(1).ok_or(Error::OutOfMemory)?;
    let old = inner.swapchain;
    let mut usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
    if std::env::var_os("FLUXEL_VULKAN_CAPTURE").is_some() {
        if !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_SRC)
        {
            return Err(Error::Unsupported(
                "Vulkan swapchain transfer source capture",
            ));
        }
        usage |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    let info = vk::SwapchainCreateInfoKHR::default()
        .surface(state.surface)
        .min_image_count(caps.min_image_count.max(2))
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(usage)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        // Keep the framebuffer axes aligned with the public RHI viewport
        // whenever the WSI permits identity. Applying the Android display's
        // current transform here rotates the rendered scene a second time on
        // devices whose native window and physical display orientations differ.
        .pre_transform(
            if caps
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                caps.current_transform
            },
        )
        .composite_alpha(
            super::composite_alpha(config.alpha_mode)
                .ok_or(Error::Unsupported("Vulkan composite alpha mode"))?,
        )
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(old);
    // SAFETY: `inner` excludes concurrent configure/acquire; old_swapchain is
    // either null or owned by this state, and all create-info handles share it.
    let created = unsafe { state.swapchain_loader.create_swapchain(&info, None) };
    if old != vk::SwapchainKHR::null() {
        // The old chain is retired by the create attempt even if creation
        // fails. It must never remain active or be passed as oldSwapchain
        // again. Keep its views and present waits until new presentation
        // provides safe retirement evidence.
        let mut waits: Vec<_> = inner.present_waits.drain(..).flatten().collect();
        waits.extend(inner.orphaned_waits.drain(..));
        let old_views = std::mem::take(&mut inner.views);
        let retired = RetiredSwapchain {
            chain: old,
            views: old_views,
            wait_semaphores: waits,
        };
        if inner.presented {
            // Without a present fence, keep this chain until presentation of
            // the new chain is proved finished by reacquiring its image.
            inner.retired.push(retired);
        } else {
            // A chain never handed to presentation only has queue work. The
            // cold resize path may wait for that work and reclaim it now.
            let idle = {
                let _queue = state
                    .device
                    .0
                    .queue_gate
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                // SAFETY: queue_gate serializes the externally synchronized
                // idle wait; this never-presented chain has no pending present.
                unsafe {
                    state
                        .device
                        .0
                        .device
                        .queue_wait_idle(state.device.0.graphics_queue)
                }
            };
            if idle.is_ok() {
                // SAFETY: successful queue idle proves this state-owned retired
                // chain, views, and semaphores have no remaining queue users.
                unsafe {
                    for view in retired.views {
                        state.device.0.device.destroy_image_view(view, None);
                    }
                    for semaphore in retired.wait_semaphores {
                        state.device.0.device.destroy_semaphore(semaphore, None);
                    }
                    state
                        .swapchain_loader
                        .destroy_swapchain(retired.chain, None);
                }
            } else {
                inner.retired.push(retired);
            }
        }
        inner.swapchain = vk::SwapchainKHR::null();
        inner.images.clear();
    }
    let chain = created.map_err(vk_error)?;
    // SAFETY: `chain` was just created by this SurfaceState and is not published
    // until all image views have been created.
    let images =
        unsafe { state.swapchain_loader.get_swapchain_images(chain) }.map_err(|error| {
            // SAFETY: image enumeration failed before publication, so this local
            // swapchain has no externally reachable users.
            unsafe { state.swapchain_loader.destroy_swapchain(chain, None) };
            vk_error(error)
        })?;
    let mut views = Vec::with_capacity(images.len());
    for image in &images {
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let view_info = vk::ImageViewCreateInfo::default()
            .image(*image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(range);
        // SAFETY: each image belongs to the new un-published chain and the view
        // description names its one color subresource on the retained device.
        match unsafe { state.device.0.device.create_image_view(&view_info, None) } {
            Ok(view) => views.push(view),
            Err(error) => {
                // SAFETY: failure occurs before publication; every accumulated
                // view and the chain are owned exclusively by this error path.
                unsafe {
                    for view in views {
                        state.device.0.device.destroy_image_view(view, None);
                    }
                    state.swapchain_loader.destroy_swapchain(chain, None);
                }
                return Err(vk_error(error));
            }
        }
    }
    inner.swapchain = chain;
    inner.images = images;
    inner.views = views;
    inner.present_waits = vec![None; inner.images.len()];
    inner.extent = extent;
    inner.format = format;
    inner.generation = next_generation;
    inner.abandoned = false;
    inner.presented = false;
    Ok(())
}
