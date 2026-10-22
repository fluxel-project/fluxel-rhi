//! Independent native Vulkan producer for real external-handle interoperability.
#![cfg(all(windows, feature = "hardware-tests"))]
#![allow(dead_code)]

use ash::{Entry, vk};
use std::os::windows::io::{FromRawHandle, OwnedHandle};

const SIZE: u64 = 256;
const PATTERN: u32 = 0x4c58_5248;

/// Lives until the consumer has released and completed use of the allocation.
struct Producer {
    _entry: Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    family: u32,
    pool: vk::CommandPool,
    image: vk::Image,
    memory: vk::DeviceMemory,
    semaphore: vk::Semaphore,
    allocation_size: u64,
    memory_type_index: u32,
    image_offset: u64,
    dedicated: bool,
}

impl Producer {
    /// Creates a second VkDevice and exports a real NT memory and semaphore handle.
    /// Returns None only when the selected physical device lacks the required capabilities.
    fn start() -> Option<(Self, OwnedHandle, OwnedHandle)> {
        // SAFETY: Vulkan loader remains live in the returned producer.
        let entry = unsafe { Entry::load().expect("Vulkan loader") };
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let info = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: create info and its referenced app remain valid for this call.
        let instance = unsafe {
            entry
                .create_instance(&info, None)
                .expect("native producer instance")
        };
        // SAFETY: physical devices are queried from the live instance.
        let physical = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("physical devices")
        }[0];
        // SAFETY: physical belongs to instance.
        let extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical)
                .unwrap()
        };
        let supports = |name: &std::ffi::CStr| {
            extensions.iter().any(|p| {
                // SAFETY: Vulkan extension names are NUL terminated.
                unsafe { std::ffi::CStr::from_ptr(p.extension_name.as_ptr()) == name }
            })
        };
        if !supports(ash::khr::external_memory_win32::NAME)
            || !supports(ash::khr::external_semaphore_win32::NAME)
        {
            unsafe { instance.destroy_instance(None) };
            return None;
        }
        let usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let handle_type = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32;
        let mut external_info =
            vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(handle_type);
        let query = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .push_next(&mut external_info);
        let mut properties = vk::ExternalImageFormatProperties::default();
        let mut image_properties = vk::ImageFormatProperties2::default().push_next(&mut properties);
        let supported = unsafe {
            instance.get_physical_device_image_format_properties2(
                physical,
                &query,
                &mut image_properties,
            )
        }
        .is_ok();
        let sem_query = vk::PhysicalDeviceExternalSemaphoreInfo::default()
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32);
        let mut sem_properties = vk::ExternalSemaphoreProperties::default();
        // SAFETY: property output structs and physical device are valid.
        unsafe {
            instance.get_physical_device_external_semaphore_properties(
                physical,
                &sem_query,
                &mut sem_properties,
            );
        }
        let required =
            vk::ExternalMemoryFeatureFlags::EXPORTABLE | vk::ExternalMemoryFeatureFlags::IMPORTABLE;
        let sem_required = vk::ExternalSemaphoreFeatureFlags::EXPORTABLE
            | vk::ExternalSemaphoreFeatureFlags::IMPORTABLE;
        if !supported
            || !properties
                .external_memory_properties
                .external_memory_features
                .contains(required)
            || !sem_properties
                .external_semaphore_features
                .contains(sem_required)
        {
            unsafe { instance.destroy_instance(None) };
            return None;
        }
        // SAFETY: physical belongs to the live instance.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let family = families
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .unwrap() as u32;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let names = [
            ash::khr::external_memory_win32::NAME.as_ptr(),
            ash::khr::external_semaphore_win32::NAME.as_ptr(),
        ];
        let create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&names);
        // SAFETY: queue and extension descriptions outlive device creation.
        let device = unsafe {
            instance
                .create_device(physical, &create, None)
                .expect("native producer device")
        };
        let queue = unsafe { device.get_device_queue(family, 0) };
        let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle_type);
        let desc = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: 8,
                height: 8,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external);
        let image = unsafe { device.create_image(&desc, None).expect("exportable image") };
        let mut dedicated_req = vk::MemoryDedicatedRequirements::default();
        let mut requirements2 = vk::MemoryRequirements2::default().push_next(&mut dedicated_req);
        unsafe {
            device.get_image_memory_requirements2(
                &vk::ImageMemoryRequirementsInfo2::default().image(image),
                &mut requirements2,
            )
        };
        let requirements = requirements2.memory_requirements;
        let dedicated = dedicated_req.requires_dedicated_allocation == vk::TRUE
            || properties
                .external_memory_properties
                .external_memory_features
                .contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY);
        let image_offset = if dedicated { 0 } else { requirements.alignment };
        let allocation_size = requirements.size + image_offset;
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let memory_type_index = (0..memory_properties.memory_type_count)
            .find(|i| requirements.memory_type_bits & (1 << i) != 0)
            .unwrap();
        let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let mut alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(allocation_size)
            .memory_type_index(memory_type_index)
            .push_next(&mut export);
        if dedicated {
            alloc = alloc.push_next(&mut dedicated_info);
        }
        let memory = unsafe {
            device
                .allocate_memory(&alloc, None)
                .expect("exportable memory")
        };
        unsafe {
            device
                .bind_image_memory(image, memory, image_offset)
                .unwrap()
        };
        let mut export_sem = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32);
        let semaphore = unsafe {
            device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut export_sem),
                    None,
                )
                .unwrap()
        };
        let memory_api = ash::khr::external_memory_win32::Device::new(&instance, &device);
        let sem_api = ash::khr::external_semaphore_win32::Device::new(&instance, &device);
        let memory_info = vk::MemoryGetWin32HandleInfoKHR::default()
            .memory(memory)
            .handle_type(handle_type);
        let sem_info = vk::SemaphoreGetWin32HandleInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32);
        // SAFETY: these are newly exported owned NT handles; OwnedHandle closes each once.
        let memory_handle = unsafe {
            OwnedHandle::from_raw_handle(
                memory_api.get_memory_win32_handle(&memory_info).unwrap() as *mut _
            )
        };
        let sem_handle = unsafe {
            OwnedHandle::from_raw_handle(
                sem_api.get_semaphore_win32_handle(&sem_info).unwrap() as *mut _
            )
        };
        let pool = unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                    None,
                )
                .unwrap()
        };
        let commands = unsafe {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()
        };
        let cmd = commands[0];
        unsafe {
            device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let initialize = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[initialize],
            );
            device.cmd_clear_color_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.2, 0.4, 0.6, 1.0],
                },
                &[range],
            );
            let release = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(family)
                .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                .image(image)
                .subresource_range(range);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[release],
            );
            device.end_command_buffer(cmd).unwrap();
            let signals = [semaphore];
            let submit = vk::SubmitInfo::default()
                .command_buffers(&commands)
                .signal_semaphores(&signals);
            device
                .queue_submit(queue, &[submit], vk::Fence::null())
                .unwrap();
        }
        Some((
            Self {
                _entry: entry,
                instance,
                device,
                queue,
                family,
                pool,
                image,
                memory,
                semaphore,
                allocation_size,
                memory_type_index,
                image_offset,
                dedicated,
            },
            memory_handle,
            sem_handle,
        ))
    }
}

impl Drop for Producer {
    fn drop(&mut self) {
        // SAFETY: consumer must complete before Producer is dropped; all objects belong to this device.
        unsafe {
            self.device.queue_wait_idle(self.queue).unwrap();
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_semaphore(self.semaphore, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[test]
fn win32_external_texture_import_and_readback() {
    use fluxel_rhi_core::api::*;
    use fluxel_rhi_vulkan::{VulkanExternalHandle, VulkanExternalMemoryHandle, VulkanInstance};
    use std::{future::Future, sync::Arc};
    fn block<F: Future>(f: F) -> F::Output {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        let mut f = std::pin::pin!(f);
        loop {
            if let std::task::Poll::Ready(value) = f.as_mut().poll(&mut cx) {
                return value;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    let Some((producer, memory_handle, semaphore_handle)) = Producer::start() else {
        eprintln!("external image or semaphore handles unavailable; skipped");
        return;
    };
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let external = device.external().expect("external handle support");
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [8, 8, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    // SAFETY: the opaque handle and metadata are exported by a live device on the same physical GPU with matching image parameters.
    let imported = unsafe {
        external.import_texture(
            VulkanExternalHandle::Memory(VulkanExternalMemoryHandle {
                handle: memory_handle,
                allocation_size: producer.allocation_size,
                memory_type_index: producer.memory_type_index,
                dedicated: producer.dedicated,
                image_offset: producer.image_offset,
            }),
            &desc,
        )
    }
    .unwrap();
    // SAFETY: producer releases image ownership and signals this semaphore.
    let sync =
        unsafe { external.import_sync(VulkanExternalHandle::Semaphore(semaphore_handle)) }.unwrap();
    let texture = external.texture(&imported).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: SIZE,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    let acquire = ExternalTextureTransition {
        aspects: TextureAspects::COLOR,
        mips: 0..1,
        layers: 0..1,
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            },
            to: TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        },
        layout: (ImageLayout::CopyDst, ImageLayout::CopySrc),
    };
    external
        .acquire_texture(&mut encoder, &imported, sync, &acquire)
        .unwrap();
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            &output,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
    let ticket = encoder.readback_buffer(&output, 0..SIZE).unwrap();
    let release = ExternalTextureTransition {
        aspects: TextureAspects::COLOR,
        mips: 0..1,
        layers: 0..1,
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
            to: TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        },
        layout: (ImageLayout::CopySrc, ImageLayout::CopySrc),
    };
    let _released = external
        .release_texture(&mut encoder, &imported, &release)
        .unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 774011,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    let receipt = match unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![encoder.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    } {
        SubmitOutcome::Accepted(r) => r,
        _ => panic!("external texture submit failed"),
    };
    let point = &receipt.checkpoints[0].point;
    block(device.wait_completion(point)).unwrap();
    let bytes = unsafe { block(device.read_after(ticket, point)).unwrap().bytes };
    for pixel in bytes.chunks_exact(4) {
        assert_eq!(pixel, &[51, 102, 153, 255]);
    }
    eprintln!(
        "external texture producer -> consumer readback passed (offset {})",
        producer.image_offset
    );
}
