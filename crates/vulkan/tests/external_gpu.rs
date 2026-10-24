//! Independent native Vulkan producer for real external-handle interoperability.
#![cfg(all(windows, feature = "hardware-tests"))]

use ash::{Entry, vk};
use std::os::windows::io::{FromRawHandle, OwnedHandle};

const SIZE: u64 = 256;
const PATTERN: u32 = 0x4c58_5248;
const RETURN_PATTERN: u32 = 0x5248_4932;

/// Lives until the consumer has released and completed use of the allocation.
struct Producer {
    _entry: Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    family: u32,
    pool: vk::CommandPool,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    semaphore: vk::Semaphore,
    allocation_size: u64,
    memory_type_index: u32,
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
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let handle_type = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32;
        let query = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(usage)
            .handle_type(handle_type);
        let mut properties = vk::ExternalBufferProperties::default();
        let sem_query = vk::PhysicalDeviceExternalSemaphoreInfo::default()
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32);
        let mut sem_properties = vk::ExternalSemaphoreProperties::default();
        // SAFETY: property output structs and physical device are valid.
        unsafe {
            instance.get_physical_device_external_buffer_properties(
                physical,
                &query,
                &mut properties,
            );
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
        if !properties
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
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
        let desc = vk::BufferCreateInfo::default()
            .size(SIZE)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        let buffer = unsafe {
            device
                .create_buffer(&desc, None)
                .expect("exportable buffer")
        };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let memory_type_index = (0..memory_properties.memory_type_count)
            .find(|i| requirements.memory_type_bits & (1 << i) != 0)
            .unwrap();
        let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut export)
            .push_next(&mut dedicated);
        let memory = unsafe {
            device
                .allocate_memory(&alloc, None)
                .expect("exportable memory")
        };
        unsafe { device.bind_buffer_memory(buffer, memory, 0).unwrap() };
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
            device.cmd_fill_buffer(cmd, buffer, 0, SIZE, PATTERN);
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .src_queue_family_index(family)
                .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                .buffer(buffer)
                .offset(0)
                .size(SIZE);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
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
                buffer,
                memory,
                semaphore,
                allocation_size: requirements.size,
                memory_type_index,
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
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Producer {
    /// Waits on the consumer's exported semaphore, reacquires queue ownership,
    /// and verifies the actual allocation using a fresh GPU-to-host copy.
    fn verify_return(&self, handle: OwnedHandle, expected: u32) {
        use std::os::windows::io::AsRawHandle;
        let sem_api = ash::khr::external_semaphore_win32::Device::new(&self.instance, &self.device);
        let semaphore = unsafe {
            self.device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap()
        };
        let import = vk::ImportSemaphoreWin32HandleInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32)
            .handle(handle.as_raw_handle() as isize);
        // SAFETY: the handle remains live for the duration of import; Win32 import does not consume it.
        unsafe { sem_api.import_semaphore_win32_handle(&import).unwrap() };
        let staging = unsafe {
            self.device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(SIZE)
                        .usage(vk::BufferUsageFlags::TRANSFER_DST)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap()
        };
        let req = unsafe { self.device.get_buffer_memory_requirements(staging) };
        let physical = unsafe { self.instance.enumerate_physical_devices().unwrap() }[0];
        let props = unsafe {
            self.instance
                .get_physical_device_memory_properties(physical)
        };
        let index = (0..props.memory_type_count)
            .find(|i| {
                req.memory_type_bits & (1 << i) != 0
                    && props.memory_types[*i as usize].property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .expect("host coherent staging memory");
        let memory = unsafe {
            self.device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(index),
                    None,
                )
                .unwrap()
        };
        unsafe { self.device.bind_buffer_memory(staging, memory, 0).unwrap() };
        let commands = unsafe {
            self.device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()
        };
        unsafe {
            self.device
                .begin_command_buffer(commands[0], &vk::CommandBufferBeginInfo::default())
                .unwrap();
            let acquire = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                .dst_queue_family_index(self.family)
                .buffer(self.buffer)
                .offset(0)
                .size(SIZE);
            self.device.cmd_pipeline_barrier(
                commands[0],
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[acquire],
                &[],
            );
            // The opaque external handoff is invisible to validation's
            // cross-device history. Re-establish conservative local memory
            // visibility after the acquire before this producer queue reads a
            // buffer it originally filled before releasing it externally.
            let visible = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.buffer)
                .offset(0)
                .size(SIZE);
            self.device.cmd_pipeline_barrier(
                commands[0],
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[visible],
                &[],
            );
            self.device.cmd_copy_buffer(
                commands[0],
                self.buffer,
                staging,
                &[vk::BufferCopy::default().size(SIZE)],
            );
            self.device.end_command_buffer(commands[0]).unwrap();
            let waits = [semaphore];
            let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&waits)
                .wait_dst_stage_mask(&stages)
                .command_buffers(&commands);
            self.device
                .queue_submit(self.queue, &[submit], vk::Fence::null())
                .unwrap();
            self.device.queue_wait_idle(self.queue).unwrap();
            let ptr = self
                .device
                .map_memory(memory, 0, SIZE, vk::MemoryMapFlags::empty())
                .unwrap();
            let bytes = std::slice::from_raw_parts(ptr.cast::<u32>(), (SIZE / 4) as usize);
            assert!(
                bytes.iter().all(|value| *value == expected),
                "returned external memory was not preserved"
            );
            self.device.unmap_memory(memory);
            self.device.destroy_buffer(staging, None);
            self.device.free_memory(memory, None);
            self.device.destroy_semaphore(semaphore, None);
        }
    }
}

#[test]
fn win32_external_buffer_and_semaphore_round_trip() {
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
        eprintln!("Vulkan external memory or semaphore handles unavailable; skipping");
        return;
    };
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let external = device
        .external()
        .expect("producer reports external handles supported");
    // SAFETY: handles come from the live producer using the same physical device and metadata;
    // producer has released ownership and signals its semaphore after its GPU fill.
    let memory = unsafe {
        external.import_memory(
            VulkanExternalHandle::Memory(VulkanExternalMemoryHandle {
                handle: memory_handle,
                allocation_size: producer.allocation_size,
                memory_type_index: producer.memory_type_index,
                dedicated: true,
                image_offset: 0,
            }),
            &ExternalResourceDesc {
                size: SIZE,
                offset: 0,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            },
        )
    }
    .unwrap();
    let sync =
        unsafe { external.import_sync(VulkanExternalHandle::Semaphore(semaphore_handle)) }.unwrap();
    let buffer = external.buffer(&memory).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: SIZE,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let marker = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 8,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let marker_readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 8,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    // Two work items in one step exercise a signal in the first submit and a
    // wait in the second. The Win32 export/import preserves the same payload.
    let mut first = device.encoder(lane).unwrap();
    // This command must remain before the imported semaphore wait.
    first
        .write_buffer(&marker, 0, &0x1122_3344u32.to_ne_bytes())
        .unwrap();
    external.acquire(&mut first, &memory, sync).unwrap();
    let handoff = external.release(&mut first, &memory).unwrap();
    // This command must remain after the release signal, on a fresh native CB.
    first
        .write_buffer(&marker, 4, &0x5566_7788u32.to_ne_bytes())
        .unwrap();
    first
        .copy_buffer(&marker, 0, &marker_readback, 0, 8)
        .unwrap();
    let marker_ticket = first.readback_buffer(&marker_readback, 0..8).unwrap();
    let handoff_handle = handoff.into_native().export_handle().unwrap();
    // SAFETY: the first command buffer releases the allocation and signals this payload.
    let handoff_sync =
        unsafe { external.import_sync(VulkanExternalHandle::Semaphore(handoff_handle)) }.unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    external
        .acquire(&mut encoder, &memory, handoff_sync)
        .unwrap();
    encoder.copy_buffer(&buffer, 0, &readback, 0, SIZE).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..SIZE).unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..SIZE,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
                to: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
            },
            ownership: None,
        })])
        .unwrap();
    let returned_bytes = RETURN_PATTERN.to_ne_bytes().repeat((SIZE / 4) as usize);
    encoder.write_buffer(&buffer, 0, &returned_bytes).unwrap();
    let release = external.release(&mut encoder, &memory).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 774002,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: the producer semaphore and ownership release precede this acquire;
    // the consumer release returns ownership and signals the imported semaphore.
    let outcome = unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![
                    first.finish().unwrap(),
                    encoder.finish().unwrap(),
                ]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("external consumer submission failed"),
    };
    let returned_handle = release.into_native().export_handle().unwrap();
    block(device.wait_completion(&point)).unwrap();
    let data = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    let marker_data = unsafe {
        block(device.read_after(marker_ticket, &point))
            .unwrap()
            .bytes
    };
    assert_eq!(
        marker_data,
        [0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55,]
    );
    assert_eq!(data.len(), SIZE as usize);
    assert!(
        data.chunks_exact(4)
            .all(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()) == PATTERN)
    );
    producer.verify_return(returned_handle, RETURN_PATTERN);
}

#[test]
fn win32_external_buffer_compute_lane_round_trip() {
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
        eprintln!("Vulkan external memory or semaphore handles unavailable; skipping");
        return;
    };
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let Some(lane) = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Compute)
        .map(|lane| lane.id)
    else {
        eprintln!("independent Vulkan compute lane unavailable; skipping");
        return;
    };
    let external = device.external().expect("external handle support");
    // SAFETY: the producer exported the live allocation with matching opaque
    // Win32 metadata and transfers ownership through the semaphore below.
    let memory = unsafe {
        external.import_memory(
            VulkanExternalHandle::Memory(VulkanExternalMemoryHandle {
                handle: memory_handle,
                allocation_size: producer.allocation_size,
                memory_type_index: producer.memory_type_index,
                dedicated: true,
                image_offset: 0,
            }),
            &ExternalResourceDesc {
                size: SIZE,
                offset: 0,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            },
        )
    }
    .unwrap();
    // SAFETY: producer signals the exported binary semaphore after writing
    // and releasing the buffer to QUEUE_FAMILY_EXTERNAL.
    let sync =
        unsafe { external.import_sync(VulkanExternalHandle::Semaphore(semaphore_handle)) }.unwrap();
    let buffer = external.buffer(&memory).unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: SIZE,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    external.acquire(&mut encoder, &memory, sync).unwrap();
    encoder.copy_buffer(&buffer, 0, &output, 0, SIZE).unwrap();
    let ticket = encoder.readback_buffer(&output, 0..SIZE).unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..SIZE,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
                to: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
            },
            ownership: None,
        })])
        .unwrap();
    let returned_bytes = RETURN_PATTERN.to_ne_bytes().repeat((SIZE / 4) as usize);
    encoder.write_buffer(&buffer, 0, &returned_bytes).unwrap();
    let release = external.release(&mut encoder, &memory).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 774003,
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
        SubmitOutcome::Accepted(receipt) => receipt,
        _ => panic!("external compute-lane submission failed"),
    };
    let point = &receipt.checkpoints[0].point;
    block(device.wait_completion(point)).unwrap();
    let bytes = unsafe { block(device.read_after(ticket, point)).unwrap().bytes };
    assert!(
        bytes
            .chunks_exact(4)
            .all(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()) == PATTERN)
    );
    producer.verify_return(
        release.into_native().export_handle().unwrap(),
        RETURN_PATTERN,
    );
}
