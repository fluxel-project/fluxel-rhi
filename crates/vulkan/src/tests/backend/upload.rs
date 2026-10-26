//! Real-device diagnostics for the private staging-page pool.
//!
//! This is intentionally a backend unit fixture: it reads private counters
//! without widening the public RHI contract. Fixtures cover cancellation,
//! submitted timeline-gated work, completion-gated reuse and device lifetime.

use crate::backend::{Vulkan, VulkanInstance};
use ash::vk;
use fluxel_rhi_core::api::*;
use std::{
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

struct TimelineGate {
    state: Arc<crate::backend::DeviceState>,
    semaphore: vk::Semaphore,
    released: bool,
}
impl TimelineGate {
    fn release(&mut self) {
        if !std::mem::replace(&mut self.released, true) {
            let info = vk::SemaphoreSignalInfo::default()
                .semaphore(self.semaphore)
                .value(1);
            unsafe { self.state.device.signal_semaphore(&info) }
                .expect("release readback test gate");
        }
    }
}
impl Drop for TimelineGate {
    fn drop(&mut self) {
        if !std::mem::replace(&mut self.released, true) {
            let info = vk::SemaphoreSignalInfo::default()
                .semaphore(self.semaphore)
                .value(1);
            let _ = unsafe { self.state.device.signal_semaphore(&info) };
        }
        unsafe { self.state.device.destroy_semaphore(self.semaphore, None) };
    }
}
fn hold_graphics_queue(device: &Device<Vulkan>) -> Option<TimelineGate> {
    let state = unsafe { device.native_ref() }.0.clone();
    if !state.timeline_semaphore {
        return None;
    }
    let mut kind = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let semaphore = unsafe {
        state.device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
            None,
        )
    }
    .ok()?;
    let waits = [semaphore];
    let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
    let values = [1];
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
    let submit = vk::SubmitInfo::default()
        .wait_semaphores(&waits)
        .wait_dst_stage_mask(&stages)
        .push_next(&mut timeline);
    let submitted = {
        let _queue = state.queue_gate.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: queue_gate serializes this private gate submission with RHI.
        unsafe {
            state
                .device
                .queue_submit(state.graphics_queue, &[submit], vk::Fence::null())
        }
        .is_ok()
    };
    if !submitted {
        unsafe { state.device.destroy_semaphore(semaphore, None) };
        return None;
    }
    Some(TimelineGate {
        state,
        semaphore,
        released: false,
    })
}

fn block<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

fn wait_until(label: &str, predicate: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while !predicate() {
        assert!(Instant::now() < deadline, "timed out waiting for {label}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn rgba_region(texture: &Texture<Vulkan>) -> TextureRegion<'_, Vulkan> {
    TextureRegion {
        texture,
        mip: 0,
        origin: [0, 0, 0],
        extent: [2, 2, 1],
        aspect: TextureAspects::COLOR,
    }
}

#[test]
fn upload_page_logical_capacity_never_uses_allocation_padding() {
    assert_eq!(super::page_size(1), super::DEFAULT_PAGE_BYTES);
    assert_eq!(
        super::page_size(super::DEFAULT_PAGE_BYTES + 7),
        super::DEFAULT_PAGE_BYTES + 7
    );
}

#[test]
#[ignore = "requires a real Vulkan adapter; run with --features hardware-tests -- --ignored"]
fn cancelled_encoder_returns_one_persistently_mapped_page_for_many_small_writes() {
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let lane = device.lanes()[0].id;
    let destination = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 4096,
            usage: BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    // SAFETY: immutable backend diagnostics are observed while `device` owns
    // the native device. This fixture never aliases the raw handles.
    let pool = unsafe { device.native_ref() }.0.upload_pool.clone();
    let before = pool.diagnostics();
    let bytes = [0x5a; 16];
    let mut cancelled = device.encoder(lane).unwrap();
    for index in 0..64 {
        cancelled
            .write_buffer(&destination, index * bytes.len() as u64, &bytes)
            .unwrap();
    }
    let allocated = pool.diagnostics();
    assert_eq!(
        allocated.creates - before.creates,
        1,
        "small writes use one page"
    );
    assert_eq!(
        allocated.maps - before.maps,
        1,
        "a page maps once, persistently"
    );
    drop(cancelled);

    let mut reused = device.encoder(lane).unwrap();
    reused.write_buffer(&destination, 0, &bytes).unwrap();
    let after = pool.diagnostics();
    assert_eq!(
        after.creates, allocated.creates,
        "cancelled page was reused"
    );
    assert_eq!(after.maps, allocated.maps, "reuse did not remap the page");
}

#[test]
#[ignore = "requires a real Vulkan adapter; run with --features hardware-tests -- --ignored"]
fn texture_write_with_nonzero_layout_and_page_offsets_round_trips() {
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let lane = device.lanes()[0].id;
    let scratch = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let texture = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [2, 2, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 40,
            // Keep this copy result device-local so `readback_buffer` must
            // acquire a pooled raw staging page rather than map it directly.
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let layout = TexelCopyBufferLayout {
        offset: 16,
        bytes_per_row: 16,
        rows_per_image: 2,
    };
    // The first 16 bytes are deliberately not texels.  The second row begins
    // after an additional eight bytes of row padding.
    let mut source = vec![0xa5; 40];
    source[16..24].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    source[32..40].copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);
    let mut encoder = device.encoder(lane).unwrap();
    // Occupy the beginning of the page first. The following texture staging
    // allocation must therefore have a nonzero page offset in addition to its
    // nonzero logical buffer layout offset.
    encoder.write_buffer(&scratch, 0, &[0x3c; 16]).unwrap();
    encoder
        .write_texture(rgba_region(&texture), layout, &source)
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &texture,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
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
            layout: Some((ImageLayout::Common, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
    encoder
        .texture_to_buffer(rgba_region(&texture), &output, layout)
        .unwrap();
    let ticket = encoder.readback_buffer(&output, 0..40).unwrap();
    let work = encoder.finish().unwrap();
    let id = PlanId {
        device: device.id(),
        ordinal: 880_001,
    };
    // SAFETY: the sole command step, its lane, and its recording all belong to
    // this device; the End completion covers the readback command.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let receipt = match unsafe {
        device.queue().submit(SubmitPacket {
            plan: plan.clone(),
            steps: vec![BoundStep {
                step: plan.step(0),
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    } {
        SubmitOutcome::Accepted(receipt) => receipt,
        _ => panic!("texture upload submission was rejected"),
    };
    let point = &receipt.checkpoints[0].point;
    block(device.wait_completion(point)).unwrap();
    // SAFETY: the awaited End point covers the texture-to-buffer copy.
    let actual = block(unsafe { device.read_after(ticket, point) })
        .unwrap()
        .bytes;
    assert_eq!(&actual[16..24], &source[16..24]);
    assert_eq!(&actual[32..40], &source[32..40]);
}

#[test]
#[ignore = "requires a real Vulkan adapter; run with --features hardware-tests -- --ignored"]
fn readback_pages_stay_busy_until_encoder_and_ticket_cancellation_releases_them() {
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let lane = device.lanes()[0].id;
    let source = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    // SAFETY: diagnostics are private immutable counters while device owns pool.
    let pool = unsafe { device.native_ref() }.0.readback_pool.clone();
    let before = pool.diagnostics();
    assert_eq!(before.idle, 0, "fresh device has no readback page");
    let mut first = device.encoder(lane).unwrap();
    let first_ticket = first.readback_buffer(&source, 0..16).unwrap();
    let allocated = pool.diagnostics();
    assert_eq!(allocated.creates - before.creates, 1);
    drop(first_ticket);
    // Encoder-held slice must keep the page unavailable even after ticket drop.
    let mut second = device.encoder(lane).unwrap();
    let second_ticket = second.readback_buffer(&source, 0..16).unwrap();
    assert_eq!(pool.diagnostics().creates - before.creates, 2);
    drop(second_ticket);
    drop(first);
    drop(second);
    // Cancelled encoders return both pages; the next ticket reuses one.
    let mut reused = device.encoder(lane).unwrap();
    let ticket = reused.readback_buffer(&source, 0..16).unwrap();
    assert_eq!(pool.diagnostics().creates - before.creates, 2);
    drop(ticket);
    drop(reused);
}

#[test]
#[ignore = "requires a real Vulkan adapter; run with --features hardware-tests -- --ignored"]
fn dropped_public_device_stays_alive_until_device_local_readback_ticket_drops() {
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let lane = device.lanes()[0].id;
    // SAFETY: this weak reference observes only private lifetime ownership.
    let weak = Arc::downgrade(&unsafe { device.native_ref() }.0);
    let source = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    let ticket = encoder.readback_buffer(&source, 0..16).unwrap();
    drop(encoder);
    drop(source);
    drop(device);
    assert!(
        weak.upgrade().is_some(),
        "ticket owns raw-page device lifetime"
    );
    drop(ticket);
    assert!(
        weak.upgrade().is_none(),
        "raw page does not form a device cycle"
    );
}

#[test]
#[ignore = "requires a real Vulkan adapter; run with --features hardware-tests -- --ignored"]
fn submitted_readback_page_is_not_reused_until_the_held_queue_completes() {
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let Some(mut gate) = hold_graphics_queue(&device) else {
        eprintln!("Vulkan timeline semaphore unavailable; skipping");
        return;
    };
    let lane = device.lanes()[0].id;
    let source = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let pool = unsafe { device.native_ref() }.0.readback_pool.clone();
    let before = pool.diagnostics();
    assert_eq!(before.idle, 0, "fresh device has no readback page");
    let mut encoder = device.encoder(lane).unwrap();
    let ticket = encoder.readback_buffer(&source, 0..16).unwrap();
    let work = encoder.finish().unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 880_002,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let receipt = match unsafe {
        device.queue().submit(SubmitPacket {
            plan: plan.clone(),
            steps: vec![BoundStep {
                step: plan.step(0),
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    } {
        SubmitOutcome::Accepted(receipt) => receipt,
        _ => panic!("gated readback submission rejected"),
    };
    drop(ticket);
    let mut while_held = device.encoder(lane).unwrap();
    let held_ticket = while_held.readback_buffer(&source, 0..16).unwrap();
    assert_eq!(
        pool.diagnostics().creates - before.creates,
        2,
        "submitted work retains its page after ticket drop"
    );
    gate.release();
    let point = &receipt.checkpoints[0].point;
    block(device.wait_completion(point)).unwrap();
    wait_until("submitted readback page retirement", || {
        pool.diagnostics().idle >= 1
    });
    // Keep the second page leased. The following allocation can therefore
    // only succeed without a create by reusing the completed submitted page.
    let mut reused = device.encoder(lane).unwrap();
    let ticket = reused.readback_buffer(&source, 0..16).unwrap();
    assert_eq!(
        pool.diagnostics().creates - before.creates,
        2,
        "completed page was reused"
    );
    drop(ticket);
    drop(reused);
    drop(held_ticket);
    drop(while_held);
}
