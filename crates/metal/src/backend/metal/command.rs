//! Metal command submission for the single-queue v13 baseline.
//!
//! Metal has no fence object which maps naturally to a Fluxel completion point.
//! A retained command buffer and its completion handler are that primitive here.
//! The handler is installed before `commit`, so every accepted batch eventually
//! wakes its completion futures even if the application does not call `poll`.
//!
//! Submission is deliberately two phase.  We allocate and encode *all* command
//! buffers first; only then are they committed in plan order.  Consequently an
//! encoding error is still an honest `submit` error, rather than an ambiguous
//! partially accepted prefix.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::task::Waker;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLCullMode, MTLDevice,
    MTLIndexType, MTLLoadAction, MTLOrigin, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLResourceOptions, MTLScissorRect, MTLSize, MTLStoreAction,
    MTLTriangleFillMode, MTLViewport, MTLVisibilityResultMode, MTLWinding,
};

use crate::api::binding::BindingResource;
use crate::api::command::ResourceUse;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::{block_extent, logical_bytes_per_block};
use crate::api::platform::{DeviceLossInfo, DeviceStatus};
use crate::api::resource::transfer::{
    ReadbackRequest, ReadbackStatus, ReadbackTexelLayout, ReadbackTicket, UploadDescriptor,
};
use crate::api::resource::{TextureAspects, TextureDimension};
use crate::api::submission::backend::{SubmissionOutcome, SubmissionRequest};
use crate::api::submission::{CompletionFailure, CompletionState};

use super::device::MetalShared;
use super::resource::{MetalBuffer, MetalTexture};

pub(super) mod native;

/// The mutable execution frontier.  This belongs to the command spine, not to
/// an individual command buffer: all command-buffer callbacks race through this
/// one authority and a terminal loss therefore wakes every pending receipt.
/// Shared completion authority also consulted by host-visible buffer mapping.
/// It is backend-private; only completion serials cross the resource seam.
pub(super) struct SpineState {
    issued: u64,
    completed: u64,
    finished: BTreeSet<u64>,
    failed: Option<(u64, CompletionFailure)>,
    lost: Option<DeviceLossInfo>,
    waiters: BTreeMap<u64, Vec<Waker>>,
    pending_readbacks: BTreeMap<u64, Vec<ReadbackTicket>>,
}

impl SpineState {
    fn new() -> Self {
        Self {
            issued: 0,
            completed: 0,
            finished: BTreeSet::new(),
            failed: None,
            lost: None,
            waiters: BTreeMap::new(),
            pending_readbacks: BTreeMap::new(),
        }
    }

    fn wake_through(&mut self, serial: u64) -> Vec<Waker> {
        let keys = self
            .waiters
            .range(..=serial)
            .map(|(&point, _)| point)
            .collect::<Vec<_>>();
        let mut out = Vec::new();
        for point in keys {
            if let Some(mut waiters) = self.waiters.remove(&point) {
                out.append(&mut waiters);
            }
        }
        out
    }

    fn wake_all(&mut self) -> Vec<Waker> {
        let mut out = Vec::new();
        for (_, mut waiters) in std::mem::take(&mut self.waiters) {
            out.append(&mut waiters);
        }
        out
    }
}

/// The backend-private execution domain used by `MetalDevice`.
pub(super) struct MetalCommandSpine {
    shared: Arc<MetalShared>,
    state: Arc<Mutex<SpineState>>,
    presentation_loss: Arc<super::presentation::MetalPresentationLoss>,
}

/// Native staging retained until the command buffer's completion handler has
/// copied it into the ticket. The public ticket owns the resulting bytes, not a
/// Metal mapping lease, so it remains valid after this native allocation drops.
pub(super) struct MetalPendingReadback {
    pub(super) ticket: ReadbackTicket,
    pub(super) staging: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    pub(super) byte_len: usize,
    pub(super) layout: Option<ReadbackTexelLayout>,
}

impl MetalCommandSpine {
    pub(super) fn new(
        shared: Arc<MetalShared>,
        presentation_loss: Arc<super::presentation::MetalPresentationLoss>,
    ) -> RhiResult<Self> {
        Ok(Self {
            shared,
            state: Arc::new(Mutex::new(SpineState::new())),
            presentation_loss,
        })
    }

    pub(super) fn poll(&self) -> RhiResult<()> {
        // Completion is callback driven.  Keeping this method intentionally
        // non-blocking is important for hosts whose platform pump owns Metal.
        Ok(())
    }

    pub(super) fn status(&self) -> DeviceStatus {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.lost.is_some() {
            DeviceStatus::Lost
        } else {
            DeviceStatus::Active
        }
    }

    pub(super) fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .lost
            .clone()
    }

    pub(super) fn wait_idle(&self) -> RhiResult<()> {
        let command_buffer = self.shared.queue.commandBuffer().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal failed to allocate idle command buffer",
            )
            .at("MetalCommandSpine::wait_idle")
        })?;
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            self.record_terminal_failure("Metal wait-idle command buffer failed");
            return Err(RhiError::new(
                RhiErrorKind::DeviceLost,
                "Metal device was lost while waiting idle",
            )
            .at("MetalCommandSpine::wait_idle"));
        }
        Ok(())
    }

    pub(super) fn completion(&self, serial: u64) -> CompletionState {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        completion_for(&state, serial)
    }

    pub(super) fn completion_or_register_waker(
        &self,
        serial: u64,
        waker: &Waker,
    ) -> CompletionState {
        completion_or_register_waker(&self.state, serial, waker)
    }

    pub(super) fn mapping_state(&self) -> Arc<Mutex<SpineState>> {
        Arc::clone(&self.state)
    }

    /// Accepts finished native command buffers and commits them in plan order.
    ///
    /// Recording owns Metal's command-buffer encoders.  Submission never
    /// lowers portable packets a second time: it only validates ownership,
    /// attaches completion bookkeeping, schedules presentation, and commits
    /// the buffers which were already closed by `CommandEncoder::finish`.
    pub(super) fn submit(&self, request: &SubmissionRequest<'_>) -> RhiResult<SubmissionOutcome> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(info) = &state.lost {
            return Err(
                RhiError::new(RhiErrorKind::DeviceLost, info.message().to_owned())
                    .at("MetalCommandSpine::submit"),
            );
        }
        if request.batches.is_empty() {
            return Ok(SubmissionOutcome {
                completion: state.issued,
                points: Vec::new(),
            });
        }

        // Validate the complete plan before transferring a single token.  This
        // preserves `submit(Err) => zero native work accepted`: a foreign,
        // already-submitted, or repeated buffer cannot leave a committed prefix.
        let mut seen = BTreeSet::new();
        for batch in request.batches {
            for work in &batch.work {
                let native = work
                    .native()
                    .as_any()
                    .downcast_ref::<native::MetalNativeCommandBuffer>()
                    .ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::InvalidUsage,
                            "Metal queue received a command buffer from another backend",
                        )
                        .at("MetalCommandSpine::submit")
                    })?;
                if !native.is_available() {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "a Metal command buffer may be submitted only once",
                    )
                    .at("MetalCommandSpine::submit"));
                }
                if !seen.insert(native as *const native::MetalNativeCommandBuffer) {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "the same Metal command buffer appears more than once in one submission",
                    )
                    .at("MetalCommandSpine::submit"));
                }
            }
        }
        // A present belongs to one plan point, hence to exactly one command
        // buffer in this single-queue baseline. Validate every association while
        // all buffers are still uncommitted: a bad plan must preserve v13's
        // `submit Err => no native work accepted` rule.
        for present in request.presents {
            if !request
                .batches
                .iter()
                .any(|batch| batch.point == present.after)
            {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal present refers to a plan point absent from this submission",
                )
                .at("MetalCommandSpine::submit"));
            }
            super::presentation::frame_attachment(&present.attachment)?;
        }
        let first = state.issued.checked_add(1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "Metal completion serial space is exhausted",
            )
        })?;
        let last = first
            .checked_add(request.batches.len() as u64 - 1)
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "Metal completion serial space is exhausted",
                )
            })?;

        // Transfer every token only after the whole plan passed preflight.
        // They remain uncommitted until presentation has also been scheduled.
        let mut prepared = Vec::with_capacity(request.batches.len());
        for batch in request.batches {
            let mut finished = Vec::with_capacity(batch.work.len());
            for work in &batch.work {
                let native = work
                    .native()
                    .as_any()
                    .downcast_ref::<native::MetalNativeCommandBuffer>()
                    .expect("Metal native buffers were preflighted");
                finished.push(
                    native
                        .take()
                        .expect("Metal native buffer was consumed once"),
                );
            }
            if finished.is_empty() {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a Metal submission batch must contain native command work",
                )
                .at("MetalCommandSpine::submit"));
            }
            prepared.push(finished);
        }

        // `presentDrawable:` is issued before commit, on the final command
        // buffer in the associated FIFO batch.
        for present in request.presents {
            let index = request
                .batches
                .iter()
                .position(|batch| batch.point == present.after)
                .expect("present associations were preflighted");
            let command_buffer = &prepared[index]
                .last()
                .expect("empty native batches were rejected")
                .command_buffer;
            super::presentation::frame_attachment(&present.attachment)?
                .schedule_present(command_buffer)?;
        }

        // Phase B: completion handlers are installed before their final queue
        // commit and the serial frontier becomes visible before native work.
        state.issued = last;
        for (index, (batch, mut finished)) in request.batches.iter().zip(prepared).enumerate() {
            let serial = first + index as u64;
            // Map requests consult this exact accepted serial.  It is recorded
            // only after preflight succeeded for the full plan and directly
            // before the command buffer becomes native work, so a failed
            // submit never makes host mapping wait on imaginary GPU use.
            mark_batch_buffers_accepted(batch, serial);
            let mut final_buffer = finished.pop().ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a Metal submission batch must contain native command work",
                )
                .at("MetalCommandSpine::submit")
            })?;
            let mut readbacks = Vec::new();
            let mut retained_staging = Vec::new();
            for buffer in &mut finished {
                readbacks.append(&mut buffer.readbacks);
                retained_staging.append(&mut buffer.retained_staging);
            }
            readbacks.append(&mut final_buffer.readbacks);
            retained_staging.append(&mut final_buffer.retained_staging);
            let tickets = readbacks.iter().map(|entry| entry.ticket.clone()).collect();
            state.pending_readbacks.insert(serial, tickets);
            // Earlier buffers must reach the queue first. Their finished
            // tokens remain owned by the final completion handler below.
            for buffer in &finished {
                buffer.command_buffer.commit();
            }
            install_completion_handler(
                &final_buffer.command_buffer,
                Arc::clone(&self.state),
                Arc::clone(&self.presentation_loss),
                serial,
                readbacks,
                finished,
                retained_staging,
            );
            // Metal preserves submission order for one command queue. The
            // final buffer's handler consequently represents the complete
            // plan batch, including every earlier recorder retained above.
            final_buffer.command_buffer.commit();
        }
        Ok(SubmissionOutcome {
            completion: last,
            points: request
                .batches
                .iter()
                .enumerate()
                .map(|(index, batch)| (batch.point, first + index as u64))
                .collect(),
        })
    }
    fn record_terminal_failure(&self, message: &str) {
        let waiters = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.lost.is_some() {
                return;
            }
            let info = DeviceLossInfo::new(message.to_owned());
            state.lost = Some(info.clone());
            for ticket in state.pending_readbacks.values().flatten() {
                ticket.set_status(ReadbackStatus::DeviceLost);
            }
            state.pending_readbacks.clear();
            self.presentation_loss.mark_lost(info);
            state.wake_all()
        };
        for waiter in waiters {
            waiter.wake();
        }
    }
}

struct RepackedTextureUpload {
    bytes: Vec<u8>,
    bytes_per_row: u64,
    bytes_per_image: u64,
}

/// The native visibility-result buffer is addressed by a dense byte offset,
/// while the portable query set is addressed by an object identity and slot.
/// This small state machine keeps the two mappings ordered and makes the
/// begin/end pairing independently testable without an Objective-C encoder.
#[derive(Default)]
struct OcclusionQuerySequence {
    next_slot: usize,
    active: Option<ActiveOcclusionQuery>,
}

#[derive(Clone, Copy)]
struct ActiveOcclusionQuery {
    set: crate::api::identity::ObjectId,
    index: u32,
    offset: usize,
}

impl OcclusionQuerySequence {
    fn is_active(&self) -> bool {
        self.active.is_some()
    }

    fn begin(&mut self, set: crate::api::identity::ObjectId, index: u32) -> RhiResult<usize> {
        if self.active.is_some() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal received nested occlusion queries",
            ));
        }
        let offset = self.next_slot.checked_mul(8).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal visibility query offset overflows",
            )
        })?;
        self.next_slot = self.next_slot.checked_add(1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal visibility query count overflows",
            )
        })?;
        self.active = Some(ActiveOcclusionQuery { set, index, offset });
        Ok(offset)
    }

    fn end(&mut self, set: crate::api::identity::ObjectId, index: u32) -> RhiResult<usize> {
        let active = self.active.take().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal received an occlusion-query end without a begin",
            )
        })?;
        if active.set != set || active.index != index {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal occlusion-query end does not match its begin",
            ));
        }
        Ok(active.offset)
    }
}

/// Converts the caller-owned host layout into Metal's buffer-copy layout.
/// Host rows are deliberately allowed to be tightly packed; imposing Metal's
/// four-byte row requirement on asset bytes would violate the upload contract.
/// Copy only logical block bytes, so caller padding never leaks into native
/// staging and compressed formats follow their block-row geometry exactly.
pub(super) fn repack_texture_upload(
    upload: &crate::api::resource::transfer::TextureUploadDescriptor,
) -> RhiResult<RepackedTextureUpload> {
    let descriptor = upload.dst.descriptor();
    repack_texture_upload_parts(
        descriptor.format,
        upload.extent,
        descriptor.dimension,
        upload.subresource.layer_count,
        upload.source_layout,
        &upload.bytes,
    )
}

/// The byte-only half of [`repack_texture_upload`].  It intentionally has no
/// Metal or logical-resource dependency, so compressed block geometry and
/// row-padding behaviour remain executable on hosts that cannot create a
/// Metal device.
fn repack_texture_upload_parts(
    format: crate::api::format::TextureFormat,
    extent: crate::api::resource::Extent3d,
    dimension: TextureDimension,
    layer_count: u32,
    source_layout: crate::api::resource::HostTexelLayout,
    source_bytes: &[u8],
) -> RhiResult<RepackedTextureUpload> {
    let block_bytes = u64::from(logical_bytes_per_block(format).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            "Metal texture upload cannot determine this format's host block size",
        )
    })?);
    let (block_width, block_height) = block_extent(format);
    let logical_row = u64::from(extent.width.div_ceil(block_width))
        .checked_mul(block_bytes)
        .ok_or_else(|| {
            RhiError::new(RhiErrorKind::InvalidUsage, "Metal upload row size overflow")
        })?;
    let bytes_per_row = logical_row
        .checked_add(3)
        .map(|value| value & !3)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal upload row alignment overflow",
            )
        })?;
    let block_rows = u64::from(extent.height.div_ceil(block_height));
    let bytes_per_image = bytes_per_row.checked_mul(block_rows).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "Metal texture-upload image stride overflow",
        )
    })?;
    let image_count = match dimension {
        TextureDimension::D3 => u64::from(extent.depth),
        TextureDimension::D1 | TextureDimension::D2 => u64::from(layer_count),
        _ => {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "unknown texture dimension",
            ));
        }
    };
    let total = bytes_per_image.checked_mul(image_count).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "Metal texture-upload staging size overflow",
        )
    })?;
    let total = usize::try_from(total).map_err(|_| {
        RhiError::new(
            RhiErrorKind::OutOfMemory,
            "Metal texture-upload staging size does not fit the host address space",
        )
    })?;
    let mut bytes = vec![0_u8; total];
    let source_row = u64::from(source_layout.bytes_per_row);
    let source_image = source_row
        .checked_mul(u64::from(source_layout.rows_per_image))
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal texture-upload source image stride overflow",
            )
        })?;
    for image in 0..image_count {
        for row in 0..block_rows {
            let src = image
                .checked_mul(source_image)
                .and_then(|value| value.checked_add(row.checked_mul(source_row)?))
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "Metal texture-upload source offset overflow",
                    )
                })?;
            let dst = image
                .checked_mul(bytes_per_image)
                .and_then(|value| value.checked_add(row.checked_mul(bytes_per_row)?))
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "Metal texture-upload staging offset overflow",
                    )
                })?;
            let src = usize::try_from(src).map_err(|_| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal upload source offset is too large",
                )
            })?;
            let dst = usize::try_from(dst).map_err(|_| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal upload staging offset is too large",
                )
            })?;
            let count = usize::try_from(logical_row).map_err(|_| {
                RhiError::new(RhiErrorKind::InvalidUsage, "Metal upload row is too large")
            })?;
            let source = source_bytes.get(src..src + count).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal texture-upload source bytes do not cover the copied row",
                )
            })?;
            bytes[dst..dst + count].copy_from_slice(source);
        }
    }
    Ok(RepackedTextureUpload {
        bytes,
        bytes_per_row,
        bytes_per_image,
    })
}

pub(super) fn end_blit(slot: &mut Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>) {
    if let Some(encoder) = slot.take() {
        encoder.endEncoding();
    }
}

pub(super) fn scope_switch_error(next: &'static str, active: &'static str) -> RhiError {
    RhiError::new(
        RhiErrorKind::InvalidUsage,
        format!("Metal cannot encode {next} while a {active} scope is open"),
    )
    .at("MetalNativeEncoder")
}

pub(super) fn ensure_blit<'a>(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    slot: &'a mut Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
) -> RhiResult<&'a ProtocolObject<dyn MTLBlitCommandEncoder>> {
    if slot.is_none() {
        *slot = Some(command_buffer.blitCommandEncoder().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal failed to create a blit command encoder",
            )
            .at("MetalNativeEncoder")
        })?);
    }
    slot.as_deref().ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::BackendFailure,
            "Metal did not retain the blit command encoder it created",
        )
        .at("MetalNativeEncoder")
    })
}

pub(super) fn metal_buffer(buffer: &crate::api::resource::Buffer) -> RhiResult<&MetalBuffer> {
    buffer
        .native()
        .as_any()
        .downcast_ref::<MetalBuffer>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "buffer is not backed by this Metal device",
            )
            .at("MetalNativeEncoder")
        })
}

pub(super) fn metal_query_set(
    set: &crate::api::query::QuerySet,
) -> RhiResult<&super::query::MetalQuerySet> {
    set.native()
        .as_any()
        .downcast_ref::<super::query::MetalQuerySet>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "query set is not backed by this Metal device",
            )
            .at("MetalNativeEncoder")
        })
}

pub(super) fn metal_texture(texture: &crate::api::resource::Texture) -> RhiResult<&MetalTexture> {
    texture
        .native()
        .as_any()
        .downcast_ref::<MetalTexture>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "texture is not backed by this Metal device",
            )
            .at("MetalNativeEncoder")
        })
}

pub(super) fn render_pass_descriptor(
    begin: &crate::api::command::record::RasterBegin,
) -> RhiResult<Retained<MTLRenderPassDescriptor>> {
    use crate::api::command::attachment::{
        ColorAttachmentView, DepthAttachmentMode, StencilAttachmentMode,
    };
    use crate::api::command::geometry::{ColorClearValue, LoadOp, StoreOp};
    let pass = MTLRenderPassDescriptor::new();
    // RasterScope validation gives every main attachment one common layer
    // count. Establish the physical array length before draw-specific vertex
    // amplification mappings select layers within it.
    pass.setRenderTargetArrayLength(render_target_array_length(begin));
    let colors = pass.colorAttachments();
    for (location, attachment) in &begin.colors {
        let native = unsafe { colors.objectAtIndexedSubscript(*location as usize) };
        match &attachment.view {
            ColorAttachmentView::Texture(view) => {
                let view = view
                    .native()
                    .as_any()
                    .downcast_ref::<super::resource::MetalTextureView>()
                    .ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::WrongDevice,
                            "color attachment view is not backed by Metal",
                        )
                    })?;
                native.setTexture(Some(&view.raw));
            }
            ColorAttachmentView::Frame(frame) => native.setTexture(Some(
                super::presentation::frame_attachment(frame)?.texture()?,
            )),
            _ => {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    "unknown color attachment view",
                ));
            }
        }
        native.setLoadAction(match attachment.load {
            LoadOp::Load => MTLLoadAction::Load,
            LoadOp::Clear(value) => {
                native.setClearColor(match value {
                    ColorClearValue::Float(v) => MTLClearColor {
                        red: v[0] as f64,
                        green: v[1] as f64,
                        blue: v[2] as f64,
                        alpha: v[3] as f64,
                    },
                    ColorClearValue::Sint(v) => MTLClearColor {
                        red: v[0] as f64,
                        green: v[1] as f64,
                        blue: v[2] as f64,
                        alpha: v[3] as f64,
                    },
                    ColorClearValue::Uint(v) => MTLClearColor {
                        red: v[0] as f64,
                        green: v[1] as f64,
                        blue: v[2] as f64,
                        alpha: v[3] as f64,
                    },
                    _ => {
                        return Err(RhiError::new(
                            RhiErrorKind::Unsupported,
                            "unknown color clear value",
                        ));
                    }
                });
                MTLLoadAction::Clear
            }
        });
        native.setStoreAction(match attachment.store {
            StoreOp::Store => MTLStoreAction::Store,
            StoreOp::Discard => MTLStoreAction::DontCare,
        });
        if let Some(resolve) = &attachment.resolve {
            let resolve = match resolve {
                ColorAttachmentView::Texture(view) => view
                    .native()
                    .as_any()
                    .downcast_ref::<super::resource::MetalTextureView>()
                    .ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::WrongDevice,
                            "resolve view is not backed by Metal",
                        )
                    })?
                    .raw
                    .clone(),
                ColorAttachmentView::Frame(_frame) => {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "Metal direct resolve into a presentation frame is not implemented",
                    )
                    .at("MetalNativeEncoder"));
                }
                _ => {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "unknown resolve attachment view",
                    ));
                }
            };
            native.setResolveTexture(Some(&resolve));
            native.setStoreAction(match attachment.store {
                StoreOp::Store => MTLStoreAction::StoreAndMultisampleResolve,
                StoreOp::Discard => MTLStoreAction::MultisampleResolve,
            });
        }
    }
    if let Some(attachment) = &begin.depth_stencil {
        let view = attachment
            .view
            .native()
            .as_any()
            .downcast_ref::<super::resource::MetalTextureView>()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "depth/stencil attachment view is not backed by Metal",
                )
            })?;
        if let Some(depth) = attachment.depth {
            let native = pass.depthAttachment();
            native.setTexture(Some(&view.raw));
            match depth {
                DepthAttachmentMode::ReadOnly => {
                    native.setLoadAction(MTLLoadAction::Load);
                    native.setStoreAction(MTLStoreAction::Store);
                }
                DepthAttachmentMode::ReadWrite { load, store } => {
                    native.setLoadAction(match load {
                        LoadOp::Load => MTLLoadAction::Load,
                        LoadOp::Clear(value) => {
                            native.setClearDepth(value as f64);
                            MTLLoadAction::Clear
                        }
                    });
                    native.setStoreAction(match store {
                        StoreOp::Store => MTLStoreAction::Store,
                        StoreOp::Discard => MTLStoreAction::DontCare,
                    });
                }
                _ => {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "unknown depth attachment mode",
                    ));
                }
            }
        }
        if let Some(stencil) = attachment.stencil {
            let native = pass.stencilAttachment();
            native.setTexture(Some(&view.raw));
            match stencil {
                StencilAttachmentMode::ReadOnly => {
                    native.setLoadAction(MTLLoadAction::Load);
                    native.setStoreAction(MTLStoreAction::Store);
                }
                StencilAttachmentMode::ReadWrite { load, store } => {
                    native.setLoadAction(match load {
                        LoadOp::Load => MTLLoadAction::Load,
                        LoadOp::Clear(value) => {
                            native.setClearStencil(value);
                            MTLLoadAction::Clear
                        }
                    });
                    native.setStoreAction(match store {
                        StoreOp::Store => MTLStoreAction::Store,
                        StoreOp::Discard => MTLStoreAction::DontCare,
                    });
                }
                _ => {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "unknown stencil attachment mode",
                    ));
                }
            }
        }
    }
    Ok(pass)
}

/// The recorder already validated a common attachment layer count. This mirrors
/// its primary-attachment order: color first, then depth/stencil-only scopes.
fn render_target_array_length(begin: &crate::api::command::record::RasterBegin) -> usize {
    begin
        .colors
        .first()
        .map(|(_, color)| match &color.view {
            crate::api::command::ColorAttachmentView::Texture(view) => view.layer_count(),
            crate::api::command::ColorAttachmentView::Frame(_) => 1,
            _ => 1,
        })
        .or_else(|| {
            begin
                .depth_stencil
                .as_ref()
                .map(|attachment| attachment.view.layer_count())
        })
        .unwrap_or(1) as usize
}

pub(super) fn raster_scope_extent(
    begin: &crate::api::command::record::RasterBegin,
) -> RhiResult<(u32, u32)> {
    if let Some((_, color)) = begin.colors.first() {
        let extent = color.view.extent();
        return Ok((extent.width, extent.height));
    }
    if let Some(depth_stencil) = &begin.depth_stencil {
        let extent = depth_stencil.view.extent();
        return Ok((extent.width, extent.height));
    }
    Err(RhiError::new(
        RhiErrorKind::InvalidUsage,
        "Metal raster scope has no attachment extent",
    ))
}

pub(super) fn bind_vertex_buffers(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    bindings: &[(u32, crate::api::resource::BufferBinding)],
) -> RhiResult<()> {
    for (slot, binding) in bindings {
        let buffer = metal_buffer(&binding.buffer)?;
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(
                Some(&buffer.raw),
                binding.range.offset as usize,
                *slot as usize,
            )
        };
    }
    Ok(())
}

pub(super) fn metal_primitive(value: crate::api::pipeline::PrimitiveTopology) -> MTLPrimitiveType {
    use crate::api::pipeline::PrimitiveTopology as P;
    match value {
        P::PointList => MTLPrimitiveType::Point,
        P::LineList => MTLPrimitiveType::Line,
        P::LineStrip => MTLPrimitiveType::LineStrip,
        P::TriangleList => MTLPrimitiveType::Triangle,
        P::TriangleStrip => MTLPrimitiveType::TriangleStrip,
        _ => MTLPrimitiveType::Triangle,
    }
}
pub(super) fn metal_index_type(value: crate::api::command::IndexFormat) -> MTLIndexType {
    match value {
        crate::api::command::IndexFormat::Uint16 => MTLIndexType::UInt16,
        crate::api::command::IndexFormat::Uint32 => MTLIndexType::UInt32,
    }
}
pub(super) fn metal_cull_mode(value: crate::api::pipeline::CullMode) -> MTLCullMode {
    match value {
        crate::api::pipeline::CullMode::None => MTLCullMode::None,
        crate::api::pipeline::CullMode::Front => MTLCullMode::Front,
        crate::api::pipeline::CullMode::Back => MTLCullMode::Back,
        _ => MTLCullMode::None,
    }
}
pub(super) fn metal_winding(value: crate::api::pipeline::FrontFace) -> MTLWinding {
    match value {
        crate::api::pipeline::FrontFace::Ccw => MTLWinding::CounterClockwise,
        crate::api::pipeline::FrontFace::Cw => MTLWinding::Clockwise,
        _ => MTLWinding::CounterClockwise,
    }
}
pub(super) fn metal_fill_mode(
    value: crate::api::pipeline::PolygonMode,
) -> RhiResult<MTLTriangleFillMode> {
    match value {
        crate::api::pipeline::PolygonMode::Fill => Ok(MTLTriangleFillMode::Fill),
        crate::api::pipeline::PolygonMode::Line => Ok(MTLTriangleFillMode::Lines),
        crate::api::pipeline::PolygonMode::Point => Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "Metal has no point polygon-mode lowering",
        )),
        _ => Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "unknown polygon mode",
        )),
    }
}
pub(super) fn metal_viewport(value: crate::api::command::Viewport) -> MTLViewport {
    MTLViewport {
        originX: value.x as f64,
        originY: value.y as f64,
        width: value.width as f64,
        height: value.height as f64,
        znear: value.min_depth as f64,
        zfar: value.max_depth as f64,
    }
}
pub(super) fn metal_scissor(value: crate::api::command::Rect) -> MTLScissorRect {
    MTLScissorRect {
        x: value.x as usize,
        y: value.y as usize,
        width: value.width as usize,
        height: value.height as usize,
    }
}

/// Applies the immutable portable packets through the ABI constructed with the
/// pipeline.  The direct Metal path intentionally binds every packet at every
/// dispatch; state-diff caching is an optimization that must not weaken the
/// group/slot/dynamic-offset proof performed here.
pub(super) fn bind_compute_groups(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    abi: &super::binding::MetalBindingAbi,
    groups: &[crate::api::command::record::BoundGroup],
) -> RhiResult<()> {
    for bound in groups {
        let packet = bound
            .group
            .native()
            .as_any()
            .downcast_ref::<super::binding::MetalBindGroup>()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "compute bind group is not backed by this Metal device",
                )
                .at("MetalNativeEncoder")
            })?;
        let group_abi = abi.group(bound.index).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal compute pipeline has no ABI for a bound group",
            )
            .at("MetalNativeEncoder")
        })?;
        let dynamic = abi.dynamic_offsets(bound.index, &bound.group, &bound.dynamic_offsets)?;
        for (slot, resource) in packet.entries() {
            // Layout packets may contain resources unused by this entry point.
            // The pipeline ABI intentionally gives those no MSL index.
            let Some(slot_abi) = group_abi.slot(*slot) else {
                continue;
            };
            let first = slot_abi.first.compute.ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "compute bind packet contains a non-compute-visible slot",
                )
                .at("MetalNativeEncoder")
            })?;
            match resource {
                BindingResource::Buffer(binding) => bind_compute_buffer(
                    encoder,
                    first,
                    binding,
                    dynamic_offset(&dynamic, *slot, 0),
                )?,
                BindingResource::BufferArray(bindings) => {
                    for (element, binding) in bindings.iter().enumerate() {
                        bind_compute_buffer(
                            encoder,
                            first + element as u32,
                            binding,
                            dynamic_offset(&dynamic, *slot, element as u32),
                        )?;
                    }
                }
                BindingResource::Texture(view) => bind_compute_texture(encoder, first, view)?,
                BindingResource::TextureArray(views) => {
                    for (element, view) in views.iter().enumerate() {
                        bind_compute_texture(encoder, first + element as u32, view)?;
                    }
                }
                BindingResource::Sampler(sampler) => bind_compute_sampler(encoder, first, sampler)?,
                BindingResource::SamplerArray(samplers) => {
                    for (element, sampler) in samplers.iter().enumerate() {
                        bind_compute_sampler(encoder, first + element as u32, sampler)?;
                    }
                }
                BindingResource::AccelerationStructure(_)
                | BindingResource::AccelerationStructureArray(_)
                | BindingResource::ExternalTexture(_) => {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "Metal direct binding does not implement this resource class",
                    )
                    .at("MetalNativeEncoder"));
                }
                _ => {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "unknown compute binding resource",
                    )
                    .at("MetalNativeEncoder"));
                }
            }
        }
    }
    Ok(())
}

pub(super) fn bind_raster_groups(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    abi: &super::binding::MetalBindingAbi,
    groups: &[crate::api::command::record::BoundGroup],
) -> RhiResult<()> {
    for bound in groups {
        let packet = bound
            .group
            .native()
            .as_any()
            .downcast_ref::<super::binding::MetalBindGroup>()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "raster bind group is not backed by this Metal device",
                )
            })?;
        let group_abi = abi.group(bound.index).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal raster pipeline has no ABI for a bound group",
            )
        })?;
        let dynamic = abi.dynamic_offsets(bound.index, &bound.group, &bound.dynamic_offsets)?;
        for (slot, resource) in packet.entries() {
            // See compute binding above: a portable bind group may be a
            // layout superset of the compiled vertex/fragment artifacts.
            let Some(entry) = group_abi.slot(*slot) else {
                continue;
            };
            match resource {
                BindingResource::Buffer(value) => bind_raster_buffer(
                    encoder,
                    entry.first.vertex,
                    entry.first.fragment,
                    value,
                    dynamic_offset(&dynamic, *slot, 0),
                )?,
                BindingResource::BufferArray(values) => {
                    for (i, value) in values.iter().enumerate() {
                        bind_raster_buffer(
                            encoder,
                            entry.first.vertex.map(|v| v + i as u32),
                            entry.first.fragment.map(|v| v + i as u32),
                            value,
                            dynamic_offset(&dynamic, *slot, i as u32),
                        )?
                    }
                }
                BindingResource::Texture(value) => {
                    bind_raster_texture(encoder, entry.first.vertex, entry.first.fragment, value)?
                }
                BindingResource::TextureArray(values) => {
                    for (i, value) in values.iter().enumerate() {
                        bind_raster_texture(
                            encoder,
                            entry.first.vertex.map(|v| v + i as u32),
                            entry.first.fragment.map(|v| v + i as u32),
                            value,
                        )?
                    }
                }
                BindingResource::Sampler(value) => {
                    bind_raster_sampler(encoder, entry.first.vertex, entry.first.fragment, value)?
                }
                BindingResource::SamplerArray(values) => {
                    for (i, value) in values.iter().enumerate() {
                        bind_raster_sampler(
                            encoder,
                            entry.first.vertex.map(|v| v + i as u32),
                            entry.first.fragment.map(|v| v + i as u32),
                            value,
                        )?
                    }
                }
                BindingResource::AccelerationStructure(_)
                | BindingResource::AccelerationStructureArray(_)
                | BindingResource::ExternalTexture(_) => {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "Metal direct raster binding does not implement this resource class",
                    ));
                }
                _ => {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "unknown raster binding resource",
                    ));
                }
            }
        }
    }
    Ok(())
}
pub(super) fn bind_raster_buffer(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    vertex: Option<u32>,
    fragment: Option<u32>,
    binding: &crate::api::resource::BufferBinding,
    dynamic: u64,
) -> RhiResult<()> {
    let buffer = metal_buffer(&binding.buffer)?;
    let offset = binding.range.offset.checked_add(dynamic).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "Metal dynamic buffer offset overflow",
        )
    })? as usize;
    if let Some(index) = vertex {
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(&buffer.raw), offset, index as usize)
        };
    }
    if let Some(index) = fragment {
        unsafe {
            encoder.setFragmentBuffer_offset_atIndex(Some(&buffer.raw), offset, index as usize)
        };
    }
    Ok(())
}
pub(super) fn bind_raster_texture(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    vertex: Option<u32>,
    fragment: Option<u32>,
    value: &crate::api::resource::TextureView,
) -> RhiResult<()> {
    let view = value
        .native()
        .as_any()
        .downcast_ref::<super::resource::MetalTextureView>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "texture view is not backed by this Metal device",
            )
        })?;
    if let Some(index) = vertex {
        unsafe { encoder.setVertexTexture_atIndex(Some(&view.raw), index as usize) };
    }
    if let Some(index) = fragment {
        unsafe { encoder.setFragmentTexture_atIndex(Some(&view.raw), index as usize) };
    }
    Ok(())
}
pub(super) fn bind_raster_sampler(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    vertex: Option<u32>,
    fragment: Option<u32>,
    value: &crate::api::resource::Sampler,
) -> RhiResult<()> {
    let sampler = value
        .native()
        .as_any()
        .downcast_ref::<super::resource::MetalSampler>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "sampler is not backed by this Metal device",
            )
        })?;
    if let Some(index) = vertex {
        unsafe { encoder.setVertexSamplerState_atIndex(Some(&sampler.raw), index as usize) };
    }
    if let Some(index) = fragment {
        unsafe { encoder.setFragmentSamplerState_atIndex(Some(&sampler.raw), index as usize) };
    }
    Ok(())
}

fn immediate_bytes(
    abi: &super::binding::MetalBindingAbi,
    writes: &[crate::api::command::record::ImmediateWrite],
) -> RhiResult<Vec<u8>> {
    let immediate = abi.immediates();
    let mut bytes = vec![0; immediate.size as usize];
    for write in writes {
        let end = write
            .offset
            .checked_add(u32::try_from(write.bytes.len()).map_err(|_| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal immediate write length exceeds u32",
                )
            })?)
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal immediate write overflows",
                )
            })?;
        for required in &immediate.requirements {
            let required_end = required.offset.checked_add(required.size).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal immediate requirement overflows",
                )
            })?;
            let copy_start = write.offset.max(required.offset);
            let copy_end = end.min(required_end);
            if copy_start >= copy_end {
                continue;
            }
            let source_start = (copy_start - write.offset) as usize;
            let source_end = (copy_end - write.offset) as usize;
            bytes[copy_start as usize..copy_end as usize]
                .copy_from_slice(&write.bytes[source_start..source_end]);
        }
    }
    Ok(bytes)
}
pub(super) fn bind_compute_immediates(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    abi: &super::binding::MetalBindingAbi,
    writes: &[crate::api::command::record::ImmediateWrite],
) -> RhiResult<()> {
    let immediate = abi.immediates();
    if immediate.size == 0 {
        return Ok(());
    }
    let bytes = immediate_bytes(abi, writes)?;
    let pointer =
        std::ptr::NonNull::new(bytes.as_ptr() as *mut core::ffi::c_void).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal immediate byte storage is null",
            )
        })?;
    if let Some(index) = immediate.indices.compute {
        unsafe { encoder.setBytes_length_atIndex(pointer, bytes.len(), index as usize) };
    }
    Ok(())
}
pub(super) fn bind_raster_immediates(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    abi: &super::binding::MetalBindingAbi,
    writes: &[crate::api::command::record::ImmediateWrite],
) -> RhiResult<()> {
    let immediate = abi.immediates();
    if immediate.size == 0 {
        return Ok(());
    }
    let bytes = immediate_bytes(abi, writes)?;
    let pointer =
        std::ptr::NonNull::new(bytes.as_ptr() as *mut core::ffi::c_void).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal immediate byte storage is null",
            )
        })?;
    if let Some(index) = immediate.indices.vertex {
        unsafe { encoder.setVertexBytes_length_atIndex(pointer, bytes.len(), index as usize) };
    }
    if let Some(index) = immediate.indices.fragment {
        unsafe { encoder.setFragmentBytes_length_atIndex(pointer, bytes.len(), index as usize) };
    }
    Ok(())
}
pub(super) fn index_element_size(value: crate::api::command::IndexFormat) -> u64 {
    match value {
        crate::api::command::IndexFormat::Uint16 => 2,
        crate::api::command::IndexFormat::Uint32 => 4,
    }
}

/// Checks whether the encoder can select the public base-vertex/base-instance
/// overload.  Older Metal profiles have only the zero-base overload; that is a
/// correct fallback for a zero base, not permission to silently drop a caller's
/// non-zero first instance or base vertex.
pub(super) fn validate_base_vertex_instance_selector(
    selector_available: bool,
    indexed: bool,
    base_vertex: i32,
    first_instance: u32,
) -> RhiResult<()> {
    if selector_available || (base_vertex == 0 && first_instance == 0) {
        return Ok(());
    }
    let operation = if indexed && base_vertex != 0 {
        "base-vertex/base-instance"
    } else {
        "base-instance"
    };
    Err(RhiError::new(
        RhiErrorKind::Unsupported,
        format!("Metal {operation} draw selector is unavailable on this device"),
    )
    .at("MetalNativeEncoder"))
}

fn dynamic_offset(
    offsets: &[super::binding::MetalDynamicOffset],
    slot: crate::api::binding::BindingSlotId,
    element: u32,
) -> u64 {
    offsets
        .iter()
        .find(|offset| offset.slot == slot && offset.element == element)
        .map_or(0, |offset| offset.offset)
}

pub(super) fn bind_compute_buffer(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: u32,
    binding: &crate::api::resource::BufferBinding,
    dynamic: u64,
) -> RhiResult<()> {
    let buffer = metal_buffer(&binding.buffer)?;
    let offset = binding.range.offset.checked_add(dynamic).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "Metal dynamic buffer offset overflow",
        )
    })?;
    unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer.raw), offset as usize, index as usize) };
    Ok(())
}

pub(super) fn bind_compute_texture(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: u32,
    view: &crate::api::resource::TextureView,
) -> RhiResult<()> {
    let view = view
        .native()
        .as_any()
        .downcast_ref::<super::resource::MetalTextureView>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "texture view is not backed by this Metal device",
            )
        })?;
    unsafe { encoder.setTexture_atIndex(Some(&view.raw), index as usize) };
    Ok(())
}

pub(super) fn bind_compute_sampler(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: u32,
    sampler: &crate::api::resource::Sampler,
) -> RhiResult<()> {
    let sampler = sampler
        .native()
        .as_any()
        .downcast_ref::<super::resource::MetalSampler>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "sampler is not backed by this Metal device",
            )
        })?;
    unsafe { encoder.setSamplerState_atIndex(Some(&sampler.raw), index as usize) };
    Ok(())
}

pub(super) fn origin(value: crate::api::resource::Origin3d) -> MTLOrigin {
    MTLOrigin {
        x: value.x as usize,
        y: value.y as usize,
        z: value.z as usize,
    }
}

pub(super) fn size(value: crate::api::resource::Extent3d) -> MTLSize {
    MTLSize {
        width: value.width as usize,
        height: value.height as usize,
        depth: value.depth as usize,
    }
}

fn install_completion_handler(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    state: Arc<Mutex<SpineState>>,
    presentation_loss: Arc<super::presentation::MetalPresentationLoss>,
    serial: u64,
    readbacks: Vec<MetalPendingReadback>,
    // Retain staging and the earlier command buffers until the final buffer in
    // this FIFO batch completes. Metal does not promise caller-owned upload
    // allocations survive merely because an encoder referenced them.
    _earlier_buffers: Vec<native::FinishedMetalCommandBuffer>,
    _retained_staging: Vec<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
) {
    let block = RcBlock::new(
        move |completed: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let command_buffer = unsafe { completed.as_ref() };
            let completed_ok = command_buffer.status() == MTLCommandBufferStatus::Completed;
            // A completion point that owns readback is not logically Complete
            // until the CPU-visible ticket has been published. Do this before
            // advancing the shared frontier, otherwise another thread can
            // observe CompletionState::Complete while the ticket is Pending.
            let mut publication_failed = false;
            if completed_ok {
                for readback in &readbacks {
                    // Shared staging has no separate map/unmap lease. The
                    // command-buffer callback establishes GPU visibility.
                    let pointer = readback.staging.contents().cast::<u8>();
                    let Some(pointer) = std::ptr::NonNull::new(pointer.as_ptr()) else {
                        readback.ticket.set_status(ReadbackStatus::Failed);
                        publication_failed = true;
                        continue;
                    };
                    let bytes = unsafe {
                        std::slice::from_raw_parts(pointer.as_ptr(), readback.byte_len).to_vec()
                    };
                    readback.ticket.publish(bytes, readback.layout);
                }
            }

            let mut state = state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let waiters = if completed_ok {
                state.pending_readbacks.remove(&serial);
                if publication_failed {
                    state.failed.get_or_insert_with(|| {
                        (
                            serial,
                            CompletionFailure::new(
                                "Metal readback staging was unavailable at completion",
                            ),
                        )
                    });
                    state.wake_all()
                } else {
                    state.finished.insert(serial);
                    loop {
                        let next = state.completed + 1;
                        if !state.finished.remove(&next) {
                            break;
                        }
                        state.completed += 1;
                    }
                    let completed = state.completed;
                    state.wake_through(completed)
                }
            } else {
                let message = "Metal command buffer terminated with an execution error";
                // Metal gives a failed command buffer after `commit` no safe
                // recovery contract for this DeviceIdentity. Treat it as the
                // execution-domain terminal state rather than allowing later
                // mapping/acquire/present futures to remain Pending.
                let info = DeviceLossInfo::new(message.to_owned());
                state.lost.get_or_insert_with(|| info.clone());
                presentation_loss.mark_lost(info);
                for ticket in state.pending_readbacks.values().flatten() {
                    ticket.set_status(ReadbackStatus::DeviceLost);
                }
                state.pending_readbacks.clear();
                state.wake_all()
            };
            drop(state);
            if !completed_ok {
                // A post-commit Metal execution error terminally invalidates
                // this DeviceIdentity. Do not publish stale staging contents.
                for readback in &readbacks {
                    readback.ticket.set_status(ReadbackStatus::DeviceLost);
                }
            }
            for waiter in waiters {
                waiter.wake();
            }
        },
    );
    unsafe {
        command_buffer.addCompletedHandler(RcBlock::as_ptr(&block));
    }
}

fn completion_for(state: &SpineState, serial: u64) -> CompletionState {
    if serial <= state.completed {
        return CompletionState::Complete;
    }
    if let Some(info) = &state.lost {
        return CompletionState::DeviceLost(info.clone());
    }
    if let Some((first, failure)) = &state.failed {
        if serial >= *first {
            return CompletionState::Failed(failure.clone());
        }
    }
    CompletionState::Pending
}

/// The mapping seam uses the same completion frontier as ordinary completion
/// futures.  Registering while the mutex is held closes the completion-before-
/// waker race; exposing a host-visible `contents()` pointer sooner would allow
/// CPU access concurrent with accepted GPU work.
pub(super) fn completion_or_register_waker(
    state: &Arc<Mutex<SpineState>>,
    serial: u64,
    waker: &Waker,
) -> CompletionState {
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let answer = completion_for(&state, serial);
    if matches!(answer, CompletionState::Pending) {
        let waiters = state.waiters.entry(serial).or_default();
        if !waiters.iter().any(|known| known.will_wake(waker)) {
            waiters.push(waker.clone());
        }
    }
    answer
}

fn mark_batch_buffers_accepted(batch: &crate::api::submission::plan::PlanBatch, serial: u64) {
    for use_record in batch
        .work
        .iter()
        .flat_map(crate::api::command::RecordedWork::resource_uses)
    {
        if let ResourceUse::Buffer(buffer) = use_record {
            if let Some(native) = buffer
                .buffer
                .native()
                .as_any()
                .downcast_ref::<MetalBuffer>()
            {
                native.mark_accepted(serial);
            }
        }
    }
}

/// Encodes a CPU-visible staging copy but deliberately does not publish it.
/// Metal can execute the blit asynchronously after this function returns; the
/// completion handler is the sole publisher so `ReadbackStatus::Ready` always
/// implies bytes from completed GPU work.
/// Clears whole selected subresources to Metal's byte-zero value.
///
/// Metal has no format-independent blit clear. Color subresources therefore
/// use a zero-filled shared buffer and the ordinary buffer-to-texture blit
/// route. This is deliberately not a shader fallback: it also works for the
/// block-compressed formats whose zero block is the command's documented
/// backend-defined zero value. Depth/stencil has no byte-copy route in the
/// published table, so it uses a one-attachment render pass instead. The
/// recorder requires `DEPTH_STENCIL_ATTACHMENT` for that case, which is the
/// native `RenderTarget` usage needed by Metal.
pub(super) fn lower_clear_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    blit: &mut Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
    texture: &crate::api::resource::Texture,
    range: crate::api::resource::TextureSubresourceRange,
) -> RhiResult<()> {
    if range.aspects == TextureAspects::COLOR {
        return lower_color_clear(device, command_buffer, blit, texture, range);
    }
    lower_depth_stencil_clear(command_buffer, blit, texture, range)
}

fn lower_color_clear(
    device: &ProtocolObject<dyn MTLDevice>,
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    blit: &mut Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
    texture: &crate::api::resource::Texture,
    range: crate::api::resource::TextureSubresourceRange,
) -> RhiResult<()> {
    let native = metal_texture(texture)?;
    let descriptor = texture.descriptor();
    let bytes_per_block = crate::api::format::logical_bytes_per_block(descriptor.format)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "Metal cannot byte-clear a color format without a fixed block layout",
            )
            .at("MetalNativeEncoder")
        })?;
    let (block_width, block_height) = crate::api::format::block_extent(descriptor.format);
    let is_3d = descriptor.dimension == TextureDimension::D3;
    let copies_per_mip = if is_3d { 1 } else { range.layer_count };

    for mip_level in range.base_mip..range.base_mip + range.mip_count {
        let extent = crate::api::resource::texture::mip_extent(
            descriptor.extent,
            descriptor.dimension,
            mip_level,
        );
        let block_columns = extent.width.div_ceil(block_width);
        let block_rows = extent.height.div_ceil(block_height);
        let tight_row = u64::from(block_columns)
            .checked_mul(u64::from(bytes_per_block))
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "Metal clear texture row pitch overflows",
                )
                .at("MetalNativeEncoder")
            })?;
        // The Metal copy facts for this backend publish the native four-byte
        // row-pitch alignment. Padding is zero too, so it cannot change a
        // copied block even for the final partial block row of a mip.
        let bytes_per_row = align_clear_row(tight_row)?;
        let bytes_per_image = bytes_per_row
            .checked_mul(u64::from(block_rows))
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "Metal clear texture image stride overflows",
                )
                .at("MetalNativeEncoder")
            })?;
        let byte_len = bytes_per_image
            .checked_mul(if is_3d { u64::from(extent.depth) } else { 1 })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "Metal clear texture staging allocation overflows",
                )
                .at("MetalNativeEncoder")
            })?;
        let byte_len = usize::try_from(byte_len).map_err(|_| {
            RhiError::new(
                RhiErrorKind::OutOfMemory,
                "Metal clear texture staging allocation exceeds host address space",
            )
            .at("MetalNativeEncoder")
        })?;
        let bytes_per_row = usize::try_from(bytes_per_row).map_err(|_| {
            RhiError::new(
                RhiErrorKind::OutOfMemory,
                "Metal clear texture row pitch exceeds host address space",
            )
            .at("MetalNativeEncoder")
        })?;
        let bytes_per_image = usize::try_from(bytes_per_image).map_err(|_| {
            RhiError::new(
                RhiErrorKind::OutOfMemory,
                "Metal clear texture image stride exceeds host address space",
            )
            .at("MetalNativeEncoder")
        })?;

        for layer in 0..copies_per_mip {
            let staging = device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::OutOfMemory,
                        "Metal clear texture staging allocation failed",
                    )
                    .at("MetalNativeEncoder")
                })?;
            // Allocation contents are intentionally written, rather than
            // relying on a platform allocator's initialisation convention.
            unsafe { std::ptr::write_bytes(staging.contents().as_ptr(), 0, byte_len) };
            let encoder = ensure_blit(command_buffer, blit)?;
            unsafe {
                encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &staging,
                    0,
                    bytes_per_row,
                    bytes_per_image,
                    size(extent),
                    &native.raw,
                    if is_3d { 0 } else { (range.base_layer + layer) as usize },
                    mip_level as usize,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
        }
    }
    Ok(())
}

fn lower_depth_stencil_clear(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    blit: &mut Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
    texture: &crate::api::resource::Texture,
    range: crate::api::resource::TextureSubresourceRange,
) -> RhiResult<()> {
    if range.aspects.contains(TextureAspects::COLOR)
        || !(range.aspects.contains(TextureAspects::DEPTH)
            || range.aspects.contains(TextureAspects::STENCIL))
    {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "Metal clear texture cannot mix color with depth/stencil aspects",
        )
        .at("MetalNativeEncoder"));
    }
    let native = metal_texture(texture)?;
    let descriptor = texture.descriptor();
    // A render-pass attachment is a 2D texture or 2D array slice. This
    // defensive check mirrors capability creation; it prevents a stale facts
    // snapshot from treating a 3D depth allocation as clearable.
    if descriptor.dimension != TextureDimension::D2 || descriptor.sample_count != 1 {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "Metal depth/stencil ClearTexture requires a single-sampled 2D texture",
        )
        .at("MetalNativeEncoder"));
    }
    let format_aspects = crate::api::format::format_aspects(descriptor.format);
    end_blit(blit);
    for mip_level in range.base_mip..range.base_mip + range.mip_count {
        for layer in range.base_layer..range.base_layer + range.layer_count {
            let pass = MTLRenderPassDescriptor::new();
            if format_aspects.contains(TextureAspects::DEPTH) {
                let attachment = pass.depthAttachment();
                attachment.setTexture(Some(&native.raw));
                attachment.setLevel(mip_level as usize);
                attachment.setSlice(layer as usize);
                if range.aspects.contains(TextureAspects::DEPTH) {
                    attachment.setClearDepth(0.0);
                    attachment.setLoadAction(MTLLoadAction::Clear);
                } else {
                    attachment.setLoadAction(MTLLoadAction::Load);
                }
                attachment.setStoreAction(MTLStoreAction::Store);
            }
            if format_aspects.contains(TextureAspects::STENCIL) {
                let attachment = pass.stencilAttachment();
                attachment.setTexture(Some(&native.raw));
                attachment.setLevel(mip_level as usize);
                attachment.setSlice(layer as usize);
                if range.aspects.contains(TextureAspects::STENCIL) {
                    attachment.setClearStencil(0);
                    attachment.setLoadAction(MTLLoadAction::Clear);
                } else {
                    attachment.setLoadAction(MTLLoadAction::Load);
                }
                attachment.setStoreAction(MTLStoreAction::Store);
            }
            let encoder = command_buffer
                .renderCommandEncoderWithDescriptor(&pass)
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        "Metal failed to create a depth/stencil clear render encoder",
                    )
                    .at("MetalNativeEncoder")
                })?;
            encoder.endEncoding();
        }
    }
    Ok(())
}

fn align_clear_row(value: u64) -> RhiResult<u64> {
    value.checked_add(3).map(|row| row & !3).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::OutOfMemory,
            "Metal clear texture row alignment overflows",
        )
        .at("MetalNativeEncoder")
    })
}

pub(super) fn lower_readback(
    device: &ProtocolObject<dyn MTLDevice>,
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    blit: &mut Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
    ticket: &ReadbackTicket,
    retained: &mut Vec<MetalPendingReadback>,
) -> RhiResult<()> {
    match ticket.request() {
        ReadbackRequest::Buffer { src, range, .. } => {
            let source = metal_buffer(src)?;
            let byte_len = usize::try_from(range.size).map_err(|_| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "Metal buffer readback exceeds host address space",
                )
                .at("MetalNativeEncoder")
            })?;
            let staging = device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::OutOfMemory,
                        "Metal buffer-readback staging allocation failed",
                    )
                    .at("MetalNativeEncoder")
                })?;
            let encoder = ensure_blit(command_buffer, blit)?;
            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &source.raw,
                    range.offset as usize,
                    &staging,
                    0,
                    byte_len,
                );
            }
            retained.push(MetalPendingReadback {
                ticket: ticket.clone(),
                staging,
                byte_len,
                layout: None,
            });
        }
        ReadbackRequest::Texture {
            src,
            subresource,
            origin: source_origin,
            extent,
            ..
        } => {
            let bytes_per_block = crate::api::format::logical_bytes_per_block(
                src.descriptor().format,
            )
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "Metal texture readback format has no byte-copy block size",
                )
                .at("MetalNativeEncoder")
            })?;
            let (_, block_height) = crate::api::format::block_extent(src.descriptor().format);
            let tight_row = extent
                .width
                .div_ceil(crate::api::format::block_extent(src.descriptor().format).0)
                .checked_mul(bytes_per_block)
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "Metal texture readback row pitch overflows",
                    )
                    .at("MetalNativeEncoder")
                })?;
            // The documented portable layout makes no promise of tight rows.
            // A 256-byte pitch satisfies Metal's blit-buffer alignment on the
            // platform families this backend supports and keeps completion data
            // directly usable without a second CPU repack.
            let bytes_per_row = align_readback_row(u64::from(tight_row))?;
            let rows_per_image = extent.height.div_ceil(block_height);
            // Array layers occupy distinct Metal source slices, while a 3D
            // copy uses one slice and makes its Z extent part of the image
            // footprint. Both layouts expose every image/depth slice through
            // the same portable rows_per_image stride.
            let is_3d = src.descriptor().dimension == TextureDimension::D3;
            let images = if is_3d {
                u64::from(extent.depth)
            } else {
                u64::from(subresource.layer_count)
            };
            let image_stride = bytes_per_row
                .checked_mul(u64::from(rows_per_image))
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "Metal texture readback image stride overflows",
                    )
                    .at("MetalNativeEncoder")
                })?;
            let total_size = image_stride.checked_mul(images).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "Metal texture readback staging size overflows",
                )
                .at("MetalNativeEncoder")
            })?;
            let byte_len = usize::try_from(total_size).map_err(|_| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "Metal texture readback exceeds host address space",
                )
                .at("MetalNativeEncoder")
            })?;
            let staging = device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::OutOfMemory,
                        "Metal texture-readback staging allocation failed",
                    )
                    .at("MetalNativeEncoder")
                })?;
            let texture = metal_texture(src)?;
            let encoder = ensure_blit(command_buffer, blit)?;
            let copies = if is_3d { 1 } else { subresource.layer_count };
            for layer in 0..copies {
                let offset = image_stride.checked_mul(u64::from(layer)).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "Metal texture readback layer offset overflows",
                    )
                    .at("MetalNativeEncoder")
                })?;
                unsafe {
                    encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                        &texture.raw,
                        if is_3d { 0 } else { (subresource.base_layer + layer) as usize },
                        subresource.mip_level as usize,
                        origin(*source_origin),
                        size(*extent),
                        &staging,
                        offset as usize,
                        bytes_per_row as usize,
                        image_stride as usize,
                    );
                }
            }
            retained.push(MetalPendingReadback {
                ticket: ticket.clone(),
                staging,
                byte_len,
                layout: Some(ReadbackTexelLayout {
                    bytes_per_row: u32::try_from(bytes_per_row).map_err(|_| {
                        RhiError::new(
                            RhiErrorKind::OutOfMemory,
                            "Metal texture readback row pitch exceeds public layout",
                        )
                        .at("MetalNativeEncoder")
                    })?,
                    rows_per_image,
                    total_size,
                }),
            });
        }
        _ => {
            return Err(
                RhiError::new(RhiErrorKind::Unsupported, "unknown readback request")
                    .at("MetalNativeEncoder"),
            );
        }
    }
    Ok(())
}

fn align_readback_row(value: u64) -> RhiResult<u64> {
    value.checked_add(255).map(|row| row & !255).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "Metal texture readback row pitch alignment overflows",
        )
        .at("MetalNativeEncoder")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::format::TextureFormat;
    use crate::api::identity::ObjectId;
    use crate::api::resource::{Extent3d, HostTexelLayout};

    #[test]
    fn repack_r8_three_texel_rows_adds_only_native_padding() {
        let upload = repack_texture_upload_parts(
            TextureFormat::R8Unorm,
            Extent3d::d2(3, 2),
            TextureDimension::D2,
            1,
            HostTexelLayout {
                bytes_per_row: 3,
                rows_per_image: 2,
            },
            &[1, 2, 3, 4, 5, 6],
        )
        .unwrap();
        assert_eq!(upload.bytes_per_row, 4);
        assert_eq!(upload.bytes_per_image, 8);
        assert_eq!(upload.bytes, vec![1, 2, 3, 0, 4, 5, 6, 0]);
    }

    #[test]
    fn repack_bc_rows_copy_blocks_not_caller_padding() {
        let upload = repack_texture_upload_parts(
            TextureFormat::Bc1RgbaUnorm,
            Extent3d::d2(4, 8),
            TextureDimension::D2,
            1,
            HostTexelLayout {
                bytes_per_row: 16,
                rows_per_image: 2,
            },
            &[
                1, 2, 3, 4, 5, 6, 7, 8, 99, 99, 99, 99, 99, 99, 99, 99, 9, 10, 11, 12, 13, 14, 15,
                16, 88, 88, 88, 88, 88, 88, 88, 88,
            ],
        )
        .unwrap();
        assert_eq!(upload.bytes_per_row, 8);
        assert_eq!(upload.bytes_per_image, 16);
        assert_eq!(
            upload.bytes,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    #[test]
    fn repack_astc_uses_block_rows_and_3d_image_stride() {
        let source = (0_u8..128).collect::<Vec<_>>();
        let upload = repack_texture_upload_parts(
            TextureFormat::Astc4x4Unorm,
            Extent3d::d3(5, 5, 2),
            TextureDimension::D3,
            1,
            HostTexelLayout {
                bytes_per_row: 32,
                rows_per_image: 2,
            },
            &source,
        )
        .unwrap();
        assert_eq!(upload.bytes_per_row, 32);
        assert_eq!(upload.bytes_per_image, 64);
        assert_eq!(upload.bytes, source);
    }

    #[test]
    fn absent_base_instance_selector_allows_only_zero_base_fallback() {
        validate_base_vertex_instance_selector(false, false, 0, 0).unwrap();
        validate_base_vertex_instance_selector(false, true, 0, 0).unwrap();
        assert_eq!(
            validate_base_vertex_instance_selector(false, false, 0, 1)
                .unwrap_err()
                .kind(),
            RhiErrorKind::Unsupported
        );
        assert_eq!(
            validate_base_vertex_instance_selector(false, true, -1, 0)
                .unwrap_err()
                .kind(),
            RhiErrorKind::Unsupported
        );
    }

    #[test]
    fn visibility_query_sequence_preserves_begin_end_order_and_offsets() {
        let first = ObjectId::next();
        let second = ObjectId::next();
        let mut sequence = OcclusionQuerySequence::default();
        assert_eq!(sequence.begin(first, 3).unwrap(), 0);
        assert!(sequence.is_active());
        assert!(sequence.begin(second, 0).is_err(), "nested query must fail");
        assert!(sequence.end(first, 4).is_err(), "end must match its begin");

        // A rejected command buffer will be discarded after a mismatched end.
        // Start a fresh sequence to pin the accepted ordering and dense offsets.
        let mut sequence = OcclusionQuerySequence::default();
        assert_eq!(sequence.begin(first, 3).unwrap(), 0);
        assert_eq!(sequence.end(first, 3).unwrap(), 0);
        assert_eq!(sequence.begin(second, 0).unwrap(), 8);
        assert_eq!(sequence.end(second, 0).unwrap(), 8);
        assert!(!sequence.is_active());
    }
}
