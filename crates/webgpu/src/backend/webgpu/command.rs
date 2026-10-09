//! Transactional WebGPU command submission.
//!
//! Browser WebGPU has no fence object.  `queue.onSubmittedWorkDone()` is its
//! completion primitive, so a submitted RHI serial owns the corresponding
//! promise. Native command buffers are fully encoded before this spine sees
//! them; it only submits those finished buffers and observes completion.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::task::Waker;

use js_sys::{Array, Function, Object, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue};

use crate::api::command::copy::{BufferTextureCopy, TextureCopy};
use crate::api::command::record::RasterBegin;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::{block_extent, logical_bytes_per_block};
use crate::api::query::QuerySet;
use crate::api::resource::{ReadbackRequest, ReadbackTexelLayout};
use crate::api::submission::backend::{SubmissionOutcome, SubmissionRequest};
use crate::api::submission::{CompletionFailure, CompletionState};

use super::js;
use super::registry::{
    self, PromisePoll, WebGpuDriver, WebGpuObjectId, WebGpuRegistration, WebGpuRequestId,
};
use super::resource::WebGpuTextureView;
use super::resource::{WebGpuBuffer, WebGpuQuerySet, WebGpuTexture};

#[derive(Clone, Debug)]
enum SerialState {
    Pending(WebGpuRequestId),
    /// GPU queue work ended but an attached readback mapping lease has not
    /// published CPU bytes yet.
    AwaitReadbacks(WebGpuRequestId),
    Complete,
    Failed(String),
}

#[derive(Default)]
struct State {
    issued: u64,
    serials: BTreeMap<u64, SerialState>,
    readbacks: Vec<PendingReadback>,
}

/// Registry-only state for one submitted map-read staging allocation.  Keeping
/// an object ID rather than a `JsValue` is what lets `WebGpuCommandSpine`
/// remain `Send + Sync` even though browser handles themselves are affine.
pub(crate) struct PendingReadback {
    pub(crate) ticket: crate::api::resource::ReadbackTicket,
    pub(crate) staging: WebGpuObjectId,
    pub(crate) bytes: u64,
    pub(crate) layout: Option<ReadbackTexelLayout>,
    pub(crate) map: Option<WebGpuRequestId>,
    /// The submitted-work promise for this plan. A plan completion cannot be
    /// published Complete until every ticket attached to it has been copied
    /// into CPU-owned bytes and published.
    pub(crate) completion: Option<WebGpuRequestId>,
}

/// The browser-side execution timeline for one WebGPU device registration.
pub(crate) struct WebGpuCommandSpine {
    driver: WebGpuDriver,
    state: Mutex<State>,
}

impl WebGpuCommandSpine {
    pub(crate) fn new(driver: WebGpuDriver) -> Self {
        Self {
            driver,
            state: Mutex::new(State::default()),
        }
    }

    fn registration(&self) -> WebGpuRegistration {
        self.driver.registration()
    }

    pub(crate) fn submit(&self, request: &SubmissionRequest<'_>) -> RhiResult<SubmissionOutcome> {
        if registry::device_status(self.registration())
            != Some(crate::api::platform::DeviceStatus::Active)
        {
            return Err(lost("WebGpuCommandSpine::submit"));
        }
        if request.batches.is_empty() {
            let issued = self.state.lock().unwrap_or_else(|p| p.into_inner()).issued;
            return Ok(SubmissionOutcome {
                completion: issued,
                points: Vec::new(),
            });
        }

        // Each work item has already closed its native GPUCommandBuffer.  Check
        // every object before touching the queue so a wrong backend or an
        // incomplete direct recording still means zero accepted work.
        let mut buffers = Vec::new();
        for batch in request.batches {
            for work in &batch.work {
                let buffer = work
                    .native()
                    .as_any()
                    .downcast_ref::<super::native::WebGpuCommandBuffer>()
                    .ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::WrongDevice,
                            "submission contains a command buffer from another backend",
                        )
                        .at("WebGpuCommandSpine::submit")
                    })?;
                if buffer.registration() != self.registration() {
                    return Err(RhiError::new(
                        RhiErrorKind::WrongDevice,
                        "submission contains a command buffer from another WebGPU device",
                    )
                    .at("WebGpuCommandSpine::submit"));
                }
                buffers.push(buffer);
            }
        }
        for present in request.presents {
            super::presentation::frame_view(&present.attachment)?;
        }

        // Queue submission is the only browser operation performed here.  All
        // draw, dispatch, copy, upload, and readback encoding occurred while
        // the caller owned its recorder.
        let native_submission = self.submit_native(&buffers, request);
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let first = state.issued + 1;
        let last = first + request.batches.len() as u64 - 1;
        state.issued = last;
        match native_submission {
            Ok((promise, mut readbacks)) => {
                let request_id = registry::start_device_promise(self.registration(), promise);
                for serial in first..=last {
                    state
                        .serials
                        .insert(serial, SerialState::Pending(request_id));
                }
                for readback in &mut readbacks {
                    readback.completion = Some(request_id);
                }
                state.readbacks.extend(readbacks);
            }
            Err(message) => {
                for present in request.presents {
                    let outcome = if let Some(info) = registry::device_loss(self.registration()) {
                        crate::api::presentation::PresentState::DeviceLost(info)
                    } else {
                        crate::api::presentation::PresentState::Failed(
                            crate::api::presentation::PresentFailure::new(message.clone()),
                        )
                    };
                    present
                        .attachment
                        .terminate_present(present.receipt, outcome);
                }
                for serial in first..=last {
                    state
                        .serials
                        .insert(serial, SerialState::Failed(message.clone()));
                }
            }
        }
        Ok(SubmissionOutcome {
            completion: last,
            points: request
                .batches
                .iter()
                .enumerate()
                .map(|(n, batch)| (batch.point, first + n as u64))
                .collect(),
        })
    }

    fn submit_native(
        &self,
        buffers: &[&super::native::WebGpuCommandBuffer],
        request: &SubmissionRequest<'_>,
    ) -> Result<(Promise, Vec<PendingReadback>), String> {
        let queue =
            registry::with_device_handles(self.registration(), |handles| handles.queue.clone())
                .ok_or_else(|| "WebGPU device registration was retired".to_owned())?;
        let list = Array::new();
        let mut readbacks = Vec::new();
        for buffer in buffers {
            let native = buffer
                .take()
                .ok_or_else(|| "finished WebGPU command buffer was already consumed".to_owned())?;
            list.push(&native);
            readbacks.extend(buffer.take_readbacks());
        }
        let list: JsValue = list.into();
        call1(&queue, "submit", &list)?;
        for present in request.presents {
            present.attachment.present(present.receipt);
        }
        for pending in &mut readbacks {
            let staging = registry::with_object(self.registration(), pending.staging, Clone::clone)
                .ok_or_else(|| {
                    "readback staging allocation was retired before mapAsync".to_owned()
                })?;
            let promise = call3_value(&staging, "mapAsync", &num(1), &num(0), &num(pending.bytes))?;
            pending.map = Some(registry::start_device_promise(
                self.registration(),
                Promise::from(promise),
            ));
        }
        let completion = call0(&queue, "onSubmittedWorkDone")
            .map(Promise::from)
            .unwrap_or_else(|message| Promise::reject(&JsValue::from_str(&message)));
        Ok((completion, readbacks))
    }

    pub(crate) fn poll(&self) -> RhiResult<()> {
        self.advance();
        Ok(())
    }
    pub(crate) fn completion(&self, serial: u64) -> CompletionState {
        self.advance();
        self.state(serial, None)
    }
    pub(crate) fn completion_or_register_waker(
        &self,
        serial: u64,
        waker: &Waker,
    ) -> CompletionState {
        self.advance();
        self.state(serial, Some(waker))
    }

    fn advance(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        // Publish readback bytes before completion. `mapAsync` resolves only
        // after the copy that precedes it, so this may finish before the queue
        // promise. If it does not, the completion below remains Pending until
        // the ticket reaches its terminal state.
        let mut failed_readbacks = Vec::new();
        let mut still_pending = Vec::with_capacity(state.readbacks.len());
        for pending in state.readbacks.drain(..) {
            let map = pending
                .map
                .expect("mapAsync is started after queue submission");
            match registry::try_take_settled_promise(map) {
                PromisePoll::Pending => still_pending.push(pending),
                PromisePoll::Failed(_) => {
                    if registry::device_loss(self.registration()).is_some() {
                        pending
                            .ticket
                            .set_status(crate::api::resource::ReadbackStatus::DeviceLost);
                    } else {
                        pending
                            .ticket
                            .set_status(crate::api::resource::ReadbackStatus::Failed);
                        if let Some(completion) = pending.completion {
                            failed_readbacks.push(completion);
                        }
                    }
                    let _ = registry::remove_object(self.registration(), pending.staging);
                }
                PromisePoll::Ready(_) => {
                    let bytes =
                        registry::with_object(self.registration(), pending.staging, |value| {
                            read_mapped_bytes(value, pending.bytes)
                        })
                        .flatten();
                    match bytes {
                        Some(bytes) => pending.ticket.publish(bytes, pending.layout),
                        None => {
                            pending
                                .ticket
                                .set_status(crate::api::resource::ReadbackStatus::Failed);
                            if let Some(completion) = pending.completion {
                                failed_readbacks.push(completion);
                            }
                        }
                    }
                    let _ =
                        registry::with_object(self.registration(), pending.staging, unmap_buffer);
                    let _ = registry::remove_object(self.registration(), pending.staging);
                }
            }
        }
        state.readbacks = still_pending;
        // Each plan shares one promise. Poll it only once, then fan the result
        // out to each point carrying that serial; `poll_promise` consumes a
        // settled request, so collecting IDs first is essential.
        let mut ids = Vec::new();
        for id in state.serials.values().filter_map(|entry| match entry {
            SerialState::Pending(id) => Some(*id),
            _ => None,
        }) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        for id in ids {
            match registry::try_take_settled_promise(id) {
                PromisePoll::Pending => {}
                PromisePoll::Ready(_) => {
                    if !state
                        .readbacks
                        .iter()
                        .any(|readback| readback.completion == Some(id))
                    {
                        for entry in state.serials.values_mut() {
                            if matches!(entry, SerialState::Pending(current) if *current == id) {
                                *entry = SerialState::Complete;
                            }
                        }
                    } else {
                        for entry in state.serials.values_mut() {
                            if matches!(entry, SerialState::Pending(current) if *current == id) {
                                *entry = SerialState::AwaitReadbacks(id);
                            }
                        }
                    }
                }
                PromisePoll::Failed(message) => {
                    for entry in state.serials.values_mut() {
                        if matches!(entry, SerialState::Pending(current) if *current == id) {
                            *entry = SerialState::Failed(message.clone());
                        }
                    }
                }
            }
        }
        for id in failed_readbacks {
            for entry in state.serials.values_mut() {
                if matches!(entry, SerialState::Pending(current) | SerialState::AwaitReadbacks(current) if *current == id)
                {
                    *entry = SerialState::Failed("WebGPU readback mapping failed".into());
                }
            }
        }
        let published_readback_completions: Vec<_> = state
            .serials
            .values()
            .filter_map(|entry| match entry {
                SerialState::AwaitReadbacks(id)
                    if !state
                        .readbacks
                        .iter()
                        .any(|readback| readback.completion == Some(*id)) =>
                {
                    Some(*id)
                }
                _ => None,
            })
            .collect();
        for id in published_readback_completions {
            for entry in state.serials.values_mut() {
                if matches!(entry, SerialState::AwaitReadbacks(current) if *current == id) {
                    *entry = SerialState::Complete;
                }
            }
        }
        // Prune completed serials. They stay answerable as `Complete` because
        // `state` answers any serial at or below `issued` that is absent from the
        // map as `Complete` (the `None` arm below), so a terminal row per
        // submission would be bookkeeping that grows without bound for no caller
        // benefit. Failed serials are *kept* so they keep answering `Failed`
        // rather than falling through to the `Complete` fallback; a row is only
        // pruned once it is genuinely terminal-complete.
        state
            .serials
            .retain(|_, entry| !matches!(entry, SerialState::Complete));
    }

    fn state(&self, serial: u64, waker: Option<&Waker>) -> CompletionState {
        if let Some(info) = registry::device_loss(self.registration()) {
            return CompletionState::DeviceLost(info);
        }
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(SerialState::AwaitReadbacks(id)) = state.serials.get(&serial) {
            let maps = state
                .readbacks
                .iter()
                .filter(|readback| readback.completion == Some(*id))
                .filter_map(|readback| readback.map)
                .collect::<Vec<_>>();
            // A browser promise callback may wake an executor synchronously.
            // Never retain the spine mutex across that foreign wake boundary.
            drop(state);
            if let Some(waker) = waker {
                for map in maps {
                    registry::register_promise_waker(map, waker);
                }
            }
            return CompletionState::Pending;
        }
        match state.serials.get(&serial) {
            Some(SerialState::Complete) => CompletionState::Complete,
            Some(SerialState::Failed(message)) => {
                CompletionState::Failed(CompletionFailure::new(message.clone()))
            }
            Some(SerialState::Pending(id)) => {
                if let Some(waker) = waker {
                    let _ = registry::poll_promise(*id, waker);
                }
                CompletionState::Pending
            }
            Some(SerialState::AwaitReadbacks(_)) => unreachable!("handled above"),
            None if serial <= state.issued => CompletionState::Complete,
            None => {
                CompletionState::Failed(CompletionFailure::new("unknown WebGPU completion serial"))
            }
        }
    }
}

pub(crate) fn buffer<'a>(
    value: &'a dyn crate::api::resource::backend::BufferBackend,
    registration: WebGpuRegistration,
    what: &'static str,
) -> RhiResult<&'a WebGpuBuffer> {
    let value = value
        .as_any()
        .downcast_ref::<WebGpuBuffer>()
        .ok_or_else(|| unsupported(what))?;
    (value.registration() == registration)
        .then_some(value)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                format!("{what} belongs to another WebGPU device"),
            )
        })
}
pub(crate) fn texture<'a>(
    value: &'a dyn crate::api::resource::backend::TextureBackend,
    registration: WebGpuRegistration,
    what: &'static str,
) -> RhiResult<&'a WebGpuTexture> {
    let value = value
        .as_any()
        .downcast_ref::<WebGpuTexture>()
        .ok_or_else(|| unsupported(what))?;
    (value.registration() == registration)
        .then_some(value)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                format!("{what} belongs to another WebGPU device"),
            )
        })
}
pub(crate) fn texture_view<'a>(
    value: &'a dyn crate::api::resource::backend::TextureViewBackend,
    registration: WebGpuRegistration,
    what: &'static str,
) -> RhiResult<&'a WebGpuTextureView> {
    let value = value
        .as_any()
        .downcast_ref::<WebGpuTextureView>()
        .ok_or_else(|| unsupported(what))?;
    (value.registration() == registration)
        .then_some(value)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                format!("{what} belongs to another WebGPU device"),
            )
        })
}
pub(crate) fn query_set<'a>(
    value: &'a QuerySet,
    registration: WebGpuRegistration,
    what: &'static str,
) -> RhiResult<&'a WebGpuQuerySet> {
    let value = value
        .native()
        .as_any()
        .downcast_ref::<WebGpuQuerySet>()
        .ok_or_else(|| unsupported(what))?;
    (value.registration() == registration)
        .then_some(value)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                format!("{what} belongs to another WebGPU device"),
            )
        })
}
pub(crate) fn object<T>(value: &T) -> Option<JsValue>
where
    T: Registered,
{
    registry::with_object(value.registration(), value.object(), Clone::clone)
}
pub(crate) trait Registered {
    fn registration(&self) -> WebGpuRegistration;
    fn object(&self) -> super::registry::WebGpuObjectId;
}
impl Registered for WebGpuBuffer {
    fn registration(&self) -> WebGpuRegistration {
        self.registration()
    }
    fn object(&self) -> super::registry::WebGpuObjectId {
        self.object()
    }
}
impl Registered for WebGpuTexture {
    fn registration(&self) -> WebGpuRegistration {
        self.registration()
    }
    fn object(&self) -> super::registry::WebGpuObjectId {
        self.object()
    }
}
impl Registered for WebGpuTextureView {
    fn registration(&self) -> WebGpuRegistration {
        self.registration()
    }
    fn object(&self) -> super::registry::WebGpuObjectId {
        self.object()
    }
}
impl Registered for WebGpuQuerySet {
    fn registration(&self) -> WebGpuRegistration {
        self.registration()
    }
    fn object(&self) -> super::registry::WebGpuObjectId {
        self.object()
    }
}
fn buffer_registration(value: &crate::api::resource::Buffer) -> RhiResult<WebGpuRegistration> {
    value
        .native()
        .as_any()
        .downcast_ref::<WebGpuBuffer>()
        .map(WebGpuBuffer::registration)
        .ok_or_else(|| unsupported("buffer"))
}
fn texture_registration(value: &crate::api::resource::Texture) -> RhiResult<WebGpuRegistration> {
    value
        .native()
        .as_any()
        .downcast_ref::<WebGpuTexture>()
        .map(WebGpuTexture::registration)
        .ok_or_else(|| unsupported("texture"))
}

pub(crate) fn lower_buffer_texture_copy(
    encoder: &JsValue,
    copy: &BufferTextureCopy,
    to_texture: bool,
) -> Result<(), String> {
    let registration = buffer_registration(&copy.buffer).map_err(|e| e.to_string())?;
    if texture_registration(&copy.texture).map_err(|e| e.to_string())? != registration {
        return Err("buffer/texture copy crosses WebGPU devices".into());
    }
    let buffer_value = object(
        buffer(copy.buffer.native(), registration, "copy buffer").map_err(|e| e.to_string())?,
    )
    .ok_or_else(|| "copy buffer retired".to_owned())?;
    let texture_value = object(
        texture(copy.texture.native(), registration, "copy texture").map_err(|e| e.to_string())?,
    )
    .ok_or_else(|| "copy texture retired".to_owned())?;
    let buffer_desc = Object::new();
    set(&buffer_desc, "buffer", &buffer_value)?;
    set(&buffer_desc, "offset", &num(copy.buffer_offset))?;
    set(&buffer_desc, "bytesPerRow", &num(copy.bytes_per_row as u64))?;
    set(
        &buffer_desc,
        "rowsPerImage",
        &num(copy.rows_per_image as u64),
    )?;
    let texture_desc = texture_copy_desc(
        &texture_value,
        copy.texture_subresource.mip_level,
        copy.texture_subresource.base_layer,
        copy.texture_subresource.aspect,
        copy.texture_origin,
    )?;
    let extent = extent(copy.extent);
    if to_texture {
        call3(
            encoder,
            "copyBufferToTexture",
            &buffer_desc.into(),
            &texture_desc.into(),
            &extent,
        )
    } else {
        call3(
            encoder,
            "copyTextureToBuffer",
            &texture_desc.into(),
            &buffer_desc.into(),
            &extent,
        )
    }
}

pub(crate) fn lower_texture_copy(encoder: &JsValue, copy: &TextureCopy) -> Result<(), String> {
    let registration = texture_registration(&copy.src).map_err(|e| e.to_string())?;
    if texture_registration(&copy.dst).map_err(|e| e.to_string())? != registration {
        return Err("texture copy crosses WebGPU devices".into());
    }
    let src =
        object(texture(copy.src.native(), registration, "copy source").map_err(|e| e.to_string())?)
            .ok_or_else(|| "copy source retired".to_owned())?;
    let dst = object(
        texture(copy.dst.native(), registration, "copy destination").map_err(|e| e.to_string())?,
    )
    .ok_or_else(|| "copy destination retired".to_owned())?;
    let src = texture_copy_desc(
        &src,
        copy.src_subresource.mip_level,
        copy.src_subresource.base_layer,
        copy.src_subresource.aspect,
        copy.src_origin,
    )?;
    let dst = texture_copy_desc(
        &dst,
        copy.dst_subresource.mip_level,
        copy.dst_subresource.base_layer,
        copy.dst_subresource.aspect,
        copy.dst_origin,
    )?;
    call3(
        encoder,
        "copyTextureToTexture",
        &src.into(),
        &dst.into(),
        &extent(copy.extent),
    )
}

pub(crate) fn lower_readback(
    encoder: &JsValue,
    ticket: &crate::api::resource::ReadbackTicket,
    registration: WebGpuRegistration,
    out: &mut Vec<PendingReadback>,
) -> Result<(), String> {
    if let ReadbackRequest::Texture { .. } = ticket.request() {
        return lower_texture_readback(encoder, ticket, registration, out);
    }
    let ReadbackRequest::Buffer { src, range, .. } = ticket.request() else {
        unreachable!()
    };
    let source =
        object(buffer(src.native(), registration, "readback source").map_err(|e| e.to_string())?)
            .ok_or_else(|| "readback source retired".to_owned())?;
    let device = registry::with_device_handles(registration, |h| h.device.clone())
        .ok_or_else(|| "WebGPU device retired".to_owned())?;
    let desc = Object::new();
    set(&desc, "size", &num(range.size))?;
    set(&desc, "usage", &num(1 | 8))?;
    let staging = call1(&device, "createBuffer", &desc.into())?;
    let staging_id = registry::insert_object(registration, staging.clone())
        .ok_or_else(|| "WebGPU device retired while retaining readback staging".to_owned())?;
    call5(
        encoder,
        "copyBufferToBuffer",
        &source,
        &num(range.offset),
        &staging,
        &num(0),
        &num(range.size),
    )?;
    out.push(PendingReadback {
        ticket: ticket.clone(),
        staging: staging_id,
        bytes: range.size,
        layout: None,
        map: None,
        completion: None,
    });
    Ok(())
}

fn texture_readback_layout(
    ticket: &crate::api::resource::ReadbackTicket,
) -> RhiResult<ReadbackTexelLayout> {
    let ReadbackRequest::Texture {
        src,
        subresource,
        extent,
        ..
    } = ticket.request()
    else {
        unreachable!();
    };
    let bytes = logical_bytes_per_block(src.descriptor().format).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            "texture format has no WebGPU copy footprint",
        )
    })?;
    let (block_width, block_height) = block_extent(src.descriptor().format);
    let columns = extent.width.div_ceil(block_width);
    let rows = extent.height.div_ceil(block_height);
    let unaligned = columns.checked_mul(bytes).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "texture readback row size overflow",
        )
    })?;
    // A private staging buffer is a GPU copy buffer, so its rows obey WebGPU's
    // 256-byte footprint alignment; the public ticket reports that padding.
    let bytes_per_row = unaligned.checked_add(255).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "texture readback row alignment overflow",
        )
    })? / 256
        * 256;
    let images = if matches!(
        src.descriptor().dimension,
        crate::api::resource::TextureDimension::D3
    ) {
        extent.depth
    } else {
        subresource.layer_count
    };
    let total_size = u64::from(bytes_per_row)
        .checked_mul(u64::from(rows))
        .and_then(|value| value.checked_mul(u64::from(images)))
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture readback staging size overflow",
            )
        })?;
    Ok(ReadbackTexelLayout {
        bytes_per_row,
        rows_per_image: rows,
        total_size,
    })
}

pub(crate) fn lower_texture_readback(
    encoder: &JsValue,
    ticket: &crate::api::resource::ReadbackTicket,
    registration: WebGpuRegistration,
    out: &mut Vec<PendingReadback>,
) -> Result<(), String> {
    let ReadbackRequest::Texture {
        src,
        subresource,
        origin,
        extent,
        ..
    } = ticket.request()
    else {
        unreachable!()
    };
    let layout = texture_readback_layout(ticket).map_err(|error| error.to_string())?;
    let source = object(
        texture(src.native(), registration, "readback texture source")
            .map_err(|error| error.to_string())?,
    )
    .ok_or_else(|| "readback texture source retired".to_owned())?;
    let device = registry::with_device_handles(registration, |h| h.device.clone())
        .ok_or_else(|| "WebGPU device retired".to_owned())?;
    let descriptor = Object::new();
    set(&descriptor, "size", &num(layout.total_size))?;
    set(&descriptor, "usage", &num(1 | 8))?;
    let staging = call1(&device, "createBuffer", &descriptor.into())?;
    let staging_id = registry::insert_object(registration, staging.clone()).ok_or_else(|| {
        "WebGPU device retired while retaining texture readback staging".to_owned()
    })?;
    let source_desc = Object::new();
    set(&source_desc, "texture", &source)?;
    set(&source_desc, "mipLevel", &num(subresource.mip_level as u64))?;
    let aspect = match subresource.aspect {
        crate::api::resource::TextureAspect::Color => "all",
        crate::api::resource::TextureAspect::Depth => "depth-only",
        crate::api::resource::TextureAspect::Stencil => "stencil-only",
        crate::api::resource::TextureAspect::Plane0
        | crate::api::resource::TextureAspect::Plane1
        | crate::api::resource::TextureAspect::Plane2 => {
            return Err("WebGPU has no portable multi-planar texture readback lowering".into());
        }
        _ => return Err("unknown texture aspect".into()),
    };
    set(&source_desc, "aspect", &JsValue::from_str(aspect))?;
    let source_origin = Object::new();
    set(&source_origin, "x", &num(origin.x as u64))?;
    set(&source_origin, "y", &num(origin.y as u64))?;
    let source_z = if matches!(
        src.descriptor().dimension,
        crate::api::resource::TextureDimension::D3
    ) {
        origin.z
    } else {
        subresource.base_layer
    };
    set(&source_origin, "z", &num(source_z as u64))?;
    set(&source_desc, "origin", &source_origin.into())?;
    let destination = Object::new();
    set(&destination, "buffer", &staging)?;
    set(&destination, "offset", &num(0))?;
    set(
        &destination,
        "bytesPerRow",
        &num(layout.bytes_per_row as u64),
    )?;
    set(
        &destination,
        "rowsPerImage",
        &num(layout.rows_per_image as u64),
    )?;
    let copy_extent = Object::new();
    set(&copy_extent, "width", &num(extent.width as u64))?;
    set(&copy_extent, "height", &num(extent.height as u64))?;
    set(
        &copy_extent,
        "depthOrArrayLayers",
        &num(
            if matches!(
                src.descriptor().dimension,
                crate::api::resource::TextureDimension::D3
            ) {
                extent.depth as u64
            } else {
                subresource.layer_count as u64
            },
        ),
    )?;
    call3(
        encoder,
        "copyTextureToBuffer",
        &source_desc.into(),
        &destination.into(),
        &copy_extent.into(),
    )?;
    out.push(PendingReadback {
        ticket: ticket.clone(),
        staging: staging_id,
        bytes: layout.total_size,
        layout: Some(layout),
        map: None,
        completion: None,
    });
    Ok(())
}
pub(crate) fn begin_raster(
    encoder: &JsValue,
    begin: &RasterBegin,
    registration: WebGpuRegistration,
) -> Result<JsValue, String> {
    let descriptor = Object::new();
    let colors = Array::new();
    // WebGPU's colorAttachments array permits null holes, exactly matching the
    // portable sparse MRT locations rather than compacting their indices.
    let mut next = 0u32;
    for (location, attachment) in &begin.colors {
        while next < *location {
            colors.push(&JsValue::NULL);
            next += 1;
        }
        let item = Object::new();
        let view = color_view(&attachment.view, registration).map_err(|e| e.to_string())?;
        set(&item, "view", &view)?;
        set(
            &item,
            "loadOp",
            &JsValue::from_str(match attachment.load {
                crate::api::command::LoadOp::Load => "load",
                crate::api::command::LoadOp::Clear(_) => "clear",
            }),
        )?;
        set(
            &item,
            "storeOp",
            &JsValue::from_str(match attachment.store {
                crate::api::command::StoreOp::Store => "store",
                crate::api::command::StoreOp::Discard => "discard",
            }),
        )?;
        if let crate::api::command::LoadOp::Clear(value) = attachment.load {
            set(&item, "clearValue", &clear_color(value).into())?;
        }
        if let Some(resolve) = &attachment.resolve {
            set(
                &item,
                "resolveTarget",
                &color_view(resolve, registration).map_err(|e| e.to_string())?,
            )?;
        }
        colors.push(&item);
        next += 1;
    }
    set(&descriptor, "colorAttachments", &colors.into())?;
    if let Some(depth) = &begin.depth_stencil {
        set(
            &descriptor,
            "depthStencilAttachment",
            &depth_attachment(depth, registration)
                .map_err(|e| e.to_string())?
                .into(),
        )?;
    }
    if let Some(query) = &begin.occlusion_query_set {
        let native_set = object(
            query_set(query, registration, "raster occlusion query set")
                .map_err(|error| error.to_string())?,
        )
        .ok_or_else(|| "raster occlusion query set was retired".to_owned())?;
        set(&descriptor, "occlusionQuerySet", &native_set)?;
    }
    call1(encoder, "beginRenderPass", &descriptor.into())
}

/// Opens a native WebGPU render pass for the direct encoder.  The direct
/// encoder owns its pass lifetime; this helper only translates the portable
/// attachment descriptor into the browser descriptor.
pub(crate) fn native_begin_raster(
    encoder: &JsValue,
    begin: &RasterBegin,
    registration: WebGpuRegistration,
) -> RhiResult<JsValue> {
    begin_raster(encoder, begin, registration).map_err(|message| {
        RhiError::new(RhiErrorKind::BackendFailure, message).at("WebGpuNativeEncoder::raster_begin")
    })
}

fn color_view(
    view: &crate::api::command::attachment::ColorAttachmentView,
    registration: WebGpuRegistration,
) -> RhiResult<JsValue> {
    match view {
        crate::api::command::attachment::ColorAttachmentView::Texture(view) => object(
            texture_view(view.native(), registration, "color attachment")?,
        )
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::DeviceLost,
                "color attachment view was retired",
            )
        }),
        crate::api::command::attachment::ColorAttachmentView::Frame(frame) => {
            super::presentation::frame_view(frame)
        }
        _ => Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "unknown color attachment view",
        )),
    }
}
fn depth_attachment(
    value: &crate::api::command::attachment::DepthStencilAttachment,
    registration: WebGpuRegistration,
) -> RhiResult<Object> {
    let out = Object::new();
    set_rhi(
        &out,
        "view",
        object(texture_view(
            value.view.native(),
            registration,
            "depth/stencil attachment",
        )?)
        .ok_or_else(|| RhiError::new(RhiErrorKind::DeviceLost, "depth/stencil view retired"))?,
    )?;
    if let Some(depth) = value.depth {
        match depth {
            crate::api::command::attachment::DepthAttachmentMode::ReadOnly => {
                set_rhi(&out, "depthReadOnly", JsValue::TRUE)?;
            }
            crate::api::command::attachment::DepthAttachmentMode::ReadWrite { load, store } => {
                set_rhi(
                    &out,
                    "depthLoadOp",
                    JsValue::from_str(match load {
                        crate::api::command::LoadOp::Load => "load",
                        crate::api::command::LoadOp::Clear(_) => "clear",
                    }),
                )?;
                set_rhi(
                    &out,
                    "depthStoreOp",
                    JsValue::from_str(match store {
                        crate::api::command::StoreOp::Store => "store",
                        crate::api::command::StoreOp::Discard => "discard",
                    }),
                )?;
                if let crate::api::command::LoadOp::Clear(v) = load {
                    set_rhi(&out, "depthClearValue", JsValue::from_f64(v as f64))?;
                }
            }
            _ => {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    "unknown depth attachment mode",
                ));
            }
        }
    }
    if let Some(stencil) = value.stencil {
        match stencil {
            crate::api::command::attachment::StencilAttachmentMode::ReadOnly => {
                set_rhi(&out, "stencilReadOnly", JsValue::TRUE)?;
            }
            crate::api::command::attachment::StencilAttachmentMode::ReadWrite { load, store } => {
                set_rhi(
                    &out,
                    "stencilLoadOp",
                    JsValue::from_str(match load {
                        crate::api::command::LoadOp::Load => "load",
                        crate::api::command::LoadOp::Clear(_) => "clear",
                    }),
                )?;
                set_rhi(
                    &out,
                    "stencilStoreOp",
                    JsValue::from_str(match store {
                        crate::api::command::StoreOp::Store => "store",
                        crate::api::command::StoreOp::Discard => "discard",
                    }),
                )?;
                if let crate::api::command::LoadOp::Clear(v) = load {
                    set_rhi(&out, "stencilClearValue", num(v as u64))?;
                }
            }
            _ => {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    "unknown stencil attachment mode",
                ));
            }
        }
    }
    Ok(out)
}

fn texture_copy_desc(
    texture: &JsValue,
    mip: u32,
    layer: u32,
    aspect: crate::api::resource::TextureAspect,
    mut origin: crate::api::resource::Origin3d,
) -> Result<Object, String> {
    // WebGPU represents the base array layer as the z coordinate of an
    // `ImageCopyTexture` origin.  Portable 3D copies carry layer zero, so this
    // addition is also the exact 3D answer.
    origin.z = origin
        .z
        .checked_add(layer)
        .ok_or_else(|| "texture array layer plus origin overflows".to_owned())?;
    let out = Object::new();
    set(&out, "texture", texture)?;
    set(&out, "mipLevel", &num(mip as u64))?;
    set(&out, "origin", &origin_obj(origin).into())?;
    let aspect = match aspect {
        crate::api::resource::TextureAspect::Color => "all",
        crate::api::resource::TextureAspect::Depth => "depth-only",
        crate::api::resource::TextureAspect::Stencil => "stencil-only",
        _ => return Err("WebGPU does not lower multi-planar texture copies".into()),
    };
    set(&out, "aspect", &JsValue::from_str(aspect))?;
    Ok(out)
}
fn origin_obj(origin: crate::api::resource::Origin3d) -> Object {
    let out = Object::new();
    let _ = set(&out, "x", &num(origin.x as u64));
    let _ = set(&out, "y", &num(origin.y as u64));
    let _ = set(&out, "z", &num(origin.z as u64));
    out
}
fn extent(value: crate::api::resource::Extent3d) -> JsValue {
    let out = Object::new();
    let _ = set(&out, "width", &num(value.width as u64));
    let _ = set(&out, "height", &num(value.height as u64));
    let _ = set(&out, "depthOrArrayLayers", &num(value.depth as u64));
    out.into()
}

fn lost(at: &'static str) -> RhiError {
    RhiError::new(RhiErrorKind::DeviceLost, "the WebGPU device is lost").at(at)
}
fn unsupported(what: &'static str) -> RhiError {
    RhiError::new(
        RhiErrorKind::Unsupported,
        format!("WebGPU command lowering does not support {what}"),
    )
    .at("WebGpuCommandSpine::native_encode")
}
pub(crate) fn num(value: u64) -> JsValue {
    JsValue::from_f64(value as f64)
}
pub(crate) fn clear_color(value: crate::api::command::ColorClearValue) -> Object {
    let out = Object::new();
    match value {
        crate::api::command::ColorClearValue::Float([r, g, b, a]) => {
            let _ = set(&out, "r", &JsValue::from_f64(r as f64));
            let _ = set(&out, "g", &JsValue::from_f64(g as f64));
            let _ = set(&out, "b", &JsValue::from_f64(b as f64));
            let _ = set(&out, "a", &JsValue::from_f64(a as f64));
        }
        crate::api::command::ColorClearValue::Sint([r, g, b, a]) => {
            let _ = set(&out, "r", &JsValue::from_f64(r as f64));
            let _ = set(&out, "g", &JsValue::from_f64(g as f64));
            let _ = set(&out, "b", &JsValue::from_f64(b as f64));
            let _ = set(&out, "a", &JsValue::from_f64(a as f64));
        }
        crate::api::command::ColorClearValue::Uint([r, g, b, a]) => {
            let _ = set(&out, "r", &num(r as u64));
            let _ = set(&out, "g", &num(g as u64));
            let _ = set(&out, "b", &num(b as u64));
            let _ = set(&out, "a", &num(a as u64));
        }
        _ => {}
    }
    out
}
pub(crate) fn set(object: &Object, name: &str, value: &JsValue) -> Result<(), String> {
    Reflect::set(object, &JsValue::from_str(name), value)
        .map_err(|e| js::message(&e))
        .and_then(|ok| {
            ok.then_some(())
                .ok_or_else(|| "WebGPU descriptor field rejected".into())
        })
}
pub(crate) fn set_rhi(object: &Object, name: &str, value: JsValue) -> RhiResult<()> {
    set(object, name, &value).map_err(|message| {
        RhiError::new(RhiErrorKind::BackendFailure, message).at("WebGPU render-pass descriptor")
    })
}
pub(crate) fn call0(receiver: &JsValue, name: &str) -> Result<JsValue, String> {
    function(receiver, name)?
        .call0(receiver)
        .map_err(|e| js::message(&e))
}
pub(crate) fn call1(receiver: &JsValue, name: &str, a: &JsValue) -> Result<JsValue, String> {
    function(receiver, name)?
        .call1(receiver, a)
        .map_err(|e| js::message(&e))
}
pub(crate) fn call3(
    receiver: &JsValue,
    name: &str,
    a: &JsValue,
    b: &JsValue,
    c: &JsValue,
) -> Result<(), String> {
    function(receiver, name)?
        .call3(receiver, a, b, c)
        .map(|_| ())
        .map_err(|e| js::message(&e))
}
pub(crate) fn call3_value(
    receiver: &JsValue,
    name: &str,
    a: &JsValue,
    b: &JsValue,
    c: &JsValue,
) -> Result<JsValue, String> {
    function(receiver, name)?
        .call3(receiver, a, b, c)
        .map_err(|e| js::message(&e))
}
pub(crate) fn call5(
    receiver: &JsValue,
    name: &str,
    a: &JsValue,
    b: &JsValue,
    c: &JsValue,
    d: &JsValue,
    e: &JsValue,
) -> Result<(), String> {
    function(receiver, name)?
        .call5(receiver, a, b, c, d, e)
        .map(|_| ())
        .map_err(|e| js::message(&e))
}
pub(crate) fn function(receiver: &JsValue, name: &str) -> Result<Function, String> {
    js::property(receiver, name)
        .map_err(|e| js::message(&e))?
        .dyn_into::<Function>()
        .map_err(|e| js::message(&e))
}
pub(crate) fn read_mapped_bytes(buffer: &JsValue, bytes: u64) -> Option<Vec<u8>> {
    let get = function(buffer, "getMappedRange").ok()?;
    let range = get.call2(buffer, &num(0), &num(bytes)).ok()?;
    Some(js_sys::Uint8Array::new(&range).to_vec())
}
pub(crate) fn unmap_buffer(buffer: &JsValue) {
    if let Ok(unmap) = function(buffer, "unmap") {
        let _ = unmap.call0(buffer);
    }
}
